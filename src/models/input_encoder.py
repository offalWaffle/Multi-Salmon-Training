"""
Input Sample Encoder for conditioning the diffusion model.
Encodes 1-5 input audio samples into a single embedding vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputSampleEncoder(nn.Module):
    """
    Encode 1-5 input samples into a single embedding vector.
    This embedding conditions the diffusion model for timbral consistency.

    Two approaches available:
    1. 'mean': Simple average pooling (fast, works well)
    2. 'attention': Attention-based aggregation (more expressive)

    Supports both raw audio and pre-encoded DAC latents for faster training.
    """

    def __init__(
        self,
        embed_dim=128,
        sample_rate=44100,
        duration=4.0,
        aggregation='attention',
        num_heads=4,
        latent_dim=64,  # Latent dimension (64 for Stable Audio VAE, 1024 for DAC)
    ):
        """
        Args:
            embed_dim: Output embedding dimension
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
            aggregation: 'mean' or 'attention'
            num_heads: Number of attention heads (if using attention)
            latent_dim: Dimension of latents (64 for Stable Audio VAE, 1024 for DAC)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.duration = duration
        self.aggregation = aggregation
        self.num_samples = int(sample_rate * duration)
        self.latent_dim = latent_dim

        # CNN encoder for individual samples
        # Input: [batch, 1, num_samples] -> Output: [batch, embed_dim]
        self.encoder = nn.Sequential(
            # Layer 1: [batch, 1, 176400] -> [batch, 64, 11025]
            nn.Conv1d(1, 64, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Layer 2: [batch, 64, 11025] -> [batch, 128, 2756]
            nn.Conv1d(64, 128, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Layer 3: [batch, 128, 2756] -> [batch, 256, 689]
            nn.Conv1d(128, 256, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Layer 4: [batch, 256, 689] -> [batch, 512, 172]
            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Global average pooling: [batch, 512, 172] -> [batch, 512, 1] -> [batch, 512]
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Project to embedding dimension
            nn.Linear(512, embed_dim),
        )

        # CNN encoder for pre-encoded latents (much faster)
        # Input: [batch, latent_dim, time] -> Output: [batch, embed_dim]
        # Adaptive architecture based on latent_dim

        if latent_dim >= 512:
            # For large latents (e.g., DAC: 1024)
            # [batch, 1024, 345] -> [batch, 512, 86] -> [batch, 256, 21]
            hidden_dim1, hidden_dim2 = 512, 256
        else:
            # For small latents (e.g., Stable Audio VAE: 64)
            # [batch, 64, 86] -> [batch, 256, 21] -> [batch, 256, 5]
            hidden_dim1, hidden_dim2 = 256, 256

        self.latent_encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(latent_dim, hidden_dim1, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Project to embedding dimension
            nn.Linear(hidden_dim2, embed_dim),
        )

        # Aggregation mechanism for multiple samples
        if aggregation == 'attention':
            # Multi-head self-attention for aggregating embeddings
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            )

            # Learnable query for aggregation
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

        elif aggregation == 'mean':
            # Simple mean pooling (no additional parameters)
            pass
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def encode_sample(self, audio):
        """
        Encode a single audio sample.

        Args:
            audio: Audio waveform [batch, 1, num_samples]

        Returns:
            Embedding [batch, embed_dim]
        """
        return self.encoder(audio)

    def aggregate_embeddings(self, embeddings, num_valid):
        """
        Aggregate multiple sample embeddings into one.

        Args:
            embeddings: Stacked embeddings [batch, max_samples, embed_dim]
            num_valid: Number of valid samples per batch item [batch]

        Returns:
            Aggregated embedding [batch, embed_dim]
        """
        batch_size, max_samples, embed_dim = embeddings.shape

        if self.aggregation == 'mean':
            # Create mask for valid samples
            mask = torch.arange(max_samples, device=embeddings.device)[None, :] < num_valid[:, None]
            mask = mask.unsqueeze(-1).float()  # [batch, max_samples, 1]

            # Masked mean
            masked_embeddings = embeddings * mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            count = num_valid.unsqueeze(-1).float()
            aggregated = sum_embeddings / count.clamp(min=1)

        elif self.aggregation == 'attention':
            # Create attention mask for padding
            # True means "attend to this", False means "ignore"
            key_padding_mask = torch.arange(max_samples, device=embeddings.device)[None, :] >= num_valid[:, None]

            # Handle case where num_valid is 0 (shouldn't happen with dataset fix, but be safe)
            # If all keys are masked, use mean of all embeddings as fallback
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                # For samples with no valid inputs, use zero embedding
                aggregated = torch.zeros(batch_size, embed_dim, device=embeddings.device)

                # For samples with valid inputs, use attention
                valid_mask = ~all_masked
                if valid_mask.any():
                    # Expand query to batch size
                    query = self.query.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]

                    # Apply attention only to valid samples
                    valid_agg, _ = self.attention(
                        query[valid_mask],
                        embeddings[valid_mask],
                        embeddings[valid_mask],
                        key_padding_mask=key_padding_mask[valid_mask],
                    )
                    aggregated[valid_mask] = valid_agg.squeeze(1)
            else:
                # Normal case: all samples have at least one valid input
                query = self.query.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]

                # Apply attention: query attends to all sample embeddings
                aggregated, _ = self.attention(
                    query,
                    embeddings,
                    embeddings,
                    key_padding_mask=key_padding_mask,
                )

                # Remove sequence dimension
                aggregated = aggregated.squeeze(1)  # [batch, embed_dim]

        return aggregated

    def forward(self, input_samples, num_valid=None):
        """
        Encode 1-5 input samples into a single embedding.

        Args:
            input_samples: Tensor [batch, max_samples, 1, num_samples] or list of tensors
            num_valid: Number of valid samples per batch [batch]
                      If None, assumes all samples are valid

        Returns:
            Embedding [batch, embed_dim]
        """
        # Handle different input formats
        if isinstance(input_samples, list):
            # List of tensors: stack them
            max_samples = max(len(samples) for samples in input_samples)
            batch_size = len(input_samples)

            # Pad to same length
            padded_samples = torch.zeros(
                batch_size, max_samples, 1, self.num_samples,
                device=input_samples[0].device
            )
            num_valid = torch.zeros(batch_size, dtype=torch.long, device=input_samples[0].device)

            for i, samples in enumerate(input_samples):
                n = len(samples)
                padded_samples[i, :n] = samples
                num_valid[i] = n

            input_samples = padded_samples
        else:
            # Tensor input
            batch_size, max_samples, _, _ = input_samples.shape
            if num_valid is None:
                # Assume all samples are valid
                num_valid = torch.full((batch_size,), max_samples, dtype=torch.long, device=input_samples.device)

        # Encode each sample
        # Reshape: [batch, max_samples, 1, num_samples] -> [batch * max_samples, 1, num_samples]
        batch_size, max_samples, _, num_samples = input_samples.shape
        flat_samples = input_samples.reshape(batch_size * max_samples, 1, num_samples)

        # Encode all samples at once
        flat_embeddings = self.encode_sample(flat_samples)  # [batch * max_samples, embed_dim]

        # Reshape back: [batch * max_samples, embed_dim] -> [batch, max_samples, embed_dim]
        embeddings = flat_embeddings.reshape(batch_size, max_samples, self.embed_dim)

        # Aggregate embeddings
        aggregated = self.aggregate_embeddings(embeddings, num_valid)

        return aggregated

    def forward_latents(self, input_latents, num_valid=None):
        """
        Encode 1-5 pre-encoded DAC latents into a single embedding.
        Much faster than forward() as DAC encoding is already done.

        Args:
            input_latents: Tensor [batch, max_samples, 1024, time]
            num_valid: Number of valid samples per batch [batch]
                      If None, assumes all samples are valid

        Returns:
            Embedding [batch, embed_dim]
        """
        batch_size, max_samples, latent_dim, time_steps = input_latents.shape

        if num_valid is None:
            # Assume all samples are valid
            num_valid = torch.full((batch_size,), max_samples, dtype=torch.long, device=input_latents.device)

        # Encode each latent
        # Reshape: [batch, max_samples, 1024, time] -> [batch * max_samples, 1024, time]
        flat_latents = input_latents.reshape(batch_size * max_samples, latent_dim, time_steps)

        # Encode all latents at once using latent encoder
        flat_embeddings = self.latent_encoder(flat_latents)  # [batch * max_samples, embed_dim]

        # Reshape back: [batch * max_samples, embed_dim] -> [batch, max_samples, embed_dim]
        embeddings = flat_embeddings.reshape(batch_size, max_samples, self.embed_dim)

        # Aggregate embeddings
        aggregated = self.aggregate_embeddings(embeddings, num_valid)

        return aggregated


class SimpleInputEncoder(nn.Module):
    """
    Simplified input encoder that uses DAC's encoder instead of custom CNN.
    More efficient if DAC is already loaded.
    """

    def __init__(
        self,
        dac_model,
        embed_dim=128,
        aggregation='mean',
    ):
        """
        Args:
            dac_model: Pre-trained DAC model
            embed_dim: Output embedding dimension
            aggregation: 'mean' or 'attention'
        """
        super().__init__()

        self.dac_model = dac_model
        self.embed_dim = embed_dim
        self.aggregation = aggregation

        # Project DAC latent to embedding
        # DAC latent is [batch, 1024, time]
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [batch, 1024, time] -> [batch, 1024, 1]
            nn.Flatten(),  # [batch, 1024]
            nn.Linear(1024, embed_dim),
        )

        if aggregation == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                batch_first=True,
            )
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, input_samples, num_valid=None):
        """
        Encode input samples using DAC.

        Args:
            input_samples: [batch, max_samples, 1, num_samples]
            num_valid: [batch] number of valid samples

        Returns:
            Embedding [batch, embed_dim]
        """
        batch_size, max_samples, _, num_samples = input_samples.shape

        if num_valid is None:
            num_valid = torch.full((batch_size,), max_samples, dtype=torch.long, device=input_samples.device)

        # Encode with DAC/StableAudioVAE
        flat_samples = input_samples.reshape(batch_size * max_samples, 1, num_samples)

        with torch.no_grad():
            z = self.dac_model.encode(flat_samples)

        # Project to embedding
        embeddings = self.projection(z)  # [batch * max_samples, embed_dim]
        embeddings = embeddings.reshape(batch_size, max_samples, self.embed_dim)

        # Aggregate
        if self.aggregation == 'mean':
            mask = torch.arange(max_samples, device=embeddings.device)[None, :] < num_valid[:, None]
            mask = mask.unsqueeze(-1).float()
            masked_embeddings = embeddings * mask
            aggregated = masked_embeddings.sum(dim=1) / num_valid.unsqueeze(-1).float().clamp(min=1)
        else:
            key_padding_mask = torch.arange(max_samples, device=embeddings.device)[None, :] >= num_valid[:, None]
            query = self.query.expand(batch_size, -1, -1)
            aggregated, _ = self.attention(query, embeddings, embeddings, key_padding_mask=key_padding_mask)
            aggregated = aggregated.squeeze(1)

        return aggregated


if __name__ == "__main__":
    print("Testing InputSampleEncoder...")

    device = 'cpu'
    sample_rate = 44100
    duration = 4.0
    num_samples = int(sample_rate * duration)

    # Test with mean aggregation
    print("\n1. Testing mean aggregation:")
    encoder_mean = InputSampleEncoder(
        embed_dim=128,
        sample_rate=sample_rate,
        duration=duration,
        aggregation='mean',
    ).to(device)

    # Test with varying number of input samples
    for num_inputs in [1, 3, 5]:
        batch_size = 2
        max_samples = 5

        # Create input with padding
        input_samples = torch.randn(batch_size, max_samples, 1, num_samples).to(device)
        num_valid = torch.tensor([num_inputs, num_inputs], dtype=torch.long).to(device)

        embedding = encoder_mean(input_samples, num_valid)
        print(f"   {num_inputs} samples -> embedding shape: {embedding.shape}")
        assert embedding.shape == (batch_size, 128), f"Expected shape (2, 128), got {embedding.shape}"

    print("   ✓ Mean aggregation works!")

    # Test with attention aggregation
    print("\n2. Testing attention aggregation:")
    encoder_attn = InputSampleEncoder(
        embed_dim=128,
        sample_rate=sample_rate,
        duration=duration,
        aggregation='attention',
        num_heads=4,
    ).to(device)

    for num_inputs in [1, 3, 5]:
        batch_size = 2
        max_samples = 5

        input_samples = torch.randn(batch_size, max_samples, 1, num_samples).to(device)
        num_valid = torch.tensor([num_inputs, num_inputs], dtype=torch.long).to(device)

        embedding = encoder_attn(input_samples, num_valid)
        print(f"   {num_inputs} samples -> embedding shape: {embedding.shape}")
        assert embedding.shape == (batch_size, 128), f"Expected shape (2, 128), got {embedding.shape}"

    print("   ✓ Attention aggregation works!")

    # Test list input format
    print("\n3. Testing list input format:")
    input_list = [
        torch.randn(3, 1, num_samples).to(device),  # 3 samples
        torch.randn(1, 1, num_samples).to(device),  # 1 sample
    ]

    embedding = encoder_mean(input_list)
    print(f"   List input -> embedding shape: {embedding.shape}")
    assert embedding.shape == (2, 128), f"Expected shape (2, 128), got {embedding.shape}"
    print("   ✓ List input format works!")

    print("\n✓ All tests passed!")
