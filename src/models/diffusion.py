"""
Latent Diffusion Model for audio generation.
Operates in DAC latent space [batch, 1024, 345].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model for audio generation.
    Operates in DAC latent space, conditioned on MIDI note and input samples.
    """

    def __init__(
        self,
        unet,
        dac_model,
        num_timesteps=1000,
        beta_schedule='cosine',
        beta_start=0.0001,
        beta_end=0.02,
        cosine_s=0.008,
        prediction_type='v_prediction',  # 'epsilon' or 'v_prediction'
        latent_mean=0.0,
        latent_std=1.0,
        timestep_weighting=False,  # Whether to use weighted timestep sampling
        timestep_weight_min=200,   # Min timestep for weighted sampling
        timestep_weight_max=800,   # Max timestep for weighted sampling
    ):
        """
        Args:
            unet: ConditionalUNet model for noise prediction
            dac_model: DAC model for encoding/decoding audio
            num_timesteps: Number of diffusion timesteps
            beta_schedule: 'cosine' or 'linear'
            beta_start: Start value for linear schedule
            beta_end: End value for linear schedule
            cosine_s: Offset for cosine schedule
            prediction_type: 'epsilon' (predict noise) or 'v_prediction' (predict velocity)
            latent_mean: Mean of DAC latents (for normalization)
            latent_std: Std of DAC latents (for normalization)
            timestep_weighting: If True, focus training on mid-range timesteps
            timestep_weight_min: Minimum timestep for weighted range
            timestep_weight_max: Maximum timestep for weighted range
        """
        super().__init__()

        self.unet = unet
        self.dac_model = dac_model
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.timestep_weighting = timestep_weighting
        self.timestep_weight_min = timestep_weight_min
        self.timestep_weight_max = timestep_weight_max

        # Latent normalization parameters
        self.register_buffer('latent_mean', torch.tensor(latent_mean))
        self.register_buffer('latent_std', torch.tensor(latent_std))

        # Create noise schedule
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, s=cosine_s)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # Pre-compute diffusion constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
        More stable than linear schedule.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at specified timesteps.

        Args:
            a: Tensor of coefficients
            t: Timestep indices [batch]
            x_shape: Shape of x for broadcasting

        Returns:
            Coefficients with shape [batch, 1, 1] for broadcasting
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def normalize_latent(self, z):
        """
        Normalize DAC latent to mean=0, std=1.
        This is critical for diffusion to work properly.
        """
        return (z - self.latent_mean) / self.latent_std

    def denormalize_latent(self, z_norm):
        """
        Denormalize latent back to original DAC scale.
        """
        return z_norm * self.latent_std + self.latent_mean

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to latent.
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod) * x_0, (1 - alpha_cumprod) * I)

        Args:
            x_0: Original latent [batch, channels, time]
            t: Timestep indices [batch]
            noise: Optional noise tensor (generated if None)

        Returns:
            Noisy latent x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise.
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t, midi_note, input_embedding, clip_denoised=True):
        """
        Compute mean and variance for reverse diffusion step p(x_{t-1} | x_t).

        Args:
            x_t: Current noisy latent [batch, channels, time]
            t: Timestep indices [batch]
            midi_note: MIDI note conditioning [batch]
            input_embedding: Input sample embedding [batch, dim]
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            Tuple of (mean, variance, log_variance)
        """
        # Predict noise or velocity
        model_output = self.unet(x_t, t, midi_note, input_embedding)

        # Convert prediction to x_0
        if self.prediction_type == 'epsilon':
            # Model predicts noise
            x_0_pred = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == 'v_prediction':
            # Model predicts velocity (v-parameterization)
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            x_0_pred = sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Clip predicted x_0 to normalized range
        # After normalization, latents are mean=0, std=1, so clip to ±3 std
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -3.0, 3.0)

        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_0_pred, x_t, t
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, midi_note, input_embedding, clip_denoised=True):
        """
        Reverse diffusion: denoise one step p(x_{t-1} | x_t).

        Args:
            x_t: Current noisy latent [batch, channels, time]
            t: Timestep indices [batch]
            midi_note: MIDI note conditioning [batch]
            input_embedding: Input sample embedding [batch, dim]
            clip_denoised: Whether to clip predictions

        Returns:
            Previous step latent x_{t-1}
        """
        # Get mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, t, midi_note, input_embedding, clip_denoised=clip_denoised
        )

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        # Sample x_{t-1}
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        midi_notes,
        input_embeddings,
        latent_shape=(1024, 345),
        progress=True,
        return_all_timesteps=False,
    ):
        """
        Generate samples through reverse diffusion (DDPM sampling).

        Args:
            batch_size: Number of samples to generate
            midi_notes: Target MIDI notes [batch]
            input_embeddings: Embeddings from input samples [batch, dim]
            latent_shape: Shape of latent (channels, time)
            progress: Whether to show progress bar
            return_all_timesteps: Whether to return all intermediate steps

        Returns:
            Generated latents [batch, channels, time]
            Or list of all timesteps if return_all_timesteps=True
        """
        device = next(self.unet.parameters()).device

        # Start from pure noise
        x_t = torch.randn(batch_size, *latent_shape).to(device)

        if return_all_timesteps:
            all_steps = [x_t]

        # Reverse diffusion process
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, t_batch, midi_notes, input_embeddings)

            if return_all_timesteps:
                all_steps.append(x_t)

        if return_all_timesteps:
            return all_steps
        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size,
        midi_notes,
        input_embeddings,
        latent_shape=(1024, 345),
        num_inference_steps=50,
        eta=0.0,
        progress=True,
        guidance_scale=1.0,
    ):
        """
        DDIM sampling (faster than DDPM) with Classifier-Free Guidance.

        Args:
            batch_size: Number of samples to generate
            midi_notes: Target MIDI notes [batch]
            input_embeddings: Embeddings from input samples [batch, dim]
            latent_shape: Shape of latent (channels, time)
            num_inference_steps: Number of sampling steps (< num_timesteps for speedup)
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            progress: Whether to show progress bar
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance, 3-7 = typical, higher = stronger)

        Returns:
            Generated latents [batch, channels, time]
        """
        device = next(self.unet.parameters()).device

        # Create subset of timesteps
        step_size = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size).to(device)
        timesteps = torch.flip(timesteps, [0])

        # Start from pure noise
        x_t = torch.randn(batch_size, *latent_shape).to(device)

        # Create iterator with optional progress bar
        timestep_iterator = tqdm(timesteps, desc="DDIM Sampling") if progress else timesteps

        for i, t in enumerate(timestep_iterator):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            # Classifier-Free Guidance: Run model with and without conditioning
            if guidance_scale != 1.0:
                # Unconditional prediction (no conditioning)
                uncond_midi = torch.zeros_like(midi_notes)
                uncond_emb = torch.zeros_like(input_embeddings)
                model_output_uncond = self.unet(x_t, t_batch, uncond_midi, uncond_emb)

                # Conditional prediction (with conditioning)
                model_output_cond = self.unet(x_t, t_batch, midi_notes, input_embeddings)

                # Apply guidance: amplify the difference between conditional and unconditional
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            else:
                # No guidance - just predict with conditioning
                model_output = self.unet(x_t, t_batch, midi_notes, input_embeddings)

            # Predict x_0 and epsilon (noise)
            if self.prediction_type == 'epsilon':
                # Model predicts noise directly
                epsilon_pred = model_output
                x_0_pred = self.predict_start_from_noise(x_t, t_batch, model_output)
            else:
                # Model predicts velocity - need to convert to both x_0 and epsilon
                sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t_batch, x_t.shape)
                sqrt_one_minus_alphas_cumprod_t = self._extract(
                    self.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape
                )
                # Convert velocity to x_0
                x_0_pred = sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
                # Convert velocity to epsilon: epsilon = sqrt(1 - alpha_t) * x_t + sqrt(alpha_t) * v
                epsilon_pred = sqrt_one_minus_alphas_cumprod_t * x_t + sqrt_alphas_cumprod_t * model_output

            # Clip predicted x_0 to normalized range
            x_0_pred = torch.clamp(x_0_pred, -3.0, 3.0)

            # Get next timestep (use original timesteps tensor, not iterator)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                t_next = torch.tensor(0, device=device)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0)

            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
            )

            # Direction pointing to x_t (use predicted noise, not model_output!)
            pred_sample_direction = torch.sqrt(1 - alpha_t_next - sigma_t ** 2) * epsilon_pred

            # Sample
            x_t = torch.sqrt(alpha_t_next) * x_0_pred + pred_sample_direction

            if sigma_t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise

        return x_t

    def forward(self, audio, midi_notes, input_embeddings, return_timesteps=False):
        """
        Training forward pass: compute diffusion loss.

        DEPRECATED: Use forward_latents() for faster training with pre-encoded latents.

        Args:
            audio: Input audio waveforms [batch, 1, samples]
            midi_notes: MIDI note labels [batch]
            input_embeddings: Embeddings from input samples [batch, dim]
            return_timesteps: If True, return (loss, timesteps) tuple

        Returns:
            Diffusion loss (MSE between predicted and actual noise/velocity)
            or (loss, timesteps) if return_timesteps=True
        """
        batch_size = audio.shape[0]
        device = audio.device

        # Encode audio to latent using DAC/StableAudioVAE
        with torch.no_grad():
            # StableAudioVAE returns just latents, DAC returns (z, codes, latents, commitment_loss, codebook_loss)
            z = self.dac_model.encode(audio)
            # z is the continuous latent [batch, channels, time]

            # CRITICAL: Normalize latent to mean=0, std=1 for proper diffusion
            z = self.normalize_latent(z)

        # Sample random timesteps (with optional weighting)
        if self.timestep_weighting:
            # Focus on harder mid-range timesteps
            t = torch.randint(
                self.timestep_weight_min,
                self.timestep_weight_max,
                (batch_size,),
                device=device
            ).long()
        else:
            # Uniform sampling across all timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(z)

        # Add noise to latent (forward diffusion)
        z_t = self.q_sample(z, t, noise=noise)

        # Predict noise or velocity
        model_output = self.unet(z_t, t, midi_notes, input_embeddings)

        # Compute loss based on prediction type
        if self.prediction_type == 'epsilon':
            # Simple noise prediction loss
            loss = F.mse_loss(model_output, noise, reduction='none')
            # Average over all dimensions except batch
            loss_per_sample = loss.mean(dim=[1, 2])
            loss = loss_per_sample.mean()
        elif self.prediction_type == 'v_prediction':
            # V-prediction loss
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, z.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, z.shape
            )
            v_target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * z
            loss = F.mse_loss(model_output, v_target, reduction='none')
            loss_per_sample = loss.mean(dim=[1, 2])
            loss = loss_per_sample.mean()
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if return_timesteps:
            return loss, t, loss_per_sample
        return loss

    def forward_latents(self, latents, midi_notes, input_embeddings, return_timesteps=False):
        """
        Training forward pass with pre-encoded latents.
        Much faster than forward() as it skips DAC encoding.

        Args:
            latents: Pre-encoded DAC latents [batch, 1024, time]
            midi_notes: MIDI note labels [batch]
            input_embeddings: Embeddings from input samples [batch, dim]
            return_timesteps: If True, return (loss, timesteps, loss_per_sample) tuple

        Returns:
            Diffusion loss (MSE between predicted and actual noise/velocity)
            or (loss, timesteps, loss_per_sample) if return_timesteps=True
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Normalize latents
        z = self.normalize_latent(latents)

        # Sample random timesteps (with optional weighting)
        if self.timestep_weighting:
            # Focus on harder mid-range timesteps (e.g., 200-800)
            # These are neither pure noise nor mostly clean, so they're most informative
            t = torch.randint(
                self.timestep_weight_min,
                self.timestep_weight_max,
                (batch_size,),
                device=device
            ).long()
        else:
            # Uniform sampling across all timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(z)

        # Add noise to latent (forward diffusion)
        z_t = self.q_sample(z, t, noise=noise)

        # Predict noise or velocity
        model_output = self.unet(z_t, t, midi_notes, input_embeddings)

        # Compute loss based on prediction type
        if self.prediction_type == 'epsilon':
            # Simple noise prediction loss
            loss = F.mse_loss(model_output, noise, reduction='none')
            # Average over all dimensions except batch
            loss_per_sample = loss.mean(dim=[1, 2])
            loss = loss_per_sample.mean()
        elif self.prediction_type == 'v_prediction':
            # V-prediction loss
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, z.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, z.shape
            )
            v_target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * z
            loss = F.mse_loss(model_output, v_target, reduction='none')
            loss_per_sample = loss.mean(dim=[1, 2])
            loss = loss_per_sample.mean()
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if return_timesteps:
            return loss, t, loss_per_sample
        return loss


if __name__ == "__main__":
    # Test the diffusion model
    from src.models.unet import ConditionalUNet

    print("Testing LatentDiffusion...")

    device = 'cpu'

    # Create U-Net
    unet = ConditionalUNet(
        in_channels=1024,
        model_channels=128,  # Smaller for testing
        out_channels=1024,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        embed_dim=256,
        input_embedding_dim=128,
    ).to(device)

    # Create mock DAC model
    class MockDAC(nn.Module):
        def encode(self, x):
            # Return mock latent [batch, 1024, 345]
            batch_size = x.shape[0]
            z = torch.randn(batch_size, 1024, 345)
            return z, None, None, 0.0, 0.0

    dac_model = MockDAC()

    # Create diffusion model
    diffusion = LatentDiffusion(
        unet=unet,
        dac_model=dac_model,
        num_timesteps=1000,
        beta_schedule='cosine',
    ).to(device)

    print(f"Number of timesteps: {diffusion.num_timesteps}")
    print(f"Beta range: {diffusion.betas.min():.4f} - {diffusion.betas.max():.4f}")

    # Test forward pass (training)
    batch_size = 2
    audio = torch.randn(batch_size, 1, 176400).to(device)  # 4 seconds at 44.1kHz
    midi_notes = torch.rand(batch_size).to(device)
    velocities = torch.rand(batch_size).to(device)
    input_embeddings = torch.randn(batch_size, 128).to(device)

    print(f"\nInput audio shape: {audio.shape}")
    loss = diffusion(audio, midi_notes, velocities, input_embeddings)
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training forward pass successful!")

    # Test sampling
    print("\nTesting sampling (5 steps)...")
    diffusion.num_timesteps = 5  # Use fewer steps for testing
    generated = diffusion.sample(
        batch_size=2,
        midi_notes=midi_notes,
        velocities=velocities,
        input_embeddings=input_embeddings,
        latent_shape=(1024, 345),
        progress=False,
    )
    print(f"Generated latent shape: {generated.shape}")
    print("✓ Sampling successful!")
