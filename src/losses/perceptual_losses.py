"""
Perceptual audio losses for high-quality reconstruction.

These losses operate in the audio domain and preserve characteristics
important for human perception, particularly for piano sounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for audio quality.

    Captures both transients (short windows) and harmonics (long windows).
    Standard in high-quality audio synthesis (HiFi-GAN, etc.)
    """

    def __init__(
        self,
        fft_sizes=[2048, 1024, 512, 256, 128],
        hop_sizes=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        window='hann'
    ):
        """
        Initialize multi-scale STFT loss.

        Args:
            fft_sizes: List of FFT sizes for different scales
            hop_sizes: List of hop sizes
            win_lengths: List of window lengths
            window: Window type ('hann', 'hamming', etc.)
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT."""
        window = torch.hann_window(win_length).to(x.device)

        # Ensure correct input shape [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # STFT expects [B, T]
        x = x.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            normalized=True
        )

        return spec

    def forward(self, pred, target):
        """
        Compute multi-scale STFT loss.

        Args:
            pred: Predicted audio [B, 1, T] or [B, T]
            target: Target audio [B, 1, T] or [B, T]

        Returns:
            Total STFT loss across all scales
        """
        total_loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFT
            pred_spec = self.stft(pred, fft_size, hop_size, win_length)
            target_spec = self.stft(target, fft_size, hop_size, win_length)

            # Magnitude loss
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)

            mag_loss = F.l1_loss(pred_mag, target_mag)

            # Log magnitude loss (emphasizes quieter components)
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-5),
                torch.log(target_mag + 1e-5)
            )

            total_loss += mag_loss + log_mag_loss

        # Average across scales
        return total_loss / len(self.fft_sizes)


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss with perceptual weighting.

    Emphasizes frequencies important for human perception.
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=0.0,
        f_max=None
    ):
        """
        Initialize mel-spectrogram loss.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency (None = sr/2)
        """
        super().__init__()

        if f_max is None:
            f_max = sample_rate / 2.0

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

    def forward(self, pred, target):
        """
        Compute mel-spectrogram loss.

        Args:
            pred: Predicted audio [B, 1, T] or [B, T]
            target: Target audio [B, 1, T] or [B, T]

        Returns:
            Mel-spectrogram L1 loss
        """
        # Ensure [B, 1, T] format
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        # Move mel_transform to correct device
        self.mel_transform = self.mel_transform.to(pred.device)

        # Compute mel spectrograms
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)

        # Log mel spectrograms (perceptually weighted)
        pred_log_mel = torch.log(pred_mel + 1e-5)
        target_log_mel = torch.log(target_mel + 1e-5)

        return F.l1_loss(pred_log_mel, target_log_mel)


class PerceptualAudioLoss(nn.Module):
    """
    Combined perceptual audio loss for high-quality reconstruction.

    Combines multiple loss components optimized for piano quality:
    - Multi-scale STFT loss (transients + harmonics)
    - Mel-spectrogram loss (perceptual weighting)
    - Time-domain L1 loss (waveform preservation)
    - Optional latent MSE (regularization)
    """

    def __init__(
        self,
        sample_rate=44100,
        stft_weight=1.0,
        mel_weight=1.0,
        time_weight=0.5,
        latent_weight=0.1
    ):
        """
        Initialize perceptual audio loss.

        Args:
            sample_rate: Audio sample rate
            stft_weight: Weight for multi-scale STFT loss
            mel_weight: Weight for mel-spectrogram loss
            time_weight: Weight for time-domain L1 loss
            latent_weight: Weight for latent space regularization
        """
        super().__init__()

        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.time_weight = time_weight
        self.latent_weight = latent_weight

        # Loss components
        self.stft_loss = MultiScaleSTFTLoss()
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)

    def forward(
        self,
        pred_audio,
        target_audio,
        pred_latent=None,
        target_latent=None
    ):
        """
        Compute perceptual audio loss.

        Args:
            pred_audio: Predicted audio [B, 1, T]
            target_audio: Target audio [B, 1, T]
            pred_latent: Optional predicted latent [B, C, T] for regularization
            target_latent: Optional target latent [B, C, T] for regularization

        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}

        # Multi-scale STFT loss (captures transients and harmonics)
        if self.stft_weight > 0:
            stft_loss = self.stft_loss(pred_audio, target_audio)
            losses['stft'] = stft_loss
        else:
            stft_loss = 0.0

        # Mel-spectrogram loss (perceptually weighted)
        if self.mel_weight > 0:
            mel_loss = self.mel_loss(pred_audio, target_audio)
            losses['mel'] = mel_loss
        else:
            mel_loss = 0.0

        # Time-domain L1 loss (preserves waveform details)
        if self.time_weight > 0:
            time_loss = F.l1_loss(pred_audio, target_audio)
            losses['time'] = time_loss
        else:
            time_loss = 0.0

        # Latent space regularization (optional, keeps model stable)
        if self.latent_weight > 0 and pred_latent is not None and target_latent is not None:
            latent_loss = F.mse_loss(pred_latent, target_latent)
            losses['latent'] = latent_loss
        else:
            latent_loss = 0.0

        # Weighted total loss
        total_loss = (
            self.stft_weight * stft_loss +
            self.mel_weight * mel_loss +
            self.time_weight * time_loss +
            self.latent_weight * latent_loss
        )

        losses['total'] = total_loss

        return losses


# Quick test
if __name__ == "__main__":
    print("Testing perceptual audio losses...")

    # Create dummy audio
    batch_size = 2
    audio_length = 44100 * 2  # 2 seconds

    pred_audio = torch.randn(batch_size, 1, audio_length)
    target_audio = torch.randn(batch_size, 1, audio_length)

    # Test individual losses
    print("\n1. Multi-scale STFT Loss:")
    stft_loss = MultiScaleSTFTLoss()
    loss = stft_loss(pred_audio, target_audio)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. Mel-spectrogram Loss:")
    mel_loss = MelSpectrogramLoss()
    loss = mel_loss(pred_audio, target_audio)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. Combined Perceptual Loss:")
    perceptual_loss = PerceptualAudioLoss()
    losses = perceptual_loss(pred_audio, target_audio)
    print(f"   Total: {losses['total'].item():.4f}")
    print(f"   STFT:  {losses['stft'].item():.4f}")
    print(f"   Mel:   {losses['mel'].item():.4f}")
    print(f"   Time:  {losses['time'].item():.4f}")

    print("\n✅ All tests passed!")
