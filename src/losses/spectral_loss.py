"""
Spectral loss functions for audio reconstruction.

Implements multi-resolution STFT loss and perceptual audio metrics
for high-quality audio generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss.

    Compares spectrograms in both magnitude and phase (complex).
    """

    def __init__(self, n_fft=2048, hop_length=512, win_length=2048):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Register Hann window as buffer (not a parameter)
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, x, y):
        """
        Compute STFT loss between x and y.

        Args:
            x: (batch, channels, time) predicted audio
            y: (batch, channels, time) target audio

        Returns:
            STFT loss value
        """
        # Compute magnitude spectrograms (real-valued — avoids complex tensor ops on MPS)
        x_mag = self.magnitude(x)
        y_mag = self.magnitude(y)

        # L1 magnitude loss
        mag_loss = F.l1_loss(x_mag, y_mag)

        # Spectral convergence: ||Y - X||_F / ||Y||_F
        # Use .pow(2).sum().sqrt() instead of torch.norm(p='fro') — the latter
        # triggers a CPU fallback on MPS.
        # Add 1e-8 inside sqrt so the backward gradient stays finite even when
        # the norms are near zero (sqrt(0) backward = 1/(2*0) = Inf → NaN via clipping).
        diff_norm = (y_mag - x_mag).pow(2).sum().add(1e-8).sqrt()
        y_norm = y_mag.pow(2).sum().add(1e-8).sqrt() + 1e-8
        spec_conv_loss = diff_norm / y_norm

        return mag_loss + spec_conv_loss

    def magnitude(self, x):
        """
        Compute magnitude spectrogram without producing a complex tensor.

        Returns real-valued magnitude via view_as_real() to stay on MPS.
        torch.abs() on complex tensors falls back to CPU on MPS.
        """
        batch, channels, time = x.shape
        x = x.reshape(batch * channels, time)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            center=True,
        )  # (batch*channels, freq, time) complex

        # Convert to real without torch.abs() on complex:
        # view_as_real gives (..., 2) real tensor, then compute sqrt(r²+i²)
        stft_real = torch.view_as_real(stft)                    # (..., freq, time, 2)
        mag = stft_real.pow(2).sum(-1).clamp(min=1e-8).sqrt()  # (..., freq, time) real

        return mag.reshape(batch, channels, mag.shape[-2], mag.shape[-1])


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.

    Computes STFT loss at multiple time-frequency resolutions
    to capture both fine details and broader structure.
    """

    def __init__(self, resolutions=None):
        """
        Args:
            resolutions: List of (n_fft, hop_length, win_length) tuples.
                        If None, uses default set of resolutions.
        """
        super().__init__()

        if resolutions is None:
            # Two resolutions cover coarse + fine structure with half the STFT cost.
            # Restore all four once training speed is acceptable.
            resolutions = [
                (2048, 512, 2048),   # High freq resolution
                (512, 128, 512),     # Low freq resolution
            ]

        self.resolutions = resolutions

        # Create STFT loss for each resolution
        self.stft_losses = nn.ModuleList([
            STFTLoss(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            for n_fft, hop_length, win_length in resolutions
        ])

    def forward(self, x, y):
        """
        Compute multi-resolution STFT loss.

        Args:
            x: (batch, channels, time) predicted audio
            y: (batch, channels, time) target audio

        Returns:
            Average STFT loss across all resolutions
        """
        total_loss = 0.0

        for stft_loss in self.stft_losses:
            total_loss += stft_loss(x, y)

        # Average across resolutions
        return total_loss / len(self.stft_losses)


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for perceptual audio quality.

    Operates in mel-frequency scale which better matches human perception.
    """

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512,
                 win_length=2048, n_mels=128, f_min=0.0, f_max=None):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        # Register window
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

        # Pre-compute and register mel filterbank so it is built once,
        # not rebuilt (via slow Python loops) on every forward pass.
        self.register_buffer('mel_basis', self._build_mel_filterbank())

    def forward(self, x, y):
        """
        Compute mel-spectrogram loss.

        Args:
            x: (batch, channels, time) predicted audio
            y: (batch, channels, time) target audio

        Returns:
            L1 loss on mel-spectrograms
        """
        # Compute mel-spectrograms
        x_mel = self.mel_spectrogram(x)
        y_mel = self.mel_spectrogram(y)

        # L1 loss on log-mel spectrograms
        x_log_mel = torch.log(x_mel + 1e-5)
        y_log_mel = torch.log(y_mel + 1e-5)

        loss = F.l1_loss(x_log_mel, y_log_mel)

        return loss

    def mel_spectrogram(self, x):
        """
        Compute mel-spectrogram.

        Args:
            x: (batch, channels, time) audio

        Returns:
            (batch, channels, n_mels, time) mel-spectrogram
        """
        batch, channels, time = x.shape

        # Reshape for batched processing
        x = x.reshape(batch * channels, time)

        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            center=True,
        )

        # Magnitude without torch.abs() on complex (CPU fallback on MPS)
        stft_real = torch.view_as_real(stft)                         # (..., freq, time, 2)
        magnitude = stft_real.pow(2).sum(-1).clamp(min=1e-8).sqrt()  # (batch*channels, freq, time)

        # Apply pre-computed mel filterbank (registered buffer, moves with model device)
        mel_spec = torch.matmul(self.mel_basis, magnitude)  # (batch * channels, n_mels, time)

        # Reshape back
        mel_spec = mel_spec.reshape(batch, channels, self.n_mels, -1)

        return mel_spec

    def _build_mel_filterbank(self):
        """
        Build triangular mel filterbank matrix using vectorised NumPy ops.
        Called once in __init__ and stored as a buffer — not recomputed per batch.
        """
        import numpy as np

        n_freqs = self.n_fft // 2 + 1
        freqs = np.linspace(0, self.sample_rate / 2, n_freqs)

        mel_min = 2595 * np.log10(1 + self.f_min / 700)
        mel_max = 2595 * np.log10(1 + self.f_max / 700)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        filterbank = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for i in range(self.n_mels):
            left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            # Vectorised over the frequency axis — no inner Python loop
            rising = (freqs >= left) & (freqs <= center)
            falling = (freqs > center) & (freqs <= right)
            filterbank[i, rising] = (freqs[rising] - left) / (center - left + 1e-8)
            filterbank[i, falling] = (right - freqs[falling]) / (right - center + 1e-8)

        return torch.from_numpy(filterbank)


class AudioReconstructionLoss(nn.Module):
    """
    Complete audio reconstruction loss combining multiple metrics.

    Combines:
    - Time-domain L1 loss
    - Multi-resolution STFT loss
    - Mel-spectrogram loss (perceptual)
    """

    def __init__(self, sample_rate=44100, time_weight=1.0, stft_weight=1.0,
                 mel_weight=1.0):
        super().__init__()

        self.time_weight = time_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight

        # Multi-resolution STFT loss
        self.stft_loss = MultiResolutionSTFTLoss()

        # Mel-spectrogram loss
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)

    def forward(self, x, y):
        """
        Compute total reconstruction loss.

        Args:
            x: (batch, channels, time) predicted audio
            y: (batch, channels, time) target audio

        Returns:
            Dictionary with individual losses and total loss
        """
        # Time-domain L1 loss
        time_loss = F.l1_loss(x, y)

        # Spectral losses
        stft_loss = self.stft_loss(x, y)
        mel_loss = self.mel_loss(x, y)

        # Weighted combination
        total_loss = (
            self.time_weight * time_loss +
            self.stft_weight * stft_loss +
            self.mel_weight * mel_loss
        )

        return {
            'loss': total_loss,
            'time_loss': time_loss.item(),
            'stft_loss': stft_loss.item(),
            'mel_loss': mel_loss.item(),
        }


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral convergence loss.

    Measures relative error in spectrogram magnitude.
    """

    def __init__(self, n_fft=2048, hop_length=512, win_length=2048):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, x, y):
        """
        Compute spectral convergence.

        Args:
            x: Predicted audio
            y: Target audio

        Returns:
            Spectral convergence loss
        """
        # Compute magnitude spectrograms
        x_mag = self._magnitude_spectrogram(x)
        y_mag = self._magnitude_spectrogram(y)

        # Spectral convergence
        loss = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-8)

        return loss

    def _magnitude_spectrogram(self, x):
        """Compute magnitude spectrogram."""
        batch, channels, time = x.shape
        x = x.reshape(batch * channels, time)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

        magnitude = torch.view_as_real(stft).pow(2).sum(-1).clamp(min=1e-8).sqrt()
        return magnitude
