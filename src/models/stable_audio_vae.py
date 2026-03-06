"""
Stable Audio VAE Wrapper

Provides a clean interface for Stable Audio's pretrained VAE,
handling mono/stereo conversion and matching DAC's API.
"""

import torch
import torch.nn as nn
from stable_audio_tools import get_pretrained_model


class StableAudioVAE(nn.Module):
    """
    Wrapper around Stable Audio's pretrained VAE.
    
    Handles:
    - Mono to stereo conversion
    - Stereo to mono reconstruction
    - Consistent interface with DAC
    """
    
    def __init__(self, model_path="stabilityai/stable-audio-open-1.0", device='cpu'):
        """
        Initialize Stable Audio VAE.
        
        Args:
            model_path: Hugging Face model path
            device: Device to load model on
        """
        super().__init__()
        
        print(f"Loading Stable Audio VAE from {model_path}...")
        
        # Load pretrained model
        model, model_config = get_pretrained_model(model_path)
        
        # Extract just the VAE (pretransform)
        self.vae = model.pretransform
        self.vae.eval()  # Freeze by default
        
        # Move to device
        self.vae = self.vae.to(device)
        
        # Store config
        self.model_config = model_config
        self.latent_channels = 64  # Stable Audio uses 64 latent channels
        self.sample_rate = 44100
        
        print(f"✓ Stable Audio VAE loaded")
        print(f"  Latent channels: {self.latent_channels}")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
    def encode(self, audio):
        """
        Encode audio to latent representation.
        
        Args:
            audio: Audio tensor [batch, channels, samples]
                  Can be mono [batch, 1, samples] or stereo [batch, 2, samples]
        
        Returns:
            latents: Latent tensor [batch, 64, time]
        """
        # Handle mono input (convert to stereo)
        if audio.shape[1] == 1:
            # Duplicate mono to stereo
            audio_stereo = audio.repeat(1, 2, 1)
        elif audio.shape[1] == 2:
            # Already stereo
            audio_stereo = audio
        else:
            raise ValueError(f"Expected 1 or 2 channels, got {audio.shape[1]}")
        
        # Encode with Stable Audio VAE
        with torch.no_grad():
            latents = self.vae.encode(audio_stereo)
        
        return latents
    
    def decode(self, latents):
        """
        Decode latents back to audio.
        
        Args:
            latents: Latent tensor [batch, 64, time]
        
        Returns:
            audio: Reconstructed audio [batch, 1, samples] (mono)
        """
        # Decode to stereo
        with torch.no_grad():
            audio_stereo = self.vae.decode(latents)
        
        # Convert stereo to mono (average channels)
        audio_mono = audio_stereo.mean(dim=1, keepdim=True)
        
        return audio_mono
    
    def forward(self, audio):
        """
        Full encode-decode cycle (for testing).
        
        Args:
            audio: Input audio [batch, channels, samples]
        
        Returns:
            reconstructed: Reconstructed audio [batch, 1, samples]
        """
        latents = self.encode(audio)
        reconstructed = self.decode(latents)
        return reconstructed
    
    def normalize_latent(self, latents, mean=0.0, std=1.0):
        """
        Normalize latents (optional, for compatibility with diffusion).
        
        Stable Audio VAE latents are already reasonably normalized,
        but you can apply additional normalization if needed.
        """
        return (latents - mean) / std
    
    def denormalize_latent(self, latents_norm, mean=0.0, std=1.0):
        """
        Denormalize latents.
        """
        return latents_norm * std + mean


if __name__ == "__main__":
    # Test the wrapper
    print("Testing StableAudioVAE wrapper...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Create wrapper
    vae = StableAudioVAE(device=device)
    
    # Test with mono audio
    print("\nTesting mono audio...")
    mono_audio = torch.randn(2, 1, 176400).to(device)
    latents = vae.encode(mono_audio)
    reconstructed = vae.decode(latents)
    
    print(f"  Input: {mono_audio.shape}")
    print(f"  Latents: {latents.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    
    # Test with stereo audio
    print("\nTesting stereo audio...")
    stereo_audio = torch.randn(2, 2, 176400).to(device)
    latents = vae.encode(stereo_audio)
    reconstructed = vae.decode(latents)
    
    print(f"  Input: {stereo_audio.shape}")
    print(f"  Latents: {latents.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    
    print("\n✓ StableAudioVAE wrapper works correctly!")
