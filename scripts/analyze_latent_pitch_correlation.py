"""Analyze correlation between latent dimensions and pitch."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_latents_and_labels(data_dir, max_samples=None):
    """
    Load all latents and their corresponding MIDI notes.

    Args:
        data_dir: Directory containing latent .pt files
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        latents: [N, 64, 86] - All latent tensors
        midi_notes: [N] - Corresponding MIDI notes
        velocities: [N] - Velocities
    """
    latent_files = sorted(Path(data_dir).glob('sample_*.pt'))

    if max_samples:
        latent_files = latent_files[:max_samples]

    latents_list = []
    midi_notes_list = []
    velocities_list = []

    print(f"Loading {len(latent_files)} latent files...")

    for latent_file in tqdm(latent_files):
        data = torch.load(latent_file, map_location='cpu')

        latents_list.append(data['latent'])
        midi_notes_list.append(data['midi_note'])
        velocities_list.append(data['velocity'])

    # Stack into tensors
    latents = torch.stack(latents_list)  # [N, 64, 86]
    midi_notes = torch.tensor(midi_notes_list, dtype=torch.long)  # [N]
    velocities = torch.tensor(velocities_list, dtype=torch.long)  # [N]

    return latents, midi_notes, velocities


def compute_pitch_correlations(latents, midi_notes):
    """
    Compute correlation between each latent dimension and pitch.

    Args:
        latents: [N, C, T] - Latent tensors
        midi_notes: [N] - MIDI notes

    Returns:
        channel_correlations: [C] - Correlation for each channel (averaged over time)
        channel_time_correlations: [C, T] - Correlation for each channel and time step
    """
    N, C, T = latents.shape

    # Convert to numpy
    latents_np = latents.numpy()  # [N, C, T]
    midi_np = midi_notes.numpy()  # [N]

    # Compute correlation for each channel and time step
    channel_time_correlations = np.zeros((C, T))

    print("Computing pitch correlations for each dimension...")

    for c in tqdm(range(C)):
        for t in range(T):
            # Get latent values for this channel and time
            values = latents_np[:, c, t]

            # Compute correlation with MIDI notes
            correlation = np.corrcoef(values, midi_np)[0, 1]

            # Handle NaN (constant values)
            if np.isnan(correlation):
                correlation = 0.0

            channel_time_correlations[c, t] = correlation

    # Average over time to get per-channel correlation
    channel_correlations = np.abs(channel_time_correlations).mean(axis=1)

    return channel_correlations, channel_time_correlations


def plot_channel_correlations(channel_correlations, output_path):
    """Plot correlation strength for each channel."""
    fig, ax = plt.subplots(figsize=(12, 6))

    channels = np.arange(len(channel_correlations))

    # Bar plot
    bars = ax.bar(channels, channel_correlations, color='steelblue', alpha=0.7)

    # Highlight top 10 channels
    top_indices = np.argsort(channel_correlations)[-10:]
    for idx in top_indices:
        bars[idx].set_color('coral')

    ax.set_xlabel('Latent Channel', fontsize=12)
    ax.set_ylabel('Absolute Correlation with Pitch', fontsize=12)
    ax.set_title('Pitch Correlation by Latent Channel (Top 10 Highlighted)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved channel correlations plot to {output_path}")
    plt.close()


def plot_correlation_heatmap(channel_time_correlations, output_path):
    """Plot heatmap of correlations across channels and time."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    sns.heatmap(
        channel_time_correlations,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation with Pitch'},
        ax=ax
    )

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Latent Channel', fontsize=12)
    ax.set_title('Pitch Correlation Heatmap (Channel × Time)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation heatmap to {output_path}")
    plt.close()


def plot_top_channels_evolution(latents, midi_notes, channel_correlations, output_path, top_k=5):
    """Plot evolution of top correlated channels across different pitches."""
    # Get top channels
    top_indices = np.argsort(channel_correlations)[-top_k:][::-1]

    # Sort samples by MIDI note
    sorted_indices = torch.argsort(midi_notes)
    latents_sorted = latents[sorted_indices]  # [N, C, T]
    midi_sorted = midi_notes[sorted_indices]  # [N]

    # Plot
    fig, axes = plt.subplots(top_k, 1, figsize=(14, 2.5 * top_k))

    if top_k == 1:
        axes = [axes]

    for i, channel_idx in enumerate(top_indices):
        ax = axes[i]

        # Get channel values across all samples and time steps
        channel_values = latents_sorted[:, channel_idx, :].numpy()  # [N, T]

        # Plot as image (samples × time)
        im = ax.imshow(
            channel_values,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Latent Value')

        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Sample (sorted by pitch)', fontsize=10)
        ax.set_title(
            f'Channel {channel_idx} (Correlation: {channel_correlations[channel_idx]:.3f})',
            fontsize=11
        )

        # Add MIDI note labels on y-axis
        num_ticks = min(10, len(midi_sorted))
        tick_indices = np.linspace(0, len(midi_sorted) - 1, num_ticks, dtype=int)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([f"MIDI {midi_sorted[idx]}" for idx in tick_indices])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved top channels evolution to {output_path}")
    plt.close()


def plot_channel_vs_pitch(latents, midi_notes, channel_idx, output_path):
    """Scatter plot of a specific channel's values vs pitch."""
    # Average over time for each sample
    channel_values = latents[:, channel_idx, :].mean(dim=1).numpy()  # [N]
    midi_np = midi_notes.numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(midi_np, channel_values, alpha=0.5, s=30)

    # Fit linear regression
    coeffs = np.polyfit(midi_np, channel_values, 1)
    fit_line = np.poly1d(coeffs)
    midi_range = np.linspace(midi_np.min(), midi_np.max(), 100)
    ax.plot(midi_range, fit_line(midi_range), 'r--', linewidth=2, label='Linear Fit')

    # Compute R²
    correlation = np.corrcoef(midi_np, channel_values)[0, 1]
    r_squared = correlation ** 2

    ax.set_xlabel('MIDI Note', fontsize=12)
    ax.set_ylabel(f'Channel {channel_idx} Value (time-averaged)', fontsize=12)
    ax.set_title(f'Channel {channel_idx} vs Pitch (R² = {r_squared:.3f})', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved channel vs pitch plot to {output_path}")
    plt.close()


def print_analysis_summary(channel_correlations, midi_notes):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("LATENT-PITCH CORRELATION ANALYSIS")
    print("=" * 60)

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(midi_notes)}")
    print(f"  MIDI range: {midi_notes.min()} - {midi_notes.max()}")
    print(f"  Latent channels: {len(channel_correlations)}")

    print(f"\nCorrelation Statistics:")
    print(f"  Mean correlation: {channel_correlations.mean():.3f}")
    print(f"  Max correlation: {channel_correlations.max():.3f}")
    print(f"  Min correlation: {channel_correlations.min():.3f}")

    # Top 10 channels
    top_indices = np.argsort(channel_correlations)[-10:][::-1]
    print(f"\nTop 10 Pitch-Correlated Channels:")
    print(f"{'Channel':<10} {'Correlation':<12}")
    print("-" * 30)
    for idx in top_indices:
        print(f"{idx:<10} {channel_correlations[idx]:<12.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze latent-pitch correlations')
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing latent .pt files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/latent_analysis',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to analyze (None = all)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    latents, midi_notes, velocities = load_latents_and_labels(
        args.data_dir,
        max_samples=args.max_samples
    )

    print(f"\nLoaded {len(latents)} samples")
    print(f"Latent shape: {latents.shape}")

    # Compute correlations
    channel_correlations, channel_time_correlations = compute_pitch_correlations(
        latents, midi_notes
    )

    # Print summary
    print_analysis_summary(channel_correlations, midi_notes)

    # Generate plots
    print("\nGenerating visualizations...")

    # 1. Channel correlations bar plot
    plot_channel_correlations(
        channel_correlations,
        output_dir / 'channel_correlations.png'
    )

    # 2. Correlation heatmap
    plot_correlation_heatmap(
        channel_time_correlations,
        output_dir / 'correlation_heatmap.png'
    )

    # 3. Top channels evolution
    plot_top_channels_evolution(
        latents,
        midi_notes,
        channel_correlations,
        output_dir / 'top_channels_evolution.png',
        top_k=5
    )

    # 4. Individual channel vs pitch (for top channel)
    top_channel = np.argmax(channel_correlations)
    plot_channel_vs_pitch(
        latents,
        midi_notes,
        top_channel,
        output_dir / f'channel_{top_channel}_vs_pitch.png'
    )

    # Save correlation data
    np.save(output_dir / 'channel_correlations.npy', channel_correlations)
    np.save(output_dir / 'channel_time_correlations.npy', channel_time_correlations)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
