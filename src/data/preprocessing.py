"""
Data preprocessing for multi-sample instruments with pitch verification.
Based on catalog approach with librosa.pyin verification.
"""

import os
import json
import random
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import Counter

from .sfz_parser import parse_sfz_file, is_drum_or_percussion
from .audio_utils import load_audio, normalize_audio, pad_or_trim_audio


# Configuration
VERIFICATION_SAMPLE_RATE = 16000  # For pitch verification
MIN_F0_CONFIDENCE = 0.75  # Confidence threshold for verification
VERIFICATION_SAMPLES = 2  # Number of samples to verify per SFZ initially
MIN_DURATION = 0.15  # Minimum audio duration in seconds


def hz_to_midi(hz: float) -> Optional[int]:
    """Convert frequency in Hz to MIDI note number"""
    if hz <= 0:
        return None
    return int(round(12 * np.log2(hz / 440.0) + 69))


def verify_pitch_with_pyin(
    audio_path: str,
    expected_midi_note: int,
    duration: float = 2.0
) -> Tuple[Optional[int], float, Optional[int], Optional[str]]:
    """
    Verify pitch using librosa.pyin.

    Returns:
        (detected_midi_note, confidence, octave_offset, failure_reason)
    """
    try:
        # Load audio (first 2 seconds, resampled to VERIFICATION_SAMPLE_RATE)
        audio, sr = librosa.load(
            audio_path,
            sr=VERIFICATION_SAMPLE_RATE,
            mono=True,
            duration=duration
        )

        # Skip if too short or silent
        if len(audio) < VERIFICATION_SAMPLE_RATE * MIN_DURATION:
            return None, 0.0, None, "too_short"

        if np.max(np.abs(audio)) < 0.01:
            return None, 0.0, None, "too_quiet"

        # Run pyin pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C1'),  # ~32 Hz
            fmax=librosa.note_to_hz('C8'),  # ~4186 Hz
            sr=VERIFICATION_SAMPLE_RATE,
            frame_length=2048,
            hop_length=512
        )

        # Filter for high confidence voiced frames
        if voiced_probs is None or len(voiced_probs) == 0:
            return None, 0.0, None, "no_voiced_probs"

        valid_mask = (voiced_flag) & (voiced_probs >= MIN_F0_CONFIDENCE) & (~np.isnan(f0))

        if not np.any(valid_mask):
            # Check if we have ANY voiced frames (even low confidence)
            if np.any(voiced_flag & ~np.isnan(f0)):
                max_confidence = np.max(voiced_probs[voiced_flag & ~np.isnan(f0)])
                return None, max_confidence, None, f"low_confidence (max: {max_confidence:.2f})"
            return None, 0.0, None, "no_voiced_frames"

        # Get median frequency of valid frames
        median_hz = np.median(f0[valid_mask])
        mean_confidence = np.mean(voiced_probs[valid_mask])

        # Convert to MIDI
        detected_midi = hz_to_midi(median_hz)

        if detected_midi is None:
            return None, mean_confidence, None, "hz_to_midi_failed"

        # Calculate octave offset (in semitones, then round to nearest octave)
        semitone_diff = detected_midi - expected_midi_note
        octave_offset = round(semitone_diff / 12.0)

        return detected_midi, mean_confidence, octave_offset, None

    except Exception as e:
        return None, 0.0, None, f"exception: {str(e)}"


class MultiSamplePreprocessor:
    """Preprocess multi-sample instruments with pitch verification."""

    def __init__(
        self,
        sample_rate: int = 44100,
        duration: float = 4.0,
        excluded_folders: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.

        Args:
            sample_rate: Target sample rate for training
            duration: Target duration in seconds
            excluded_folders: List of folder paths to exclude
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.excluded_folders = excluded_folders or []

    def is_excluded(self, sample_path: str) -> bool:
        """Check if a sample path is in an excluded folder"""
        sample_path_str = str(Path(sample_path).resolve())
        for excluded in self.excluded_folders:
            if excluded in sample_path_str:
                return True
        return False

    def catalog_from_sfz(
        self,
        root_dir: str,
        library_name: str
    ) -> pd.DataFrame:
        """
        Create catalog by parsing all SFZ files with pitch verification.

        Args:
            root_dir: Root directory containing SFZ files
            library_name: Name of the library

        Returns:
            DataFrame with verified samples
        """
        root = Path(root_dir)
        sfz_files = list(root.rglob('*.sfz'))

        print(f"\n{'=' * 80}")
        print(f"Cataloging {library_name}: {len(sfz_files)} SFZ files")
        print(f"{'=' * 80}")

        all_data = []
        drum_count = 0
        pitched_count = 0
        excluded_count = 0
        excluded_sfz_count = 0
        verification_stats = {
            'verified': 0,
            'octave_offsets_found': 0,
            'verification_failed': 0
        }
        failure_reasons = []

        for sfz_path in tqdm(sfz_files, desc=f"Processing {library_name}"):
            # Check if SFZ file is in excluded folder
            if self.is_excluded(str(sfz_path)):
                excluded_sfz_count += 1
                continue

            # Parse SFZ
            regions = parse_sfz_file(sfz_path)

            # Check if drums
            if is_drum_or_percussion(sfz_path, regions):
                drum_count += 1
                continue

            pitched_count += 1

            # Collect all samples from this SFZ
            sfz_samples = []

            for region in regions:
                # Get sample path
                sample_name = region.sample
                sample_path = sfz_path.parent / sample_name

                if not sample_path.exists():
                    continue

                # Check if excluded
                if self.is_excluded(str(sample_path)):
                    excluded_count += 1
                    continue

                # Get pitch
                pitch = region.get_pitch()
                if pitch is None:
                    continue

                # Get velocity range
                lovel, hivel = region.get_velocity_range()

                # Get audio info
                try:
                    info = sf.info(sample_path)
                    duration = info.duration
                    sample_rate_orig = info.samplerate
                except:
                    duration = None
                    sample_rate_orig = None

                sfz_samples.append({
                    'library': library_name,
                    'sfz_file': sfz_path.name,
                    'instrument': sfz_path.parent.name,
                    'path': str(sample_path),
                    'filename': sample_path.name,
                    'midi_note': pitch,
                    'lovel': lovel,
                    'hivel': hivel,
                    'velocity_mid': (lovel + hivel) // 2,
                    'has_loop': region.has_loop(),
                    'loop_start': region.loop_start,
                    'loop_end': region.loop_end,
                    'group_label': region.group_label,
                    'duration': duration,
                    'sample_rate': sample_rate_orig,
                })

            # Skip if no valid samples
            if not sfz_samples:
                continue

            # Verify pitch with pyin
            octave_offset = 0  # Default: no offset

            # Select samples to verify
            samples_to_verify = random.sample(
                sfz_samples,
                min(VERIFICATION_SAMPLES, len(sfz_samples))
            )

            octave_offsets = []
            verification_details = []

            for sample in samples_to_verify:
                detected_midi, confidence, offset, failure_reason = verify_pitch_with_pyin(
                    sample['path'],
                    sample['midi_note']
                )

                verification_details.append({
                    'sample': sample['filename'],
                    'expected_midi': sample['midi_note'],
                    'detected_midi': detected_midi,
                    'confidence': confidence,
                    'offset': offset,
                    'failure_reason': failure_reason
                })

                if offset is not None and confidence >= MIN_F0_CONFIDENCE:
                    octave_offsets.append(offset)

            # Fallback: If initial samples failed, try all samples
            if not octave_offsets and len(sfz_samples) > VERIFICATION_SAMPLES:
                all_results = []

                for sample in sfz_samples:
                    detected_midi, confidence, offset, failure_reason = verify_pitch_with_pyin(
                        sample['path'],
                        sample['midi_note']
                    )

                    if offset is not None and confidence > 0:
                        all_results.append({
                            'sample': sample['filename'],
                            'expected_midi': sample['midi_note'],
                            'detected_midi': detected_midi,
                            'confidence': confidence,
                            'offset': offset,
                            'failure_reason': failure_reason
                        })

                # Sort by confidence and take top 2
                if all_results:
                    all_results.sort(key=lambda x: x['confidence'], reverse=True)
                    top_2 = all_results[:2]

                    for result in top_2:
                        if result['confidence'] >= MIN_F0_CONFIDENCE:
                            octave_offsets.append(result['offset'])

                    verification_details = top_2

            # Calculate consensus octave offset
            if octave_offsets:
                # Use most common offset (mode)
                offset_counts = Counter(octave_offsets)
                octave_offset = offset_counts.most_common(1)[0][0]

                verification_stats['verified'] += 1
                if octave_offset != 0:
                    verification_stats['octave_offsets_found'] += 1
                    print(f"\n  ⚠️  Octave offset detected in {sfz_path.name}: {octave_offset:+d} octaves")
            else:
                verification_stats['verification_failed'] += 1
                for detail in verification_details:
                    if detail['failure_reason']:
                        failure_reasons.append(detail['failure_reason'])

            # Add octave_offset and verification status to all samples
            verification_passed = len(octave_offsets) > 0
            for sample in sfz_samples:
                sample['octave_offset'] = octave_offset
                sample['verification_passed'] = verification_passed
                sample['sfz_sample_count'] = len(sfz_samples)
                all_data.append(sample)

        print(f"\nResults:")
        print(f"  Pitched instruments: {pitched_count}")
        print(f"  Drums (skipped):     {drum_count}")
        print(f"  Excluded SFZ files:  {excluded_sfz_count}")
        print(f"  Excluded samples:    {excluded_count}")
        print(f"  Total samples:       {len(all_data)}")

        print(f"\nPitch verification:")
        print(f"  SFZ files verified:       {verification_stats['verified']}")
        print(f"  Octave offsets detected:  {verification_stats['octave_offsets_found']}")
        print(f"  Verification failed:      {verification_stats['verification_failed']}")

        # Show failure reason breakdown
        if failure_reasons:
            reason_counts = Counter(failure_reasons)
            print(f"\n  Failure reason breakdown:")
            for reason, count in reason_counts.most_common():
                print(f"    • {reason}: {count}")

        # Create DataFrame
        df_all = pd.DataFrame(all_data)

        if len(df_all) > 0:
            initial_count = len(df_all)

            # Filter out failed verifications
            failed_verification = (~df_all['verification_passed']).sum()
            df_filtered = df_all[df_all['verification_passed'] == True]

            # Filter out single-sample SFZ files
            single_sample = (df_filtered['sfz_sample_count'] == 1).sum()
            df_filtered = df_filtered[df_filtered['sfz_sample_count'] > 1]

            final_count = len(df_filtered)

            print(f"\nFiltering:")
            print(f"  Initial samples:              {initial_count}")
            print(f"  Failed verification samples:  {failed_verification}")
            print(f"  Single-sample SFZ files:      {single_sample}")
            print(f"  Final samples:                {final_count}")

            return df_filtered

        return df_all

    def process_catalog_to_tensors(
        self,
        catalog: pd.DataFrame,
        output_dir: str,
        train_split: float = 0.9
    ):
        """
        Process catalog samples into PyTorch tensors.

        Args:
            catalog: DataFrame from catalog_from_sfz
            output_dir: Output directory
            train_split: Train/val split ratio
        """
        # Group by instrument to assign instrument IDs
        instrument_ids = {}
        for idx, instrument_name in enumerate(catalog['instrument'].unique()):
            instrument_ids[instrument_name] = idx

        # Process all samples
        all_samples = []

        for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Processing audio"):
            try:
                # Load and process audio
                audio = load_audio(row['path'], self.sample_rate)
                audio = normalize_audio(audio, method='peak')
                audio = pad_or_trim_audio(audio, self.num_samples)

                # Apply octave offset to MIDI note
                midi_note = row['midi_note'] + (row['octave_offset'] * 12)

                sample_data = {
                    'audio': audio,
                    'midi_note': midi_note,
                    'velocity': row['velocity_mid'],
                    'instrument_id': instrument_ids[row['instrument']],
                    'file_path': row['path'],
                    'has_loop': row['has_loop'],
                    'loop_start': row['loop_start'] if row['has_loop'] else None,
                    'loop_end': row['loop_end'] if row['has_loop'] else None,
                }

                all_samples.append(sample_data)

            except Exception as e:
                print(f"Error processing {row['path']}: {e}")
                continue

        if len(all_samples) == 0:
            print("No samples were successfully processed")
            return

        print(f"\nTotal samples processed: {len(all_samples)}")

        # Split into train/val with instrument awareness
        # Ensure each instrument has at least 2 samples in each split
        random.seed(42)

        # Group samples by instrument
        from collections import defaultdict
        instrument_samples = defaultdict(list)
        for sample in all_samples:
            instrument_samples[sample['instrument_id']].append(sample)

        train_samples = []
        val_samples = []
        excluded_instruments = []

        print(f"\nPerforming instrument-aware train/val split...")
        print(f"  Minimum samples per instrument per split: 2")

        for instrument_id, samples in instrument_samples.items():
            num_samples = len(samples)

            # Skip instruments with fewer than 4 samples (can't split 2/2 minimum)
            if num_samples < 4:
                excluded_instruments.append((instrument_id, num_samples))
                print(f"  ⏭️  Excluding instrument {instrument_id}: only {num_samples} samples (need ≥4)")
                continue

            # Shuffle this instrument's samples
            random.shuffle(samples)

            # Calculate split ensuring at least 2 in each set
            split_idx = max(2, int(num_samples * train_split))
            # Make sure val also has at least 2
            if num_samples - split_idx < 2:
                split_idx = num_samples - 2

            train_samples.extend(samples[:split_idx])
            val_samples.extend(samples[split_idx:])

        # Shuffle the final splits
        random.shuffle(train_samples)
        random.shuffle(val_samples)

        print(f"\nSplit results:")
        print(f"  Instruments used: {len(instrument_samples) - len(excluded_instruments)}")
        print(f"  Instruments excluded: {len(excluded_instruments)}")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")

        if excluded_instruments:
            print(f"\n  Excluded instrument details:")
            for inst_id, num in sorted(excluded_instruments):
                print(f"    - Instrument {inst_id}: {num} samples")

        # Save samples
        self.save_processed_samples(train_samples, output_dir, 'train')
        self.save_processed_samples(val_samples, output_dir, 'val')

    def save_processed_samples(
        self,
        samples: List[Dict],
        output_dir: str,
        split: str = 'train'
    ):
        """Save processed samples to disk."""
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Save each sample
        for i, sample in enumerate(tqdm(samples, desc=f"Saving {split} samples")):
            sample_path = os.path.join(split_dir, f'sample_{i:06d}.pt')
            torch.save(sample, sample_path)

        # Save metadata
        metadata = {
            'num_samples': len(samples),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'num_instruments': len(set(s['instrument_id'] for s in samples))
        }

        metadata_path = os.path.join(split_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(samples)} samples to {split_dir}")
