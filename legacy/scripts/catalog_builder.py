"""
Create catalog from SFZ files (ground truth)
Verifies SFZ pitch values with librosa.pyin and detects octave offsets
"""
import json
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sfz_parser import parse_sfz_file, is_drum_or_percussion

# Configuration
SAMPLE_RATE = 16000
MIN_F0_CONFIDENCE = 0.75  # Confidence threshold for verification
VERIFICATION_SAMPLES = 2  # Number of samples to verify per SFZ file initially
MIN_DURATION = 0.15  # Minimum audio duration in seconds for pitch detection


def load_exclusion_config(config_path=None):
    """Load folder exclusion configuration"""
    if config_path is None:
        # Compute project root (script is in scripts/, project root is parent)
        script_dir = Path(__file__).parent
        proj_root = script_dir.parent
        config_path = proj_root / 'config' / 'data_prep_config.json'

    print(f"\nLooking for config at: {config_path}")
    print(f"Config exists: {config_path.exists()}")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        excluded = config.get('excluded_folders', [])

        print(f"\nLoaded exclusion config:")
        print(f"  Excluding {len(excluded)} folder patterns")
        for folder in excluded:
            print(f"    - {folder}")
        return excluded
    return []


def is_excluded(sample_path, excluded_folders):
    """Check if a sample path contains any excluded folder name as a path component"""
    # Get all path components (parts)
    path_parts = Path(sample_path).resolve().parts

    # Check if any excluded folder name matches any path component
    for excluded_folder in excluded_folders:
        if excluded_folder in path_parts:
            return True

    return False


def hz_to_midi(hz):
    """Convert frequency in Hz to MIDI note number"""
    if hz <= 0:
        return None
    return 12 * np.log2(hz / 440.0) + 69


def verify_pitch_with_pyin(audio_path, expected_midi_note, duration=2.0):
    """
    Verify pitch using librosa.pyin
    Returns (detected_midi_note, confidence, octave_offset, failure_reason)
    """
    try:
        # Load audio (first 2 seconds, resampled to SAMPLE_RATE)
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=duration)

        # Skip if too short or silent (after resampling to 16kHz)
        if len(audio) < SAMPLE_RATE * MIN_DURATION:
            return None, 0.0, None, "too_short"

        if np.max(np.abs(audio)) < 0.01:
            return None, 0.0, None, "too_quiet"

        # Run pyin pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C1'),  # ~32 Hz
            fmax=librosa.note_to_hz('C8'),  # ~4186 Hz
            sr=SAMPLE_RATE,
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
        print(f"\n  Warning: pyin verification failed for {Path(audio_path).name}: {e}")
        return None, 0.0, None, f"exception: {str(e)}"


def catalog_from_sfz(root_dir, library_name, excluded_folders=None):
    """Create catalog by parsing all SFZ files with pitch verification"""
    if excluded_folders is None:
        excluded_folders = []

    root = Path(root_dir)

    # Use glob with recursive and follow_symlinks to find SFZ files
    # rglob() doesn't follow symlinks in Python < 3.12
    import glob
    sfz_pattern = str(root / '**' / '*.sfz')

    print(f"\n{'=' * 80}")
    print(f"🔍 SCANNING FOR SFZ FILES")
    print(f"{'=' * 80}")
    print(f"Library: {library_name}")
    print(f"Root directory: {root}")
    print(f"Root exists: {root.exists()}")
    print(f"Root is symlink: {root.is_symlink()}")
    if root.is_symlink():
        print(f"Root resolves to: {root.resolve()}")
    print(f"Search pattern: {sfz_pattern}")

    sfz_files = [Path(p) for p in glob.glob(sfz_pattern, recursive=True)]

    print(f"\n{'=' * 80}")
    print(f"Cataloging {library_name}: {len(sfz_files)} SFZ files")
    print(f"{'=' * 80}")

    # Check if Baby Grand Piano is in the list
    baby_grand_files = [f for f in sfz_files if 'Baby Grand' in str(f)]
    if baby_grand_files:
        print(f"\n🎹 FOUND {len(baby_grand_files)} Baby Grand Piano SFZ files:")
        for f in baby_grand_files:
            print(f"   ✓ {f.relative_to(root)}")
    else:
        print(f"\n⚠️  Baby Grand Piano NOT found in initial SFZ list")

    # Show first 10 SFZ files found
    print(f"\n📋 First 10 SFZ files found:")
    for f in sfz_files[:10]:
        print(f"   {f.relative_to(root)}")

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
    failure_reasons = []  # Track all failure reasons for summary
    failed_sfz_files = []  # Track SFZ files that failed verification

    # Track all SFZ files by category
    sfz_tracking = {
        'excluded': [],
        'drums': [],
        'verified': [],
        'failed_verification': []
    }

    for sfz_path in tqdm(sfz_files, desc=f"Processing {library_name}"):
        # Debug logging for Baby Grand Piano
        is_baby_grand = 'Baby Grand' in str(sfz_path)

        if is_baby_grand:
            print(f"\n🎹 DEBUG: Processing Baby Grand: {sfz_path.name}")
            print(f"    Full path: {sfz_path}")
            print(f"    Exists: {sfz_path.exists()}")

        # Check if SFZ file is in excluded folder
        if is_excluded(sfz_path, excluded_folders):
            excluded_sfz_count += 1
            sfz_tracking['excluded'].append(sfz_path.name)
            print(f"\n  ⏭️  Skipping excluded: {sfz_path.name}")
            if is_baby_grand:
                print(f"    ⚠️  Baby Grand was EXCLUDED!")
            continue  # Skip entire SFZ file if in excluded folder

        # Parse SFZ
        regions = parse_sfz_file(sfz_path)

        if is_baby_grand:
            print(f"    Regions parsed: {len(regions)}")

        # Check if drums
        if is_drum_or_percussion(sfz_path, regions):
            drum_count += 1
            sfz_tracking['drums'].append(sfz_path.name)
            if is_baby_grand:
                print(f"    ⚠️  Baby Grand classified as DRUM/PERCUSSION!")
            continue  # Skip drums

        pitched_count += 1

        if is_baby_grand:
            print(f"    Classified as PITCHED instrument")

        # Collect all samples from this SFZ first
        sfz_samples = []

        # Debug counters for Baby Grand
        if is_baby_grand:
            debug_counts = {
                'total_regions': len(regions),
                'sample_not_found': 0,
                'excluded': 0,
                'no_pitch': 0,
                'valid': 0
            }

        for region in regions:
            # Get sample path (relative to SFZ file)
            sample_name = region.sample
            sample_path = sfz_path.parent / sample_name

            if not sample_path.exists():
                if is_baby_grand and debug_counts['sample_not_found'] < 5:
                    print(f"      Sample not found: {sample_name}")
                    print(f"        Looking at: {sample_path}")
                if is_baby_grand:
                    debug_counts['sample_not_found'] += 1
                continue

            # Check if excluded
            if is_excluded(sample_path, excluded_folders):
                excluded_count += 1
                if is_baby_grand and debug_counts['excluded'] < 5:
                    print(f"      Sample excluded: {sample_name}")
                if is_baby_grand:
                    debug_counts['excluded'] += 1
                continue

            # Get pitch
            pitch = region.get_pitch()
            if pitch is None:
                if is_baby_grand and debug_counts['no_pitch'] < 5:
                    print(f"      No pitch for sample: {sample_name}")
                if is_baby_grand:
                    debug_counts['no_pitch'] += 1
                continue

            if is_baby_grand:
                debug_counts['valid'] += 1

            # Get velocity range
            lovel, hivel = region.get_velocity_range()

            # Get audio info
            try:
                info = sf.info(sample_path)
                duration = info.duration
                sample_rate = info.samplerate
            except:
                duration = None
                sample_rate = None

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
                'sample_rate': sample_rate,
            })

        # Skip if no valid samples
        if not sfz_samples:
            if is_baby_grand:
                print(f"    ❌ SKIPPED: No valid samples collected!")
                print(f"    Debug breakdown:")
                print(f"      Total regions: {debug_counts['total_regions']}")
                print(f"      Sample not found: {debug_counts['sample_not_found']}")
                print(f"      Excluded: {debug_counts['excluded']}")
                print(f"      No pitch: {debug_counts['no_pitch']}")
                print(f"      Valid: {debug_counts['valid']}")
            continue

        if is_baby_grand:
            print(f"    Collected {len(sfz_samples)} valid samples")

        # Verify pitch with pyin on sample of files
        octave_offset = 0  # Default: no offset

        # Select samples to verify (up to VERIFICATION_SAMPLES)
        samples_to_verify = random.sample(sfz_samples, min(VERIFICATION_SAMPLES, len(sfz_samples)))

        octave_offsets = []
        verification_details = []

        for sample in samples_to_verify:
            detected_midi, confidence, offset, failure_reason = verify_pitch_with_pyin(
                sample['path'],
                sample['midi_note']
            )

            # Store details for logging
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

        # Fallback: If initial samples failed, try all samples and use 2 highest confidence
        if not octave_offsets and len(sfz_samples) > VERIFICATION_SAMPLES:
            print(f"\n  🔄 Fallback: Testing all {len(sfz_samples)} samples in {sfz_path.name}...")
            all_results = []

            for sample in sfz_samples:
                detected_midi, confidence, offset, failure_reason = verify_pitch_with_pyin(
                    sample['path'],
                    sample['midi_note']
                )

                if offset is not None and confidence > 0:  # Any confidence > 0
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

                print(f"      Found {len(all_results)} samples with detectable pitch")
                print(f"      Using top 2 by confidence:")
                for result in top_2:
                    expected_note = librosa.midi_to_note(result['expected_midi'])
                    detected_note = librosa.midi_to_note(result['detected_midi'])
                    print(f"        • {result['sample']}: {expected_note} → {detected_note} (conf: {result['confidence']:.2f})")

                    if result['confidence'] >= MIN_F0_CONFIDENCE:
                        octave_offsets.append(result['offset'])

                # Update verification_details to show the top 2
                verification_details = top_2

        # Calculate consensus octave offset
        if octave_offsets:
            # Use most common offset (mode)
            from collections import Counter
            offset_counts = Counter(octave_offsets)
            octave_offset = offset_counts.most_common(1)[0][0]

            verification_stats['verified'] += 1
            sfz_tracking['verified'].append(sfz_path.name)
            if octave_offset != 0:
                verification_stats['octave_offsets_found'] += 1
                # Log octave offset detection
                print(f"\n  ⚠️  Octave offset detected in {sfz_path.name}:")
                print(f"      Offset: {octave_offset:+d} octaves")
                for detail in verification_details:
                    if detail['offset'] is not None:
                        expected_note = librosa.midi_to_note(detail['expected_midi'])
                        detected_note = librosa.midi_to_note(detail['detected_midi'])
                        print(f"      • {detail['sample']}: {expected_note} → {detected_note} (conf: {detail['confidence']:.2f})")
        else:
            verification_stats['verification_failed'] += 1
            failed_sfz_files.append(sfz_path.name)  # Track failed SFZ
            sfz_tracking['failed_verification'].append(sfz_path.name)
            # Log verification failure with details
            print(f"\n  ❌ Verification FAILED for {sfz_path.name}:")
            for detail in verification_details:
                expected_note = librosa.midi_to_note(detail['expected_midi'])
                if detail['failure_reason']:
                    print(f"      • {detail['sample']} (expected {expected_note}): {detail['failure_reason']}")
                    failure_reasons.append(detail['failure_reason'])
                else:
                    detected_note = librosa.midi_to_note(detail['detected_midi']) if detail['detected_midi'] else "N/A"
                    print(f"      • {detail['sample']}: {expected_note} → {detected_note} (conf: {detail['confidence']:.2f})")
                    failure_reasons.append("unknown")

        # Add octave_offset and verification status to all samples from this SFZ
        verification_passed = len(octave_offsets) > 0
        for sample in sfz_samples:
            sample['octave_offset'] = octave_offset
            sample['verification_passed'] = verification_passed
            sample['sfz_sample_count'] = len(sfz_samples)
            all_data.append(sample)

        if is_baby_grand:
            print(f"    Verification passed: {verification_passed}")
            print(f"    Octave offset: {octave_offset}")
            print(f"    Added {len(sfz_samples)} samples to catalog")
            print(f"    Total samples in catalog so far: {len(all_data)}")

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
        from collections import Counter
        reason_counts = Counter(failure_reasons)
        print(f"\n  Failure reason breakdown:")
        for reason, count in reason_counts.most_common():
            print(f"    • {reason}: {count} samples")

    # Filter out samples from failed verifications and single-sample SFZ files
    df_all = pd.DataFrame(all_data)

    # Debug: Check if Baby Grand made it to DataFrame
    if len(df_all) > 0:
        baby_grand_df = df_all[df_all['instrument'].str.contains('Baby Grand', na=False)]
        print(f"\n🎹 DEBUG: Baby Grand in DataFrame BEFORE filtering:")
        print(f"    Samples: {len(baby_grand_df)}")
        if len(baby_grand_df) > 0:
            print(f"    SFZ files: {baby_grand_df['sfz_file'].unique()}")
            print(f"    Sample counts: {baby_grand_df['sfz_sample_count'].unique()}")
            print(f"    Verification status: {baby_grand_df['verification_passed'].unique()}")

    if len(df_all) > 0:
        initial_count = len(df_all)

        # Filter out failed verifications
        failed_verification = (~df_all['verification_passed']).sum()
        df_filtered = df_all[df_all['verification_passed'] == True]

        # Filter out single-sample SFZ files
        single_sample = (df_filtered['sfz_sample_count'] == 1).sum()
        df_filtered = df_filtered[df_filtered['sfz_sample_count'] > 1]

        final_count = len(df_filtered)
        filtered_count = initial_count - final_count

        print(f"\nFiltering:")
        print(f"  Initial samples:              {initial_count}")
        print(f"  Failed verification samples:  {failed_verification}")
        print(f"  Single-sample SFZ files:      {single_sample}")
        print(f"  Final samples:                {final_count}")
        print(f"  Total filtered out:           {filtered_count}")

        # Debug: Check if Baby Grand survived filtering
        baby_grand_final = df_filtered[df_filtered['instrument'].str.contains('Baby Grand', na=False)]
        print(f"\n🎹 DEBUG: Baby Grand in DataFrame AFTER filtering:")
        print(f"    Samples: {len(baby_grand_final)}")
        if len(baby_grand_final) > 0:
            print(f"    SFZ files: {baby_grand_final['sfz_file'].unique()}")
        else:
            print(f"    ❌ Baby Grand was FILTERED OUT!")

        return df_filtered, failed_sfz_files, sfz_tracking

    return df_all, failed_sfz_files, sfz_tracking


def main():
    print("=" * 80)
    print("CATALOG FROM SFZ FILES WITH PITCH VERIFICATION (librosa.pyin)")
    print("=" * 80)
    print("\nThis will:")
    print("1. Parse all SFZ files for ground truth metadata")
    print("2. Skip drums and unpitched percussion")
    print("3. Extract pitch, velocity, and loop info")
    print(f"4. Verify pitch accuracy with librosa.pyin ({VERIFICATION_SAMPLES} samples per SFZ)")
    print("5. Detect and record octave offsets")
    print("6. Exclude configured folders (if config exists)")

    # Set random seed for reproducibility
    random.seed(42)

    # Load exclusion config
    excluded_folders = load_exclusion_config()

    # Catalog libraries from data/raw (project_root/data/raw)
    data_raw = project_root / 'data' / 'raw'

    catalogs = []
    all_failed_sfz = []  # Track all failed SFZ files across libraries
    all_sfz_tracking = {}  # Track all SFZ files by library and category

    # Check for equator
    equator_path = data_raw / 'equator'
    if equator_path.exists():
        df_eq1, failed_eq1, tracking_eq1 = catalog_from_sfz(equator_path, 'equator1', excluded_folders)
        catalogs.append(df_eq1)
        all_failed_sfz.extend([('equator1', f) for f in failed_eq1])
        all_sfz_tracking['equator1'] = tracking_eq1

    # Check for equator2
    equator2_path = data_raw / 'equator2' / 'sampler' / 'wav'
    if equator2_path.exists():
        df_eq2, failed_eq2, tracking_eq2 = catalog_from_sfz(equator2_path, 'equator2', excluded_folders)
        catalogs.append(df_eq2)
        all_failed_sfz.extend([('equator2', f) for f in failed_eq2])
        all_sfz_tracking['equator2'] = tracking_eq2

    # Combine all catalogs
    if not catalogs:
        print("\n❌ No data found in data/raw/")
        return

    df_all = pd.concat(catalogs, ignore_index=True)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (after filtering)")
    print("=" * 80)
    print(f"Total samples:       {len(df_all)} (excludes failed verifications & single-sample SFZ files)")
    print(f"Pitch range:         MIDI {df_all['midi_note'].min():.0f} - {df_all['midi_note'].max():.0f}")
    print(f"Velocity layers:     {df_all.groupby(['library', 'instrument']).size().mean():.1f} avg per instrument")
    print(f"Samples with loops:  {df_all['has_loop'].sum()} ({df_all['has_loop'].sum() / len(df_all) * 100:.1f}%)")

    print(f"\nBy library:")
    print(df_all.groupby('library').size())

    print(f"\nTop 20 instruments by sample count:")
    print(df_all.groupby('instrument').size().sort_values(ascending=False).head(20))

    # Velocity layer analysis
    print(f"\nVelocity layers per instrument:")
    vel_layers = df_all.groupby(['library', 'instrument'])['velocity_mid'].nunique()
    print(f"  Min layers:  {vel_layers.min()}")
    print(f"  Max layers:  {vel_layers.max()}")
    print(f"  Mean layers: {vel_layers.mean():.1f}")

    # Octave offset analysis
    print(f"\nOctave offset analysis:")
    offset_counts = df_all['octave_offset'].value_counts().sort_index()
    print(f"  Samples with no offset (0):     {offset_counts.get(0, 0)} ({offset_counts.get(0, 0)/len(df_all)*100:.1f}%)")
    if len(offset_counts) > 1:
        print(f"  Samples with offset detected:   {len(df_all) - offset_counts.get(0, 0)} ({(len(df_all) - offset_counts.get(0, 0))/len(df_all)*100:.1f}%)")
        print(f"\n  Offset distribution:")
        for offset, count in offset_counts.items():
            if offset != 0:
                octaves = "octave" if abs(offset) == 1 else "octaves"
                direction = "up" if offset > 0 else "down"
                print(f"    {offset:+2d} ({abs(offset)} {octaves} {direction}): {count} samples ({count/len(df_all)*100:.1f}%)")

    # Save
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'sample_catalog.csv'
    df_all.to_csv(output_file, index=False)

    print(f"\n✓ Saved catalog to {output_file}")

    # Print comprehensive SFZ file summary
    print("\n" + "=" * 80)
    print("SFZ FILE PROCESSING SUMMARY")
    print("=" * 80)

    for library in sorted(all_sfz_tracking.keys()):
        tracking = all_sfz_tracking[library]
        total = sum(len(v) for v in tracking.values())

        print(f"\n{library.upper()}:")
        print(f"  Total SFZ files found: {total}")
        print(f"  ✓ Verified:            {len(tracking['verified'])}")
        print(f"  ✗ Failed verification: {len(tracking['failed_verification'])}")
        print(f"  🥁 Drums (skipped):    {len(tracking['drums'])}")
        print(f"  ⏭️  Excluded:           {len(tracking['excluded'])}")

        # Show verified files (included in catalog)
        if tracking['verified']:
            print(f"\n  Verified SFZ files ({len(tracking['verified'])} - INCLUDED in catalog):")
            for sfz in sorted(tracking['verified']):
                print(f"    ✓ {sfz}")

        # Show failed verification files
        if tracking['failed_verification']:
            print(f"\n  Failed verification ({len(tracking['failed_verification'])} - EXCLUDED from catalog):")
            for sfz in sorted(tracking['failed_verification']):
                print(f"    ✗ {sfz}")

        # Show drums (optional, can be collapsed)
        if tracking['drums'] and len(tracking['drums']) <= 20:
            print(f"\n  Drums/percussion ({len(tracking['drums'])} - EXCLUDED from catalog):")
            for sfz in sorted(tracking['drums']):
                print(f"    🥁 {sfz}")
        elif tracking['drums']:
            print(f"\n  Drums/percussion ({len(tracking['drums'])} - EXCLUDED from catalog - list truncated)")

        # Show excluded
        if tracking['excluded']:
            print(f"\n  Excluded by config ({len(tracking['excluded'])}):")
            for sfz in sorted(tracking['excluded']):
                print(f"    ⏭️  {sfz}")

    print("\n" + "=" * 80)

    # Print summary of failed verifications (old format, now redundant but keeping for compatibility)
    if all_failed_sfz:
        print("\n" + "=" * 80)
        print(f"SFZ FILES WITH FAILED VERIFICATION ({len(all_failed_sfz)} total)")
        print("=" * 80)
        print("\nThese SFZ files were excluded because pitch verification failed.")
        print("Samples from these instruments are NOT in the catalog.\n")

        # Group by library
        from collections import defaultdict
        by_library = defaultdict(list)
        for library, sfz_name in all_failed_sfz:
            by_library[library].append(sfz_name)

        for library in sorted(by_library.keys()):
            failed = by_library[library]
            print(f"\n{library} ({len(failed)} failed):")
            for sfz_name in sorted(failed):
                print(f"  • {sfz_name}")

        print("\n" + "=" * 80)

    return df_all


if __name__ == '__main__':
    main()
