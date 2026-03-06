"""
Parse SFZ files to extract sample metadata
"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SFZRegion:
    """Represents one region in an SFZ file"""
    sample: str = ''
    pitch_keycenter: Optional[int] = None
    key: Optional[int] = None
    lokey: Optional[int] = None
    hikey: Optional[int] = None
    lovel: Optional[int] = None
    hivel: Optional[int] = None
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None
    group_label: Optional[str] = None

    def get_pitch(self) -> Optional[int]:
        """Get the pitch for this region (key or pitch_keycenter)"""
        return self.key if self.key is not None else self.pitch_keycenter

    def get_velocity_range(self) -> tuple:
        """Get velocity range, default to full range if not specified"""
        lovel = self.lovel if self.lovel is not None else 0
        hivel = self.hivel if self.hivel is not None else 127
        return (lovel, hivel)

    def has_loop(self) -> bool:
        """Check if this region has loop points"""
        return self.loop_start is not None and self.loop_end is not None


def parse_sfz_file(sfz_path: Path) -> List[SFZRegion]:
    """
    Parse an SFZ file and extract all regions
    Handles both single-line and multi-line region formats
    """
    with open(sfz_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    regions = []
    current_group_label = None
    current_region = None

    # Process line by line
    lines = content.split('\n')

    for line in lines:
        # Remove comments
        comment_idx = line.find('//')
        if comment_idx >= 0:
            line = line[:comment_idx]

        line = line.strip()
        if not line:
            continue

        # Check for <group> tag
        if '<group>' in line.lower():
            # Extract group_label if present on same line
            group_match = re.search(r'group_label\s*=\s*([^\s<>]+)', line, re.IGNORECASE)
            if group_match:
                current_group_label = group_match.group(1)
            continue

        # Check for <region> tag
        if '<region>' in line.lower():
            # Save previous region if exists
            if current_region and current_region.sample:
                regions.append(current_region)

            # Start new region
            current_region = SFZRegion(group_label=current_group_label)

            # Parse any parameters on the same line
            region_line = line[line.lower().find('<region>') + 9:]
            parse_parameters_from_line(region_line, current_region)
            continue

        # Parse parameters for current region
        if current_region is not None:
            parse_parameters_from_line(line, current_region)

    # Don't forget the last region!
    if current_region and current_region.sample:
        regions.append(current_region)

    return regions


def parse_parameters_from_line(line: str, region: SFZRegion):
    """Parse SFZ parameters from a line and update region"""

    # sample=filename.wav (handle filenames with spaces)
    # Match until we hit whitespace followed by another param (word=) or end of line/tag
    sample_match = re.search(r'sample\s*=\s*(.+?)(?:\s+\w+\s*=|<|>|$)', line, re.IGNORECASE)
    if sample_match:
        region.sample = sample_match.group(1).strip()

    # pitch_keycenter=60
    pitch_match = re.search(r'pitch_keycenter\s*=\s*(\d+)', line, re.IGNORECASE)
    if pitch_match:
        region.pitch_keycenter = int(pitch_match.group(1))

    # key=60 (use word boundary to avoid matching lokey/hikey)
    key_match = re.search(r'\bkey\s*=\s*(\d+)', line, re.IGNORECASE)
    if key_match:
        region.key = int(key_match.group(1))

    # lokey=60
    lokey_match = re.search(r'lokey\s*=\s*(\d+)', line, re.IGNORECASE)
    if lokey_match:
        region.lokey = int(lokey_match.group(1))

    # hikey=72
    hikey_match = re.search(r'hikey\s*=\s*(\d+)', line, re.IGNORECASE)
    if hikey_match:
        region.hikey = int(hikey_match.group(1))

    # lovel=0
    lovel_match = re.search(r'lovel\s*=\s*(\d+)', line, re.IGNORECASE)
    if lovel_match:
        region.lovel = int(lovel_match.group(1))

    # hivel=127
    hivel_match = re.search(r'hivel\s*=\s*(\d+)', line, re.IGNORECASE)
    if hivel_match:
        region.hivel = int(hivel_match.group(1))

    # loop_start=12345
    loop_start_match = re.search(r'loop_start\s*=\s*(\d+)', line, re.IGNORECASE)
    if loop_start_match:
        region.loop_start = int(loop_start_match.group(1))

    # loop_end=67890
    loop_end_match = re.search(r'loop_end\s*=\s*(\d+)', line, re.IGNORECASE)
    if loop_end_match:
        region.loop_end = int(loop_end_match.group(1))


def is_drum_or_percussion(sfz_path: Path, regions: List[SFZRegion]) -> bool:
    """
    Determine if this is a drum/unpitched percussion instrument
    """
    path_str = str(sfz_path).lower()

    # Check folder/filename for drum keywords
    drum_keywords = [
        'drum', 'kick', 'snare', 'hat', 'cymbal', 'tom',
        'perc', 'clap', 'snap', 'fx', 'noise', 'impact',
        'shaker', 'tambourine', 'conga', 'bongo', 'rim'
    ]

    for keyword in drum_keywords:
        if keyword in path_str:
            return True

    # Check if regions have no pitch info
    has_pitch = any(r.get_pitch() is not None for r in regions)
    if not has_pitch:
        return True

    return False


if __name__ == '__main__':
    # Test
    import librosa

    test_files = [
        '/Users/laurentclerc/sound-samples/equator/wav/Rhodes Glass Mallets/Rhodes Glass Mallets.sfz',
        '/Users/laurentclerc/sound-samples/equator2/sampler/wav/Strings/Hurdy Gurdy/Hurdy Gurdy - Sustain/Hurdy Gurdy - Sustain.sfz',
    ]

    for sfz_path in test_files:
        sfz_path = Path(sfz_path)
        if not sfz_path.exists():
            print(f"❌ Not found: {sfz_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Parsing: {sfz_path.name}")
        print(f"{'=' * 80}")

        regions = parse_sfz_file(sfz_path)

        print(f"Found {len(regions)} regions")

        # Show first few regions
        print(f"\nFirst 10 regions:")
        print(f"{'Sample':<50} {'Pitch':<6} {'LoVel':<6} {'HiVel':<6}")
        print("-" * 80)

        for region in regions[:10]:
            pitch = region.get_pitch()
            note = librosa.midi_to_note(pitch) if pitch else "?"
            lovel, hivel = region.get_velocity_range()

            print(f"{region.sample:<50} {note:<6} {lovel:<6} {hivel:<6}")