# Single-Sample Instrument Filtering - Complete Fix

**Date**: 2025-12-30
**Status**: ✅ FIXED at multiple levels

## Problem

Single-sample instruments (instruments with only 1 audio sample) caused training issues:
- **Validation NaN bug**: InputSampleEncoder's attention mechanism fails when trying to attend to 0 input samples
- **Poor multi-sample conditioning**: Can't learn meaningful timbral relationships with only 1 example
- **Dataset quality**: Defeats the purpose of multi-sample instrument generation

## Multi-Level Solution

We've implemented fixes at **three levels** to ensure robust filtering:

### Level 1: Cataloging (catalog_builder.py & preprocessing.py)

**Location**: Lines 356-358 in `src/data/preprocessing.py`

Filters out single-sample SFZ files during catalog creation:

```python
# Filter out single-sample SFZ files
single_sample = (df_filtered['sfz_sample_count'] == 1).sum()
df_filtered = df_filtered[df_filtered['sfz_sample_count'] > 1]
```

**Result**: SFZ files with only 1 sample are excluded from the catalog entirely.

### Level 2: Train/Val Split (preprocessing.py) - **NEW**

**Location**: Lines 427-478 in `src/data/preprocessing.py`

Implements **instrument-aware splitting** to ensure each instrument has at least 2 samples in both train and val sets:

```python
# Split into train/val with instrument awareness
# Ensure each instrument has at least 2 samples in each split

# Group samples by instrument
instrument_samples = defaultdict(list)
for sample in all_samples:
    instrument_samples[sample['instrument_id']].append(sample)

train_samples = []
val_samples = []
excluded_instruments = []

for instrument_id, samples in instrument_samples.items():
    num_samples = len(samples)

    # Skip instruments with fewer than 4 samples (can't split 2/2 minimum)
    if num_samples < 4:
        excluded_instruments.append((instrument_id, num_samples))
        continue

    # Calculate split ensuring at least 2 in each set
    split_idx = max(2, int(num_samples * train_split))
    if num_samples - split_idx < 2:
        split_idx = num_samples - 2

    train_samples.extend(samples[:split_idx])
    val_samples.extend(samples[split_idx:])
```

**Changes**:
- **Before**: Random split across all samples → instruments could end up with 1 or 0 samples in a split
- **After**: Per-instrument split → guarantees ≥2 samples per instrument in each split
- **Minimum requirement**: Instruments need ≥4 samples total (2 train + 2 val minimum)

**Result**: No instrument in train or val will have fewer than 2 samples.

### Level 3: Dataset Runtime Fallback (dataset.py)

**Location**: Lines 127-136 in `src/data/dataset.py`

Handles edge case if a single-sample instrument somehow makes it through:

```python
if len(available_samples) >= actual_num_input:
    input_indices = random.sample(available_samples, actual_num_input)
elif len(available_samples) > 0:
    # If not enough other samples, use what we have
    input_indices = available_samples
    actual_num_input = len(input_indices)
else:
    # If no other samples (single-sample instrument), use current sample
    input_indices = [idx]
    actual_num_input = 1
```

**Result**: Gracefully handles single-sample instruments by using the sample itself as input.

### Level 4: InputSampleEncoder Safety Check (input_encoder.py)

**Location**: Lines 135-169 in `src/models/input_encoder.py`

Prevents NaN when attention has no valid keys to attend to:

```python
# Handle case where num_valid is 0 (shouldn't happen with dataset fix, but be safe)
all_masked = key_padding_mask.all(dim=1)
if all_masked.any():
    # For samples with no valid inputs, use zero embedding
    aggregated = torch.zeros(batch_size, embed_dim, device=embeddings.device)

    # For samples with valid inputs, use attention
    valid_mask = ~all_masked
    if valid_mask.any():
        # Apply attention only to valid samples
        ...
```

**Result**: Even if 0 input samples occur, returns zero embedding instead of NaN.

## Why This Multi-Level Approach?

1. **Defense in Depth**: Multiple layers of protection ensure robustness
2. **Fail-Safe**: If one filter is bypassed, others catch the issue
3. **Clear Error Messages**: Each level provides informative logging
4. **Future-Proof**: Handles edge cases even as data sources change

## What Gets Filtered

### During Cataloging:
- ✅ SFZ files with only 1 sample
- ✅ Failed pitch verification
- ✅ Drums and percussion
- ✅ Excluded folders (per config)

### During Train/Val Split (NEW):
- ✅ Instruments with <4 total samples
- Shows exactly which instruments were excluded and why

### At Runtime:
- ✅ Handles any remaining edge cases gracefully

## Expected Output When Preprocessing

```
Filtering:
  Initial samples:              1250
  Failed verification samples:  45
  Single-sample SFZ files:      12
  Final samples:                1193

Performing instrument-aware train/val split...
  Minimum samples per instrument per split: 2
  ⏭️  Excluding instrument 5: only 3 samples (need ≥4)
  ⏭️  Excluding instrument 12: only 2 samples (need ≥4)

Split results:
  Instruments used: 22
  Instruments excluded: 2
  Train samples: 1051
  Val samples: 142

  Excluded instrument details:
    - Instrument 5: 3 samples
    - Instrument 12: 2 samples
```

## Reprocessing Your Data

To apply these fixes, you'll need to reprocess your dataset:

```bash
# Step 1: Create catalog (already filters single-sample SFZ files)
python scripts/catalog_builder.py

# Step 2: Preprocess with new instrument-aware splitting
python scripts/preprocess_dataset.py --input data/raw --output data/processed

# Step 3: Resume training with clean data
python scripts/train_diffusion.py --resume checkpoints/diffusion/checkpoint_epoch006.pt
```

## Benefits

1. **No more validation NaN**: All instruments have ≥2 samples for conditioning
2. **Better training quality**: Only instruments with sufficient samples
3. **Faster debugging**: Clear logging of what was excluded and why
4. **Consistent behavior**: Works regardless of how SFZ files are structured

## Migration Notes

- Existing processed data may still have single-sample instruments in splits
- Recommend reprocessing with the updated preprocessing script
- The dataset runtime fix (Level 3) allows existing checkpoints to continue training without errors
- For best results, reprocess and start training fresh

## Testing

Verified fixes work by:
1. ✅ All 5 validation batches return valid loss (no NaN)
2. ✅ Tested on both CPU and MPS devices
3. ✅ Tested with existing checkpoint (epoch 5)
4. ✅ Instrument-aware split logic tested with sample data
