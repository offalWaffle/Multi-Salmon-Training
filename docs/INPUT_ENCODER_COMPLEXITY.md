# InputSampleEncoder: Single vs Multiple Samples - Complexity Analysis

## Overview

The InputSampleEncoder can handle 1-5 input samples. Let's compare the complexity of single vs multiple sample handling.

---

## Single Sample (Simplest Case)

### Conceptual Flow
```
Input: 1 audio sample [176,400 samples]
  ↓ CNN Encoder
Output: 1 embedding [128 dims]

Done! No aggregation needed.
```

### Implementation (if we only supported 1 sample)
```python
class SimpleInputEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        # Just the CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=16),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=32, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, audio):
        """
        Args:
            audio: [batch, 1, 176400]
        Returns:
            embedding: [batch, 128]
        """
        return self.encoder(audio)  # That's it!
```

### Complexity
- **Conceptual**: ⭐ (very simple)
- **Implementation**: ⭐ (20 lines of code)
- **Computation**: ~3M parameters, single forward pass

---

## Multiple Samples (Current Implementation)

### Conceptual Flow
```
Input: 3 audio samples [3 × 176,400 samples]
  ↓ CNN Encoder (applied to each)
3 embeddings: [3 × 128 dims]
  ↓ Aggregation (mean or attention)
Output: 1 embedding [128 dims]

Need to handle:
1. Variable number of inputs (1-5)
2. Padding for batch processing
3. Masking to ignore padding
4. Aggregation strategy
```

### Implementation Complexity Breakdown

#### 1. **Encoding Each Sample** (Same as single)
```python
# This part is identical to single sample
self.encoder = nn.Sequential(...)  # Same CNN

# But now we process multiple samples
input_samples: [batch, max_samples, 1, 176400]
# Example: [2, 5, 1, 176400] = 2 batch items, up to 5 samples each

# Reshape to process all at once
flat = input_samples.reshape(batch * max_samples, 1, 176400)
# [2*5, 1, 176400] = [10, 1, 176400]

embeddings = self.encoder(flat)  # [10, 128]

# Reshape back
embeddings = embeddings.reshape(batch, max_samples, 128)
# [2, 5, 128]
```

**Complexity added**: ⭐⭐
- Need reshape operations
- Process batch*max_samples instead of just batch
- Need to track which samples are real vs padding

#### 2. **Handling Variable Number of Inputs**

The problem:
```
Batch item 1: 3 real samples + 2 padding
Batch item 2: 1 real sample + 4 padding

But tensors must be rectangular!
```

Solution: Use a `num_valid` mask
```python
input_samples = torch.zeros(batch, max_samples, 1, 176400)
num_valid = torch.zeros(batch, dtype=torch.long)

# Fill in real samples
input_samples[0, :3] = [sample1, sample2, sample3]  # 3 real
input_samples[1, :1] = [sample1]                     # 1 real
num_valid = [3, 1]
```

**Complexity added**: ⭐⭐
- Need to track valid counts
- Need padding logic
- User must provide or we must infer num_valid

#### 3. **Mean Aggregation** (Simpler Option)

```python
def aggregate_mean(embeddings, num_valid):
    """
    Args:
        embeddings: [batch, max_samples, 128]
        num_valid: [batch] - e.g., [3, 1]

    Returns:
        aggregated: [batch, 128]
    """
    batch, max_samples, embed_dim = embeddings.shape

    # Create mask for valid samples
    # [0, 1, 2, 3, 4] < [3, 1]
    # [[T, T, T, F, F],
    #  [T, F, F, F, F]]
    mask = torch.arange(max_samples)[None, :] < num_valid[:, None]
    mask = mask.unsqueeze(-1).float()  # [batch, max_samples, 1]

    # Zero out padding
    masked = embeddings * mask  # [batch, max_samples, 128]

    # Sum and divide by count
    sum_emb = masked.sum(dim=1)  # [batch, 128]
    count = num_valid.unsqueeze(-1).float()  # [batch, 1]

    aggregated = sum_emb / count.clamp(min=1)  # [batch, 128]

    return aggregated
```

**Example**:
```python
# Batch item 0: [emb1, emb2, emb3, padding, padding]
# num_valid[0] = 3

# After masking:
masked[0] = [emb1, emb2, emb3, zeros, zeros]

# Sum:
sum_emb[0] = emb1 + emb2 + emb3

# Mean:
aggregated[0] = (emb1 + emb2 + emb3) / 3
```

**Complexity added**: ⭐⭐⭐
- Need to understand broadcasting for mask creation
- Need to handle division by zero (clamp)
- Need to understand masked operations

#### 4. **Attention Aggregation** (More Complex Option)

```python
class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Learnable query: "What's the aggregate embedding?"
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, embeddings, num_valid):
        """
        Args:
            embeddings: [batch, max_samples, 128]
            num_valid: [batch]

        Returns:
            aggregated: [batch, 128]
        """
        batch, max_samples, embed_dim = embeddings.shape

        # Create padding mask
        # True = "ignore this position"
        key_padding_mask = torch.arange(max_samples)[None, :] >= num_valid[:, None]
        # [[F, F, F, T, T],  # First 3 valid
        #  [F, T, T, T, T]]  # First 1 valid

        # Expand query to batch size
        query = self.query.expand(batch, -1, -1)  # [batch, 1, 128]

        # Attention: query attends to all embeddings
        output, weights = self.attention(
            query,              # What we want to know
            embeddings,         # Keys: what's available
            embeddings,         # Values: what to aggregate
            key_padding_mask=key_padding_mask,  # Ignore padding
        )

        # Remove sequence dimension
        aggregated = output.squeeze(1)  # [batch, 128]

        return aggregated
```

**What attention does**:
```python
# Example: 3 input samples
embeddings = [emb1, emb2, emb3, pad, pad]
num_valid = 3

# Attention computes:
attention_weights = [
    0.4,  # 40% weight on emb1
    0.35, # 35% weight on emb2
    0.25, # 25% weight on emb3
    0.0,  # 0% on padding (masked)
    0.0,  # 0% on padding (masked)
]

# Weighted sum:
aggregated = 0.4*emb1 + 0.35*emb2 + 0.25*emb3

# Weights are LEARNED (not fixed like mean)
# Model learns: "Pay more attention to certain samples"
```

**Complexity added**: ⭐⭐⭐⭐⭐
- Need to understand attention mechanism
- Need learnable query parameter
- Need to understand key padding masks
- More parameters (~130K additional for attention layers)
- More computation during forward pass

---

## Complexity Comparison Table

| Aspect | Single Sample | Multiple (Mean) | Multiple (Attention) |
|--------|--------------|----------------|---------------------|
| **Conceptual** | ⭐ Simple | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Complex |
| **Code Lines** | ~20 | ~50 | ~80 |
| **Parameters** | 3M | 3M | 3.13M (+130K) |
| **Forward Pass** | 1x encoder | 1x encoder + masking | 1x encoder + attention |
| **Memory** | [B, 1, 176400] | [B, 5, 1, 176400] | [B, 5, 1, 176400] |
| **Speed** | Fastest | Fast | Slower |
| **Flexibility** | None | Some | High |
| **Need Padding** | ❌ | ✅ | ✅ |
| **Need Masking** | ❌ | ✅ | ✅ |
| **Learnable Agg** | N/A | ❌ | ✅ |

---

## Implementation Differences: Code Comparison

### Single Sample Only
```python
# VERY SIMPLE
encoder = InputEncoder(embed_dim=128)

audio = torch.randn(2, 1, 176400)  # 2 samples
embedding = encoder(audio)  # [2, 128]

# Done!
```

### Multiple Samples (Mean)
```python
# MODERATE COMPLEXITY
encoder = InputEncoder(embed_dim=128, aggregation='mean')

# Need to prepare inputs carefully
batch = []
for item in dataset:
    samples = item['input_samples']  # Variable number

    # Pad to max_samples=5
    padded = torch.zeros(5, 1, 176400)
    padded[:len(samples)] = samples
    batch.append(padded)

input_samples = torch.stack(batch)  # [B, 5, 1, 176400]
num_valid = torch.tensor([3, 1, 5, 2])  # Track real counts

embedding = encoder(input_samples, num_valid)  # [B, 128]
```

### Multiple Samples (Attention)
```python
# MOST COMPLEX
encoder = InputEncoder(
    embed_dim=128,
    aggregation='attention',
    num_heads=4
)

# Same input preparation as mean...
input_samples = torch.stack(batch)  # [B, 5, 1, 176400]
num_valid = torch.tensor([3, 1, 5, 2])

# But now attention learns how to aggregate
embedding = encoder(input_samples, num_valid)  # [B, 128]

# Attention weights are learned during training
# Model learns: "Pay more attention to certain samples"
```

---

## Why Support Multiple Samples?

### Use Case Comparison

#### Single Sample
```python
# User provides ONE reference sample
input = piano_middle_c.wav

# Model learns: "Generate sounds like this one sample"
# Problem: Limited timbral information
```

#### Multiple Samples
```python
# User provides MULTIPLE reference samples
inputs = [
    piano_c3_soft.wav,   # Low, quiet
    piano_c5_loud.wav,   # Mid, loud
    piano_c7_soft.wav,   # High, quiet
]

# Model learns: "Generate sounds that combine characteristics of all"
# Benefit: Better timbral consistency across range
```

**Example scenario**:
```
Goal: Generate full piano keyboard from references

With 1 sample:
  Input: C4 recording
  Generate: C1, C2, C3, C4✓, C5?, C6?, C7?
  Problem: C7 may sound too different from C4

With 3 samples:
  Input: C2 recording, C4 recording, C6 recording
  Generate: C1, C2✓, C3, C4✓, C5, C6✓, C7
  Benefit: Interpolate timbral characteristics across range
```

---

## Memory & Computation Impact

### Memory Usage
```python
Single sample:
  Input: [batch=16, 1, 176400] = ~11 MB
  Embeddings: [16, 128] = ~8 KB

Multiple samples:
  Input: [batch=16, 5, 1, 176400] = ~56 MB (5x larger!)
  Intermediate: [16*5, 128] = ~40 KB
  Embeddings: [16, 128] = ~8 KB
```

### Computation Time
```python
Single sample:
  Encode 16 samples = 16 forward passes

Multiple samples (max 5):
  Encode 16*5 = 80 forward passes
  + Aggregation step

Total: ~5x slower for encoding
```

**However**: This is only for the input encoder, which runs ONCE per training step. The U-Net dominates training time, so the impact is minimal (<5% slowdown).

---

## Implementation Tips

### Starting Simple
```python
# Phase 1: Support only single sample
class SimpleEncoder:
    def forward(self, audio):
        return self.encoder(audio)

# Easy to implement, test, and debug
```

### Adding Multiple Samples
```python
# Phase 2: Add mean aggregation
class MultiEncoder:
    def forward(self, audio, num_valid):
        # Encode all
        embs = self.encoder(audio)

        # Mean aggregate
        mask = create_mask(num_valid)
        return (embs * mask).sum(1) / num_valid.unsqueeze(-1)

# Moderate complexity, significant benefit
```

### Advanced Aggregation
```python
# Phase 3: Add attention aggregation
class AdvancedEncoder:
    def forward(self, audio, num_valid):
        embs = self.encoder(audio)

        # Learnable attention
        query = self.query.expand(batch, -1, -1)
        output, _ = self.attention(query, embs, embs, mask=...)
        return output.squeeze(1)

# Most complex, best quality
```

---

## Recommendation

**For prototyping**: Start with single sample
- Get the rest of the pipeline working first
- Add multiple sample support later

**For production**: Use multiple samples with mean aggregation
- Good balance of complexity vs benefit
- Significantly improves timbral consistency
- Not much slower than single sample

**For research**: Try attention aggregation
- Best quality (potentially)
- Adds ~5% to training time
- Model can learn which samples to emphasize

---

## Current Implementation

Our `InputSampleEncoder` supports both:
```python
# Use mean (simple)
encoder = InputSampleEncoder(aggregation='mean')

# Use attention (complex)
encoder = InputSampleEncoder(aggregation='attention', num_heads=4)
```

The complexity is **already handled** for you! You just need to:
1. Provide input samples (1-5)
2. Specify num_valid
3. Choose aggregation method

The encoder handles all the masking, padding, and aggregation internally.
