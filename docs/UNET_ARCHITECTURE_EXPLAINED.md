# U-Net Architecture - Detailed Explanation

## Overview: What is a U-Net?

A U-Net is a neural network architecture shaped like the letter "U". It was originally designed for image segmentation but works excellently for diffusion models. The key idea:

1. **Encoder (downsampling)**: Compresses the input to capture high-level features
2. **Bottleneck**: Processes the most compressed representation
3. **Decoder (upsampling)**: Expands back to original size
4. **Skip connections**: Copy features from encoder to decoder at matching resolutions

```
Input [1024, 345]
    ↓
┌───────────────────────────────────────┐
│  ENCODER (Downsampling Path)         │
├───────────────────────────────────────┤
│  [1024, 345] → [256, 345]    ←──────┐│  Skip 1
│       ↓                              ││
│  [256, 345] → [256, 173]             ││
│       ↓                              ││
│  [256, 173] → [512, 173]    ←──────┐││  Skip 2
│       ↓                             │││
│  [512, 173] → [512, 87]             │││
│       ↓                             │││
│  [512, 87] → [1024, 87]     ←──────┐│││  Skip 3
│       ↓                            ││││
│  [1024, 87] → [1024, 44]           ││││
│       ↓                            ││││
│  [1024, 44] → [2048, 44]   ←──────┐││││  Skip 4
│       ↓                           │││││
│  [2048, 44] → [2048, 22]          │││││
└───────────────────────────────────┘│││││
                                     │││││
┌───────────────────────────────────┐│││││
│  BOTTLENECK                       ││││││
├───────────────────────────────────┤│││││
│  [2048, 22] → [2048, 22]          ││││││
│  (ResBlock + Attention + ResBlock)││││││
└───────────────────────────────────┘│││││
                                     │││││
┌───────────────────────────────────┐│││││
│  DECODER (Upsampling Path)        ││││││
├───────────────────────────────────┤│││││
│  [2048, 22] + Skip 4 ─────────────┘││││
│       ↓                             ││││
│  [4096, 22] → [2048, 22]            ││││
│       ↓                             ││││
│  [2048, 22] → [2048, 44]            ││││
│       ↓                             ││││
│  [2048, 44] + Skip 3 ───────────────┘││
│       ↓                              ││
│  [3072, 44] → [1024, 44]             ││
│       ↓                              ││
│  [1024, 44] → [1024, 87]             ││
│       ↓                              ││
│  [1024, 87] + Skip 2 ────────────────┘│
│       ↓                               │
│  [1536, 87] → [512, 87]               │
│       ↓                               │
│  [512, 87] → [512, 173]               │
│       ↓                               │
│  [512, 173] + Skip 1 ─────────────────┘
│       ↓
│  [768, 173] → [256, 173]
│       ↓
│  [256, 173] → [256, 345]
│       ↓
│  [256, 345] → [1024, 345]
└───────────────────────────────────┘
        ↓
Output [1024, 345] (predicted noise)
```

---

## Why 360M Parameters?

The parameter count comes from:

1. **Residual blocks** (majority of parameters):
   - Multiple conv layers at different channel sizes
   - Example: One conv layer at [2048, 2048] has ~12M parameters

2. **Attention blocks**:
   - QKV projections: 2048 channels × 3 × 2048 = ~12M parameters each

3. **Channel multipliers**: (1, 2, 4, 8)
   - Base: 256 channels
   - Level 1: 256 channels
   - Level 2: 512 channels
   - Level 3: 1024 channels
   - Level 4: 2048 channels

**Calculation example for one residual block at 2048 channels**:
```
Conv1: 2048 × 2048 × 3 (kernel) = 12,583,936 params
Conv2: 2048 × 2048 × 3 (kernel) = 12,583,936 params
Embedding projection: 1024 × 2048 = 2,097,152 params
Total per block: ~27M params
```

With 8 levels × 2 blocks × 2 paths (down+up) = 32 residual blocks
Plus attention blocks and projections → **~360M parameters**

---

## 1. Conditioning Mechanisms

The U-Net needs to know **what** to generate. We condition it on 4 things:

### A. Timestep Embedding (Sinusoidal)

**Purpose**: Tell the model which diffusion step we're at (t=0 to t=1000)

**How it works**:
```python
# Timestep t=500 becomes a 512-dimensional vector
t = 500

# Create sinusoidal features (like in Transformers)
# Uses different frequencies to encode position
half_dim = 256
freqs = exp(arange(256) * -log(10000) / 255)
# freqs = [1.0, 0.956, 0.914, ..., 0.0001]

embedding = [
    sin(500 * 1.0),    # Low frequency
    cos(500 * 1.0),
    sin(500 * 0.956),
    cos(500 * 0.956),
    ...
    sin(500 * 0.0001),  # High frequency
    cos(500 * 0.0001),
]
# Result: [512] unique vector for t=500
```

**Why sinusoidal?**
- Each timestep gets a unique pattern
- Similar timesteps have similar embeddings
- Model learns: "high t = very noisy, low t = almost clean"

**Then we project through MLP**:
```python
t_emb = Linear(512 → 2048)(embedding)
t_emb = SiLU(t_emb)  # Activation
t_emb = Linear(2048 → 512)(t_emb)
# Final: [512] rich timestep embedding
```

### B. MIDI Note Embedding

**Purpose**: Tell the model which pitch to generate (C0 = 0.0, C8 = 1.0)

**Example**:
```python
midi_note = 60  # Middle C
normalized = (60 - 21) / (108 - 21) = 0.448  # Normalize to [0, 1]

# Project through MLP
midi_emb = Linear(1 → 170)(0.448)
midi_emb = SiLU(midi_emb)
midi_emb = Linear(170 → 170)(midi_emb)
# Result: [170] vector representing "Middle C"
```

### C. Velocity Embedding

**Purpose**: Tell the model how loud the note should be (0-127 → 0.0-1.0)

**Example**:
```python
velocity = 96  # Forte (loud)
normalized = 96 / 127 = 0.756

# Project through MLP
vel_emb = Linear(1 → 170)(0.756)
vel_emb = SiLU(vel_emb)
vel_emb = Linear(170 → 170)(vel_emb)
# Result: [170] vector representing "loud"
```

### D. Input Sample Embedding

**Purpose**: Encode the 1-5 input samples to preserve their timbre

**Process**:
1. **Encode each input sample** (InputSampleEncoder):
   ```python
   Input: [3 samples, each 176,400 samples at 44.1kHz]
   ↓ CNN encoder
   Output: [3 embeddings, each 128-dim]
   ```

2. **Aggregate** (via attention or mean):
   ```python
   [emb1, emb2, emb3] → Attention/Mean → [128] single embedding
   ```

3. **Project**:
   ```python
   input_emb = Linear(128 → 172)(aggregated)
   input_emb = SiLU(input_emb)
   input_emb = Linear(172 → 172)(input_emb)
   # Result: [172] vector capturing input timbre
   ```

### Combined Conditioning

All embeddings are concatenated:
```python
t_emb:     [512]
midi_emb:  [170]
vel_emb:   [170]
input_emb: [172]
──────────────────
combined:  [1024]  ← This conditions every residual block
```

---

## 2. Architecture Features

### A. Residual Blocks with Group Normalization

**What is a residual block?**
```python
Input x: [batch, channels, time]
    ↓
┌─────────────────────────────────┐
│ GroupNorm → SiLU → Conv         │
│         ↓                        │
│ Add embedding (timestep+cond)   │
│         ↓                        │
│ GroupNorm → SiLU → Conv         │
└─────────────────────────────────┘
    ↓
Add input x (skip connection)
    ↓
Output: [batch, channels, time]
```

**Key components**:

1. **Group Normalization** (instead of Batch Normalization):
   ```python
   # Split channels into 32 groups
   channels = 1024
   groups = 32
   channels_per_group = 32

   # Normalize each group independently
   for group in range(32):
       normalize(x[:, group*32:(group+1)*32, :])
   ```

   **Why GroupNorm?**
   - Works well with small batch sizes (we use batch=16)
   - More stable than BatchNorm for generative models
   - Each group learns different features

2. **SiLU activation** (Sigmoid Linear Unit):
   ```python
   SiLU(x) = x * sigmoid(x)

   # Example:
   x = 2.0
   SiLU(2.0) = 2.0 * sigmoid(2.0) = 2.0 * 0.88 = 1.76
   ```

   **Why SiLU?**
   - Smoother than ReLU
   - Better gradient flow
   - Works well in diffusion models

3. **Embedding injection**:
   ```python
   h = conv1(x)  # [batch, 1024, 345]

   # Project conditioning to match channels
   emb_out = Linear(1024 → 1024)(combined_embedding)  # [batch, 1024]
   emb_out = emb_out[:, :, None]  # [batch, 1024, 1]

   # Add to every time step
   h = h + emb_out  # Broadcasting: [batch, 1024, 345]
   ```

   This tells the model: "Generate with THESE conditions"

4. **Residual connection**:
   ```python
   output = processed_features + input
   ```

   **Why residuals?**
   - Helps gradients flow during training
   - Allows learning "refinements" instead of full transformations
   - Essential for deep networks (our U-Net has ~50 layers)

### B. Self-Attention Blocks

**Purpose**: Let the model look at the entire time sequence, not just local neighbors

**How it works**:
```python
Input: [batch, 1024 channels, 345 time steps]

# Create Queries, Keys, Values
Q = Conv(x)  # [batch, 1024, 345]
K = Conv(x)  # [batch, 1024, 345]
V = Conv(x)  # [batch, 1024, 345]

# Reshape for multi-head attention (4 heads)
# [batch, 4 heads, 256 channels/head, 345 time]

# Compute attention: "Which time steps are related?"
scores = Q @ K.transpose()  # [batch, 4, 345, 345]
# Each of 345 time steps computes similarity to all others

attention = softmax(scores / sqrt(256))  # Normalize

# Apply attention to values
output = attention @ V  # [batch, 4, 345, 256]

# Merge heads back
output = reshape([batch, 1024, 345])
```

**Concrete example**:
```python
# Piano sample with attack and sustain
Time steps:  [0-50: attack, 51-345: sustain]

# Attention might learn:
Attention[10, 20] = 0.8   # Attack correlates with attack
Attention[10, 200] = 0.1  # Attack weakly correlates with sustain
Attention[200, 300] = 0.9 # Sustain correlates with sustain
```

**Why attention at specific resolutions?**
```python
attention_resolutions=(2,)  # Only at 2x downsampling
```

- **At full resolution** (345 time steps): 345×345 = 119,025 attention pairs → SLOW
- **At 2x downsampled** (173 time steps): 173×173 = 29,929 pairs → Faster
- **Trade-off**: Full resolution = expensive but detailed
              Lower resolution = cheaper but less detail

We use attention at 2x downsampling (middle level) as a sweet spot.

### C. Skip Connections

**The Problem**:
When you downsample and upsample, you lose fine details:
```
Input [1024, 345]
  ↓ downsample
[2048, 22]  ← Lost 345 → 22 resolution!
  ↓ upsample
[1024, 345]  ← Can't perfectly reconstruct lost details
```

**The Solution**:
Copy features from encoder to decoder at matching resolutions:

```python
# ENCODER
level1_out = ResBlock(input)      # [256, 345]
level2_out = Downsample(level1)   # [256, 173]
level2_out = ResBlock(level2_out) # [512, 173]
...

# DECODER (later)
x = Upsample(bottleneck)          # [512, 173]

# Concatenate with matching encoder features
x = concat([x, level2_out], dim=1)  # [512, 173] + [512, 173] = [1024, 173]
x = ResBlock(x)  # Process combined features
```

**Why this helps**:
- **Encoder features**: "What was at this resolution?"
- **Decoder features**: "What I've upsampled so far"
- **Combined**: Best of both → better reconstruction

**Dimension handling**:
```python
# Problem: Odd dimensions don't divide evenly
345 → downsample by 2 → 172 (not 172.5!)
172 → upsample by 2 → 344 (not 345!)

# Solution: Pad when concatenating
if h.shape[-1] != skip.shape[-1]:  # 344 != 345
    diff = 345 - 344 = 1
    h = pad(h, (0, 1))  # Pad 1 time step

# Now can concatenate
h = concat([h, skip])  # Both [?, 345]
```

### D. Handling DAC Latent Space [batch, 1024, 345]

**What is this shape?**
```python
batch = 2           # Process 2 samples at once
channels = 1024     # DAC's latent has 1024 feature channels
time = 345          # 4 seconds of audio → 345 time steps in latent

# Total: [2, 1024, 345]
```

**Why these dimensions?**

1. **1024 channels**:
   - DAC compresses audio into 1024 feature dimensions
   - Each channel captures different aspects:
     - Channel 0: Maybe low frequencies
     - Channel 500: Maybe transients
     - Channel 1000: Maybe harmonic content

2. **345 time steps**:
   - Original audio: 176,400 samples (4 sec × 44,100 Hz)
   - DAC compresses by ~512x: 176,400 / 512 ≈ 345 time steps
   - Each time step represents ~11.6ms of audio

3. **Continuous vs Discrete**:
   ```python
   # DAC Encoding process:
   audio [1, 176400]
       ↓ Encoder
   z (continuous) [1024, 345]  ← We use this!
       ↓ VectorQuantization
   codes (discrete) [9, 345]   ← 9 codebooks
   ```

   We operate on **continuous z** before quantization because:
   - Diffusion needs smooth latent space
   - Quantized codes are discrete → hard to add noise
   - Continuous latent → can smoothly interpolate

**How U-Net processes it**:

```python
# Input
x = [batch, 1024, 345]

# Level 1: Same resolution
x = Conv1d(1024 → 256)(x)  # [batch, 256, 345]
x = ResBlock(x)             # [batch, 256, 345]

# Level 2: Downsample by 2
x = Downsample(x)           # [batch, 256, 173]
x = ResBlock(256 → 512)(x)  # [batch, 512, 173]

# Level 3: Downsample by 2
x = Downsample(x)           # [batch, 512, 87]
x = ResBlock(512 → 1024)(x) # [batch, 1024, 87]

# Level 4: Downsample by 2
x = Downsample(x)           # [batch, 1024, 44]
x = ResBlock(1024 → 2048)(x)# [batch, 2048, 44]

# Bottleneck: Process at lowest resolution
x = ResBlock(x)             # [batch, 2048, 22]
x = Attention(x)            # Look at all 22 time steps
x = ResBlock(x)             # [batch, 2048, 22]

# Then upsample back to [batch, 1024, 345]
```

**Why this architecture for audio?**

1. **Hierarchical processing**:
   - High resolution (345): Fine details, transients
   - Mid resolution (87): Harmonic structure
   - Low resolution (22): Overall envelope, timbre

2. **Efficient**:
   - Don't need to process all 345 time steps at full 1024 channels
   - Downsample to reduce computation
   - Upsample with skip connections to restore details

3. **Matches DAC's structure**:
   - DAC already compressed audio hierarchically
   - U-Net's levels align with acoustic features
   - Natural fit for audio generation

---

## Putting It All Together

**Training step**:
```python
# 1. Encode audio with DAC
audio = [batch, 1, 176400]  # Raw audio
z = DAC.encode(audio)       # [batch, 1024, 345]

# 2. Add noise (forward diffusion)
t = random.randint(0, 1000)  # Random timestep
noise = randn_like(z)
z_noisy = sqrt(alpha_t) * z + sqrt(1 - alpha_t) * noise

# 3. Prepare conditioning
t_emb = timestep_embed(t)              # [512]
midi_emb = midi_embed(midi_note)       # [170]
vel_emb = velocity_embed(velocity)     # [170]
input_emb = input_encoder(ref_samples) # [172]
combined = concat([t_emb, midi_emb, vel_emb, input_emb])  # [1024]

# 4. Predict noise
predicted_noise = UNet(z_noisy, combined)  # [batch, 1024, 345]

# 5. Compute loss
loss = MSE(predicted_noise, noise)

# 6. Backprop and update weights
loss.backward()
optimizer.step()
```

**Generation**:
```python
# Start from pure noise
z = randn([batch, 1024, 345])

# Denoise for 1000 steps
for t in [999, 998, 997, ..., 1, 0]:
    # Predict noise
    predicted_noise = UNet(z, conditioning)

    # Remove a bit of noise
    z = denoise_step(z, predicted_noise, t)

# Decode with DAC
audio = DAC.decode(z)  # [batch, 1, 176400]
```

---

## Summary

The U-Net is the "brain" of the diffusion model:

1. **360M parameters**: Large enough to learn complex audio patterns
2. **Conditioning**: Knows WHEN (timestep), WHAT pitch (MIDI), HOW loud (velocity), and WHAT timbre (input samples)
3. **Residual blocks**: Deep architecture with stable training
4. **Attention**: Captures long-range dependencies in audio
5. **Skip connections**: Preserves fine details during up/downsampling
6. **DAC latent space**: Operates on compressed, meaningful audio representation

It takes noisy latent code and conditioning → predicts what noise to remove → gradually reveals clean audio matching the conditions.
