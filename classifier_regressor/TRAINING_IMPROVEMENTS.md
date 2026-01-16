# FNO Training Improvements for Better Generalization

## Problem Analysis

With ~1,800 samples from 240 datasets:
- **Training loss: 1.33, Validation loss: 4.48** → Clear overfitting
- **Validation accuracy: 0.66** → Room for improvement
- **AUC: 0.9** → Model has good discriminative ability but poor calibration

## Dimensional Flow Verification ✓

The FFT is **correctly** performed over the sensor dimension:

1. Input: `(B, S, 4)` where B=batch, S=sensors, 4=channels
2. Line 193: `x.permute(0, 2, 1)` → `(B, 4, S)` 
3. Line 194: `self.lift(x)` → `(B, width, S)`
4. FNOBlock processes: `(B, width, S)`
5. **Line 142: `torch.fft.rfft(x, dim=-1)`** → FFT over **last dimension (S=sensors)** ✓

The architecture correctly applies Fourier transforms over the spatial sensor array.

## Implemented Improvements

### 1. **Enhanced Data Augmentation** (Lines 79-91)
- Increased noise amplitude: `0.05` → `0.08`
- Added rotation-like mixing between real/imag channels (30% probability, ±0.2 rad)
- Helps model learn rotation-invariant features

### 2. **Stronger Regularization**
- **Dropout in FNO blocks**: Added 0.5×dropout to each block (Line 157)
- **BatchNorm in classifier head**: Added after linear layer (Line 177)
- **Increased mask probability**: `0.1` → `0.15` (more aggressive sensor dropout)
- **Higher dropout**: `0.3` → `0.4`
- **Label smoothing**: Added `0.1` for softer targets

### 3. **Better Optimizer & Scheduler**
- Changed from `Adam` to `AdamW` (better weight decay handling)
- Replaced ReduceLROnPlateau with `CosineAnnealingWarmRestarts`
  - T_0=10, T_mult=2 (periodic restarts help escape local minima)
  - eta_min=1e-6 (prevents learning rate from going to zero)

### 4. **Multi-Core CPU Utilization** (Lines 246-251)
- Added `num_workers=4` to DataLoader
- Enabled `pin_memory=True` for faster GPU transfer
- Added `persistent_workers=True` to reuse worker processes

### 5. **Optimized Hyperparameters for Small Dataset**
```python
TrainConfig(
    batch_size=32,        # Smaller batches → more gradient updates
    lr=5e-4,              # Lower LR for stability
    weight_decay=1e-3,    # Stronger L2 regularization
    patience=40,          # More patience with better regularization
    modes=20,             # Slightly more Fourier modes
    width=160,            # Reasonable network width
    dropout=0.4,          # High dropout for 1,800 samples
    label_smoothing=0.1,  # Prevent overconfident predictions
    num_workers=4         # Parallel data loading
)
```

## Expected Improvements

1. **Reduced overfitting**: Gap between train/val loss should decrease
2. **Better validation accuracy**: Target 75-80% (from 66%)
3. **Faster training**: Multi-core loading reduces I/O bottleneck
4. **Better generalization**: Label smoothing + augmentation + dropout

## Usage

```bash
cd /home/rianc/Documents/Synthetic_Mirnov/classifier_regressor
python fno_predictor.py
```

Monitor the train/val loss gap - it should be much smaller now.

## Further Improvements (if needed)

1. **Mix-up augmentation**: Blend samples and labels
2. **Stratified sampling**: Ensure balanced classes in train/val split
3. **Ensemble**: Train 3-5 models with different seeds, average predictions
4. **Cross-validation**: 5-fold CV for better use of limited data
5. **Progressive unfreezing**: Train head first, then unfreeze blocks gradually
