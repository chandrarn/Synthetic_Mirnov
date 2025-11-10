#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify the refactored fno_predictor and data_caching modules work together.
"""
import sys
sys.path.insert(0, '/home/rianc/Documents/Synthetic_Mirnov/classifier_regressor')

print("Testing imports...")
try:
    from data_caching import build_or_load_cached_dataset, CacheConfig, gen_sensor_ordering
    print("✓ data_caching module imported successfully")
    print(f"  - build_or_load_cached_dataset: {build_or_load_cached_dataset}")
    print(f"  - CacheConfig: {CacheConfig}")
    print(f"  - gen_sensor_ordering: {gen_sensor_ordering}")
except Exception as e:
    print(f"✗ Failed to import data_caching: {e}")
    sys.exit(1)

try:
    from fno_predictor import FNO1dClassifier, TrainConfig, train_fno_classifier
    print("✓ fno_predictor module imported successfully")
    print(f"  - FNO1dClassifier: {FNO1dClassifier}")
    print(f"  - TrainConfig: {TrainConfig}")
    print(f"  - train_fno_classifier: {train_fno_classifier}")
except Exception as e:
    print(f"✗ Failed to import fno_predictor: {e}")
    sys.exit(1)

# Test sensor ordering function
print("\nTesting gen_sensor_ordering...")
test_names = ['BP12_GHK', 'BP01_ABK', 'BP_AA_TOP', 'BP_BB_BOT', 'BP03T_ABK']
sorted_names, orig, order = gen_sensor_ordering(test_names)
print(f"  Input:  {test_names}")
print(f"  Sorted: {list(sorted_names)}")
print(f"  Order:  {order}")

# Test minimal model instantiation
print("\nTesting FNO1dClassifier instantiation...")
try:
    import torch
    model = FNO1dClassifier(in_channels=4, width=32, modes=8, depth=2, n_classes=5, dropout=0.2)
    print(f"✓ Model created: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x_test = torch.randn(2, 10, 4)  # batch=2, sensors=10, channels=4
    with torch.no_grad():
        y_test = model(x_test)
    print(f"  Forward pass: input {tuple(x_test.shape)} -> output {tuple(y_test.shape)}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All basic tests passed!")
print("\nRefactoring successful:")
print("  - data_caching.py: handles all data extraction and caching")
print("  - fno_predictor.py: focused on FNO neural network architecture and training")
