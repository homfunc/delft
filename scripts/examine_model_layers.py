#!/usr/bin/env python3
"""
Script to examine the layers in model_weights.hdf5 file
This will help identify why there are 2 layers saved when only 1 is expected.
"""

import h5py
import numpy as np
import sys
import os
import argparse

def examine_hdf5_structure(filepath):
    """Examine the structure of an HDF5 file recursively."""
    print(f"Examining HDF5 file: {filepath}")
    print("=" * 50)
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist!")
        return
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("Root level keys:")
            for key in f.keys():
                print(f"  - {key}")
            print()
            
            # Recursively examine the structure
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
                    # Print attributes if any
                    if obj.attrs:
                        print(f"{indent}  Attributes:")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}    {attr_name}: {attr_value}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name}")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Dtype: {obj.dtype}")
                    print(f"{indent}  Size: {obj.size}")
                    # Print attributes if any
                    if obj.attrs:
                        print(f"{indent}  Attributes:")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"{indent}    {attr_name}: {attr_value}")
                    
                    # For smaller datasets, show some data
                    if obj.size <= 10:
                        print(f"{indent}  Data: {obj[...]}")
                    elif len(obj.shape) > 0 and obj.shape[0] <= 5:
                        print(f"{indent}  First few elements: {obj[:min(5, obj.shape[0])]}")
            
            print("Complete structure:")
            f.visititems(print_structure)
            
            # Look specifically for model-related information
            print("\n" + "=" * 50)
            print("ANALYSIS:")
            
            # Check for common Keras/TensorFlow patterns
            if 'model_weights' in f:
                print("Found 'model_weights' group")
                model_weights = f['model_weights']
                layer_names = []
                for key in model_weights.keys():
                    layer_names.append(key)
                    print(f"Layer: {key}")
                
                print(f"\nTotal layers found: {len(layer_names)}")
                print(f"Layer names: {layer_names}")
                
                # Examine each layer in detail
                for layer_name in layer_names:
                    layer_group = model_weights[layer_name]
                    print(f"\nDetailed info for layer '{layer_name}':")
                    if hasattr(layer_group, 'keys'):
                        weight_names = list(layer_group.keys())
                        print(f"  Weight components: {weight_names}")
                        
                        for weight_name in weight_names:
                            weight_data = layer_group[weight_name]
                            print(f"    {weight_name}: shape={weight_data.shape}, dtype={weight_data.dtype}")
                            
                            # Show weight statistics
                            data = weight_data[...]
                            print(f"      Stats: min={np.min(data):.6f}, max={np.max(data):.6f}, mean={np.mean(data):.6f}")
            
            # Check for other common patterns
            if 'layer_names' in f.attrs:
                print(f"\nLayer names from attributes: {f.attrs['layer_names']}")
            
            if 'keras_version' in f.attrs:
                print(f"Keras version: {f.attrs['keras_version']}")
                
            if 'backend' in f.attrs:
                print(f"Backend: {f.attrs['backend']}")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Examine the layers in model_weights.hdf5 file")
    parser.add_argument('filepath', type=str, help='Path to the HDF5 file')
    args = parser.parse_args()
    filepath = args.filepath
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        print(f"Current directory: {os.getcwd()}")
        return
    
    examine_hdf5_structure(filepath)
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("1. Check if one layer is just a wrapper or container")
    print("2. Look for duplicate layers or versioning artifacts")
    print("3. Verify if the model architecture matches the saved weights")
    print("4. Consider if there are input/output layers being counted")

if __name__ == "__main__":
    main()
