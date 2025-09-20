#!/usr/bin/env python3
"""
Helper script to repair Windows wheels for CUDA projects.
This script ensures that CUDA runtime DLLs are properly bundled with the wheel.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import zipfile
import tempfile
import shutil


def find_cuda_dlls(cuda_path):
    """Find required CUDA DLLs in the CUDA installation directory."""
    cuda_bin = Path(cuda_path) / "bin"
    required_dlls = [
        "cudart64_*.dll",
        "cublas64_*.dll", 
        "cublasLt64_*.dll",
        "cufft64_*.dll",
        "curand64_*.dll",
        "cusparse64_*.dll",
        "cusolver64_*.dll",
        "nvrtc64_*.dll",
    ]
    
    found_dlls = []
    for pattern in required_dlls:
        dlls = list(cuda_bin.glob(pattern))
        found_dlls.extend(dlls)
    
    return found_dlls


def repair_wheel(wheel_path, cuda_path, output_dir):
    """
    Repair a Windows wheel by bundling CUDA DLLs.
    
    Args:
        wheel_path: Path to the input wheel
        cuda_path: Path to CUDA installation
        output_dir: Directory to save the repaired wheel
    """
    wheel_path = Path(wheel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CUDA DLLs
    cuda_dlls = find_cuda_dlls(cuda_path)
    if not cuda_dlls:
        print(f"Warning: No CUDA DLLs found in {cuda_path}/bin")
        # If no DLLs found, just copy the wheel as-is
        shutil.copy2(wheel_path, output_dir / wheel_path.name)
        return output_dir / wheel_path.name
    
    print(f"Found {len(cuda_dlls)} CUDA DLLs to bundle")
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract wheel
        print(f"Extracting wheel: {wheel_path}")
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(temp_path)
        
        # Find the module directory (usually ccc_cuda_ext)
        module_dirs = list(temp_path.glob("ccc_cuda_ext*"))
        if not module_dirs:
            # Try looking for any .pyd files
            pyd_files = list(temp_path.glob("**/*.pyd"))
            if pyd_files:
                module_dir = pyd_files[0].parent
            else:
                print("Warning: Could not find module directory in wheel")
                module_dir = temp_path
        else:
            module_dir = module_dirs[0]
        
        # Copy CUDA DLLs to module directory
        print(f"Copying CUDA DLLs to {module_dir.name}")
        for dll in cuda_dlls:
            target = module_dir / dll.name
            shutil.copy2(dll, target)
            print(f"  - {dll.name}")
        
        # Create new wheel
        output_wheel = output_dir / wheel_path.name
        print(f"Creating repaired wheel: {output_wheel}")
        
        with zipfile.ZipFile(output_wheel, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_path)
                    zf.write(file_path, arcname)
        
        return output_wheel


def main():
    parser = argparse.ArgumentParser(
        description="Repair Windows wheels by bundling CUDA DLLs"
    )
    parser.add_argument(
        "wheel",
        help="Path to the wheel file to repair"
    )
    parser.add_argument(
        "--cuda-path",
        default=os.environ.get("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"),
        help="Path to CUDA installation (default: from CUDA_PATH env or standard location)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for repaired wheel (default: current directory)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.wheel).exists():
        print(f"Error: Wheel file not found: {args.wheel}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.cuda_path).exists():
        print(f"Error: CUDA path not found: {args.cuda_path}", file=sys.stderr)
        print("Please set CUDA_PATH environment variable or use --cuda-path", file=sys.stderr)
        sys.exit(1)
    
    try:
        repaired_wheel = repair_wheel(args.wheel, args.cuda_path, args.output_dir)
        print(f"Successfully repaired wheel: {repaired_wheel}")
    except Exception as e:
        print(f"Error repairing wheel: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()