"""
GPU and CUDA verification script for image similarity search system.
Checks GPU availability, CUDA installation, and optimal settings.
"""

import torch
import torchvision
import sys
import psutil
import time

def check_gpu_setup():
    """Check GPU and CUDA setup for optimal performance."""
    print("=== GPU and CUDA Setup Check ===")
    print()
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print()
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        # GPU details for each device
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print()
            
        # Current GPU memory usage
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print()
        
        # Test GPU performance
        print("Testing GPU performance...")
        test_gpu_performance()
        
    else:
        print("CUDA not available. Reasons could be:")
        print("1. No NVIDIA GPU installed")
        print("2. CUDA drivers not installed")
        print("3. PyTorch installed without CUDA support")
        print("4. Incompatible CUDA/PyTorch versions")
        print()
        
        # Check if NVIDIA GPU exists
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("NVIDIA GPU detected but CUDA not accessible to PyTorch")
                print("nvidia-smi output:")
                print(result.stdout)
            else:
                print("No NVIDIA GPU detected or nvidia-smi not found")
        except:
            print("Could not check for NVIDIA GPU")
    
    # System memory info
    print("System Information:")
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print()
    
    # Recommendations
    print("Performance Recommendations:")
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            print("✅ Excellent GPU memory (>=8GB) - Use batch sizes 50-100")
        elif gpu_memory >= 4:
            print("✅ Good GPU memory (>=4GB) - Use batch sizes 25-50")
        else:
            print("⚠️  Limited GPU memory (<4GB) - Use batch sizes 10-25")
            
        print("✅ Use GPU acceleration for 10-50x speedup")
        print("✅ Enable mixed precision training if supported")
    else:
        print("⚠️  No GPU acceleration - Processing will be slower")
        print("💡 Consider installing CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    return cuda_available

def test_gpu_performance():
    """Run a quick performance test on GPU."""
    if not torch.cuda.is_available():
        return
        
    device = torch.cuda.current_device()
    
    # Warm up GPU
    dummy_tensor = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    
    # Test matrix multiplication (common in neural networks)
    sizes = [1000, 2000, 4000]
    for size in sizes:
        try:
            # Create test tensors
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            # Time the operation
            torch.cuda.synchronize()
            start_time = time.time()
            
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate performance metrics
            operations = 2 * size**3  # Approximate FLOPs for matrix multiplication
            gflops = operations / elapsed / 1e9
            
            print(f"  Matrix {size}x{size}: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
            
            # Clean up
            del a, b, c
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"  Matrix {size}x{size}: Failed ({e})")
            torch.cuda.empty_cache()
            break

def recommend_optimal_settings():
    """Recommend optimal settings based on hardware."""
    print("\n=== Optimal Settings Recommendations ===")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_properties(0).name
        
        print(f"Detected GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print()
        
        # Batch size recommendations
        if gpu_memory >= 12:
            batch_size = 64
            embed_batch = 128
        elif gpu_memory >= 8:
            batch_size = 32
            embed_batch = 64
        elif gpu_memory >= 6:
            batch_size = 16
            embed_batch = 32
        elif gpu_memory >= 4:
            batch_size = 8
            embed_batch = 16
        else:
            batch_size = 4
            embed_batch = 8
            
        print(f"Recommended settings for your GPU:")
        print(f"  --batch-size {batch_size}")
        print(f"  Embedding batch size: {embed_batch}")
        
        # Model recommendations
        if gpu_memory >= 6:
            print(f"  --model resnet50 (recommended)")
            print(f"  Can handle full ResNet50 with good batch sizes")
        else:
            print(f"  --model resnet18 (recommended for limited memory)")
            print(f"  Consider using ResNet18 for better memory efficiency")
            
        # Additional optimizations
        print("\nOptimization flags to use:")
        print("  Set DEVICE=cuda in .env file")
        print("  Use mixed precision: export TORCH_AUTOCAST=1")
        
    else:
        print("No GPU detected - CPU recommendations:")
        print("  --batch-size 8")
        print("  --model resnet18 (faster on CPU)")
        print("  Consider processing fewer images for testing")

def install_cuda_pytorch():
    """Provide instructions for installing CUDA-enabled PyTorch."""
    print("\n=== Installing CUDA-enabled PyTorch ===")
    print()
    
    if not torch.cuda.is_available():
        print("To install PyTorch with CUDA support:")
        print()
        
        # Check CUDA version if nvidia-smi is available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                # Try to extract CUDA version from nvidia-smi output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        major_version = cuda_version.split('.')[0]
                        
                        if major_version >= '12':
                            print("For CUDA 12.x:")
                            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                        elif major_version >= '11':
                            print("For CUDA 11.x:")
                            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                        else:
                            print("For older CUDA versions, check: https://pytorch.org/get-started/locally/")
                        break
                else:
                    print("Visit https://pytorch.org/get-started/locally/ for installation instructions")
            else:
                print("Visit https://pytorch.org/get-started/locally/ for installation instructions")
                
        except:
            print("Visit https://pytorch.org/get-started/locally/ for installation instructions")
    else:
        print("✅ CUDA-enabled PyTorch already installed!")

def main():
    """Main function to run all checks."""
    cuda_available = check_gpu_setup()
    recommend_optimal_settings()
    
    if not cuda_available:
        install_cuda_pytorch()
    
    print("\n" + "="*50)
    if cuda_available:
        print("🚀 Your system is ready for GPU acceleration!")
        print("Run the initialization with larger batch sizes for faster processing.")
    else:
        print("💡 Install CUDA-enabled PyTorch for significant speedup.")
        print("Current setup will work but will be slower on CPU.")

if __name__ == "__main__":
    main()