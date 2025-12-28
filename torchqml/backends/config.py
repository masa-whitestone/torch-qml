"""Global backend configuration"""

from typing import Optional
from contextlib import contextmanager
import os

class BackendConfig:
    """Singleton for backend configuration"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._backend = "nvidia"  # Default
            cls._instance._initialized = False
        return cls._instance
    
    @property
    def backend(self) -> str:
        return self._backend
    
    @backend.setter
    def backend(self, value: str):
        valid_backends = [
            "nvidia",        # Single GPU (cuStateVec)
            "nvidia-mgpu",   # Multi-GPU
            "tensornet",     # Tensor network
            "tensornet-mps", # MPS
            "qpp-cpu",       # CPU (for debugging)
        ]
        if value not in valid_backends:
            # Warn but allow other backends if new ones appear
            pass
        self._backend = value
        self._initialized = False  # Re-initialization required
    
    def initialize(self):
        """Set CUDA-Q target"""
        if not self._initialized:
            try:
                import cudaq
                # Only set target if it hasn't been set externally or if we want to enforce it
                # cudaq.set_target(self._backend) 
                # Note: set_target might fail if already initialized in some contexts, 
                # but we'll try to follow the design.
                try:
                    cudaq.set_target(self._backend)
                except Exception as e:
                    print(f"Warning: Failed to set CUDA-Q target to {self._backend}: {e}")
                
                # Gate fusion setting (for nvidia/nvidia-mgpu)
                if self._backend in ["nvidia", "nvidia-mgpu"]:
                    os.environ.setdefault("CUDAQ_MGPU_FUSE", "6")
                
                self._initialized = True
            except ImportError:
                print("Warning: CUDA-Q not found. Functionality will be limited.")


_config = BackendConfig()


def set_backend(backend: str):
    """Set global backend"""
    _config.backend = backend


def get_backend() -> str:
    """Get current backend"""
    return _config.backend


@contextmanager
def backend(name: str):
    """Context manager to temporarily change backend"""
    old_backend = _config.backend
    try:
        _config.backend = name
        yield
    finally:
        _config.backend = old_backend


def set_random_seed(seed: int):
    """Set random seed"""
    try:
        import cudaq
        cudaq.set_random_seed(seed)
    except ImportError:
        pass


def _ensure_initialized():
    """Ensure backend is initialized"""
    _config.initialize()
