try:
    import torch
    print(f"torch: {torch.__version__}")
    
    import torchqml as tq
    print("Successfully imported torchqml")
    print(f"Version: {tq.__version__}")
    print(f"Backend: {tq.get_backend()}")
    
    # Check components
    q = tq.qvector(1)
    print("Created qvector")
    
    op = tq.X(0)
    print(f"Created operator: {op}")
    
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
