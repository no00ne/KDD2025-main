
import numpy as np, inspect, os
import torch, numpy as np
print("torch:", torch.__version__, "| CUDA avail:", torch.cuda.is_available())
x = torch.randn(2)
print("torch OK:", x)
print("NumPy OK:", np.arange(3))
