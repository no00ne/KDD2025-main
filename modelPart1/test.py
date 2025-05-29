import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("CUDA_AVAIL =", torch.cuda.is_available())
if torch.cuda.is_available():
    a = torch.randn(2).cuda()
    print("Tensor on GPU OK:", a)
