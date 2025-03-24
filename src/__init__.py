# 로컬 GPU 구동 확인
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
<<<<<<< HEAD
print(device)  # cuda
=======
print(device)  # cuda
>>>>>>> 9993d9f06d30238b0e101c2cf831773b9e8ff6db
