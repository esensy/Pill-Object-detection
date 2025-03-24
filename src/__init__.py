# 로컬 GPU 구동 확인
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
<<<<<<< HEAD
print(device)  # cuda
=======
print(device)  # cuda
>>>>>>> f1423eadd50d4996b1fa3e4ec388a8288987b86f
