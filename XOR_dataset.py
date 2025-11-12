import numpy as np
import torch
#we import numpy to create the dataset
#we import torch[tensor] to create the tensor[multidimensional array]

X=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32)
Y=torch.tensor([[0],[1],[1],[0]],dtype=torch.float32)

print("Input Features(X):")
print(X)
print("\nLabels (Y):")
print(Y)
print("\nDataset shape:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")