import torch 

x = torch.rand(3)
print(x)

y = torch.cuda.is_available()
print(y)