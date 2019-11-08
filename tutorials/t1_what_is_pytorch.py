import numpy as np
import torch

x = torch.empty(5, 3)
print(x)

x1 = torch.rand(5, 3)
print(x1)

random_np = np.random.rand(100, 100)
random_tensor = torch.tensor(random_np)
print(random_tensor)
one_tensor = torch.ones_like(random_tensor, dtype=torch.float32)

#====Operations

#slice
y = torch.rand(5, 3)
z = x1 + y
print(z.shape)

print(z[:2, :2])

#reshape

x2 = torch.randn(4, 4)
y2 = torch.reshape(x2, (16, ))
y2_same = x2.view(16)
z2 = torch.reshape(x2, (-1, 8))
z2_same = x2.view(-1, 8)
x2[0, 0] = 9
print(y2)
print(y2_same)
print(z2)
print(z2_same)

c = torch.tensor([[[[[423]]]]])
print(c.size())
print(c.item())

#------numpy bridge

a = np.ones(5)
b1 = torch.from_numpy(a)
b2 = torch.tensor(a)
np.add(a, 1, out=a)
print(a)
print(b1)
print(b2)

# ----- cuda tensors
device = torch.device("cuda")

y = torch.rand(5, 3, device=device)
print(y)
x = torch.rand(5, 3)
print(x.device)
x = x.to(device)
print(x)
z = x + y
print(z)
print(z.to("cpu"))
