import torch

device = torch.device("cuda")

# x = torch.ones(2, 2, device=device, requires_grad=True)
# y = x + 2
# print(y)
# z = y * y * 3
# print(z)
# print(z.requires_grad)

#-----change requires_grad in the middle

inp = torch.rand(3, 3, requires_grad=True)
print(inp.device)
# with torch.no_grad():
a = inp.to(device)
b = torch.rand(3, 3, requires_grad=True, device=device)
c = a + b
out = (c * b).sum()
out.backward()
# print(inp.grad)
# print(a.grad)
print(b.grad)
print(inp.grad)
# print(a.grad)

#-----vector-jacobian product

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y *= 2
print(y)
y.backward(torch.tensor([0.1, 0.2, 0.3]))
print(x.grad)
