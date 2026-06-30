import torch

x = torch.tensor(2.).requires_grad_()
y = torch.tensor(3.).requires_grad_()

z = x * x * y

# grad_x = torch.autograd.grad(outputs=z, inputs=x, retain_graph=True)

# print(grad_x) # (tensor(12.),)

# grad_xx = torch.autograd.grad(outputs=grad_x, inputs=x)

# print(grad_xx[0])


# the problem is autograd computed the derivative and returned the numerical value, but it did not build a graph describing how that derivative was computed.
# now theres nothing left to differentiate, so you need to explciity create graph.

grad_x = torch.autograd.grad(outputs=z, inputs=x, create_graph=True)

print(grad_x) # (tensor(12.),)

grad_xx = torch.autograd.grad(outputs=grad_x, inputs=x)

print(grad_xx[0])

################################################################
# double backward

x = torch.tensor(2.).requires_grad_()
y = torch.tensor(3.).requires_grad_()

z = x * x * y

z.backward(create_graph=True) # x.grad = 12
x.grad.backward()

print(x.grad) # we get 18 and not 6? why? this is because backward() accumulates gradients and not overwrite it, so it did a 6 + 12
# work this out manually if you need clarity.

# so you always need to manually clear gradients
x = torch.tensor(2.).requires_grad_()
y = torch.tensor(3.).requires_grad_()

z = x * x * y

z.backward(create_graph=True) # x.grad = 12
x.grad.data.zero_()
x.grad.backward()

print(x.grad)