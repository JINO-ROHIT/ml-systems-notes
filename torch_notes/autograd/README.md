### understanding autograd in pytorch

this section is a pre-requisite before you can move onto the aotautograd section.

torch autograd is the engine that powers neural network training by doing automatic differentiation.

say we have to train a 10 layer model, this happens in two stages -
1. forward pass - here the input is flowing through all the layers and at the final layer, it makes a guess for the predicted output.
2. backward pass - here the predicted output is compared against the truth labels, error and gradients are calculated and then propogated in the reverse order. 

pytorch handles all of this for you using `dynamic graph` also known as the eager mode. this means it runs the code line by line, and the graph can change
dynamically depends on the input. another way to think about it is both the graph construction and the computation is happening side by side.

`static graph` on the other hand is built before computation can happen. it just captures a fixed graph that never changes.

both has its own pros and cons but lets stop here and move onto the actual concepts.

cool, pytorch's computation graph has only two types of elements, its the data (tensors) and the operations.

operations can be addition, subtraction, multiplication, division, square root, exponentiation, exponentiation, trigonometric functions, and other differentiable operations etc.

data is the actual input values. it is divided into leaf nodes and non-leaf nodes. leaf nodes are user created nodes that do not depend on other nodes. 
the difference between them is that after backpropagation, the gradients of non-leaf nodes are released, and only the gradients of leaf nodes are retained, thus saving memory. if you want to retain the gradients of non-leaf nodes, you can use `retain_grad()`.


there are two ways you can calculate gradients, you can either use `backward()` or `torch.autograd.grad()`. they slightly differ in what happens with the gradients.
1. `backward()` stores the .grad field in the leaf node.
2. `autograd().grad()` directly returns the gradient but does not store it.