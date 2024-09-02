
import torch
# The autograd package provides automatic differentiation 
# for all operations on Tensors
# autograd包提供自动微分
# 对于张量上的所有操作

# requires_grad = True -> tracks all operations on the tensor. 
# require_grad = True -> 跟踪张量上的所有操作。
x = torch.randn(3, requires_grad=True)
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute.
# grad_fn: references a Function that has created the Tensor
# y 是作为操作的结果而创建的，因此它具有 grad_fn 属性。
# grad_fn：引用创建张量的函数
# grad_fn 是 PyTorch 中张量（Tensor）对象的一个属性，
# 它表示创建该张量的操作函数（Function）。
# 具体来说，它用于追踪在计算图中生成这个张量的那一步操作。
print(x) # created by the user -> grad_fn is None
print(y)
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
print(z)
z = z.mean()
print(z)

# Let's compute the gradients with backpropagation
# When we finish our computation we can call .backward() and have all the gradients computed automatically.
# The gradient for this tensor will be accumulated into .grad attribute.
# It is the partial derivate of the function w.r.t. the tensor
# 让我们用反向传播来计算梯度
# 当我们完成计算时，我们可以调用 .backward() 并自动计算所有梯度。
# 该张量的梯度将累积到 .grad 属性中。
# 它是函数 w.r.t. 张量的偏导数。

z.backward()
print(x.grad) # dz/dx

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product
# It computes partial derivates while applying the chain rule
# 一般来说，torch.autograd是一个计算向量雅可比积的引擎
# 应用链式法则计算偏导数
# -------------
# Model with non-scalar output:
# If a Tensor is non-scalar (more than 1 elements), we need to specify arguments for backward() 
# specify a gradient argument that is a tensor of matching shape.
# needed for vector-Jacobian product
# 具有非标量输出的模型：
# 如果张量是非标量（超过 1 个元素），我们需要为backward()指定参数
# 指定一个梯度参数，它是匹配形状的张量。
# 需要矢量雅可比积
x = torch.randn(3, requires_grad=True)

y = x * 2
for _ in range(10):
    y = y * 2

print(y)
print(y.shape)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

# -------------
# Stop a tensor from tracking history:
# For example during our training loop when we want to update our weights
# then this update operation should not be part of the gradient computation
# - x.requires_grad_(False)
# - x.detach()
# - wrap in 'with torch.no_grad():'
# 停止张量跟踪历史记录：
# 例如，在我们的训练循环中，当我们想要更新权重时
# 那么这个更新操作不应该是梯度计算的一部分
# - x.requires_grad_(False)
# - x.detach()
# - 包含在 'with torch.no_grad():' 中

# .requires_grad_(...) changes an existing flag in-place.
# .requires_grad_(...) 就地更改现有标志。
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# .detach(): get a new Tensor with the same content but no gradient computation:
# .detach()：得到一个内容相同但不进行梯度计算的新Tensor：
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)

# wrap in 'with torch.no_grad():'
# 包装在 'with torch.no_grad（）：' 中
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# -------------
# backward() accumulates the gradient for this tensor into .grad attribute.
#-------------
# backward() 将此张量的梯度累积到 .grad 属性中。
# !!! We need to be careful during optimization !!!
# Use .zero_() to empty the gradients before a new optimization step!
# !!!优化的时候一定要小心！！！
# 在新的优化步骤之前使用 .zero_() 清空梯度！
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    # 只是一个虚拟示例
    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)

    # optimize model, i.e. adjust weights...
    # 优化模型，即调整权重...
    with torch.no_grad():
        weights -= 0.1 * weights.grad

    # this is important! It affects the final weights & output
    # 这很重要！它影响最终的权重和输出
    weights.grad.zero_()

print(weights)
print(model_output)

# Optimizer has zero_grad() method
# optimizer = torch.optim.SGD([weights], lr=0.1)
# During training:
# optimizer.step()
# optimizer.zero_grad()
# 优化器有 zero_grad（） 方法
# 优化器 = torch.optim.SGD（[weights]， lr=0.1）
# 训练期间：
# optimizer.step（）
# optimizer.zero_grad（）