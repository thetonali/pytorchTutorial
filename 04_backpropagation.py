import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# This is the parameter we want to optimize -> requires_grad=True
# 这是我们要优化的参数->requires_grad=True
w = torch.tensor(1.0, requires_grad=True)

# forward pass to compute loss
# 前向传递计算损失
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

# backward pass to compute gradient dLoss/dw
# 向后传递计算梯度 dLoss/dw
# !!!损失对权重的梯度
loss.backward()
print(w.grad)

# update weights
# next forward and backward pass...
# 更新权重
# 接下来的向前和向后传递...

# continue optimizing:
# update weights, this operation should not be part of the computational graph
# 继续优化：
# 更新权重，这个操作不应该是计算图的一部分
with torch.no_grad():
    w -= 0.01 * w.grad
# don't forget to zero the gradients
# 不要忘记将梯度归零
w.grad.zero_()

# next forward and backward pass...
# 接下来的向前和向后传递...
