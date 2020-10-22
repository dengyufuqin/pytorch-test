import torch
import torch.nn as nn
import torch.nn.functional as F



class Swish(nn.Module):
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.save_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemorySwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class hSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x+3, inplace=True) / 6
        return out

