from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw
    
    # 使用 view 和 permute 重塑
    # batch x channel x new_height x kh x new_width x kw
    t = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    
    # 重排维度: batch, channel, new_height, new_width, kh*kw
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    t = t.view(batch, channel, new_height, new_width, kh * kw)
    
    return t, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    # 在最后一个维度上平均
    return tiled.mean(dim=4)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        dim_val = int(dim.item())
        
        # 创建 argmax mask
        max_vals = max_reduce(input, dim_val)
        mask = input == max_vals
        
        # 处理多个最大值的情况
        mask_sum = mask.sum(dim=dim_val)
        mask = mask / mask_sum
        
        return grad_output * mask, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # 减去最大值以提高数值稳定性
    input_max = max(input, dim)
    exp_input = (input - input_max).exp()
    return exp_input / exp_input.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    input_max = max(input, dim)
    # log(sum(exp(x - max))) + max
    lse = input_max + (input - input_max).exp().sum(dim=dim).log()
    return input - lse


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    # 在最后一个维度上取最大值
    return max(tiled, 4)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore or rate == 0.0:
        return input
    if rate == 1.0:
        return input.zeros(input.shape)
    
    # 生成 dropout mask
    mask = rand(input.shape) > rate
    return input * mask
    
