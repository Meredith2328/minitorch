from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Numba prange for parallel execution
    for i in prange(out_size):
        # 计算输出索引
        out_idx = np.zeros(3, dtype=np.int32)
        to_index(i, out_shape, out_idx)
        b = out_idx[0]
        oc = out_idx[1]
        w = out_idx[2]
        
        # 累加器
        acc = 0.0
        
        # 遍历输入通道和卷积核
        for ic in range(in_channels):
            for k in range(kw):
                if reverse:
                    # 右对齐 (用于反向传播)
                    input_w = w - k
                else:
                    # 左对齐 (用于前向传播)
                    input_w = w + k - (kw - 1)
                
                # 边界检查
                if 0 <= input_w < width:
                    # 计算输入位置
                    input_pos = (
                        b * input_strides[0] +
                        ic * input_strides[1] +
                        input_w * input_strides[2]
                    )
                    
                    # 计算权重位置
                    weight_pos = (
                        oc * weight_strides[0] +
                        ic * weight_strides[1] +
                        k * weight_strides[2]
                    )
                    
                    acc += input[input_pos] * weight[weight_pos]
        
        # 写入输出
        out_pos = (
            b * out_strides[0] +
            oc * out_strides[1] +
            w * out_strides[2]
        )
        out[out_pos] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # 预取步长以加速
    s1_0, s1_1, s1_2, s1_3 = input_strides[0], input_strides[1], input_strides[2], input_strides[3]
    s2_0, s2_1, s2_2, s2_3 = weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3]
    s_out_0, s_out_1, s_out_2, s_out_3 = out_strides[0], out_strides[1], out_strides[2], out_strides[3]
    
    # Numba prange for parallel execution
    for i in prange(out_size):
        # 计算输出索引
        out_idx = np.zeros(4, dtype=np.int32)
        to_index(i, out_shape, out_idx)
        b = out_idx[0]
        oc = out_idx[1]
        h = out_idx[2]
        w = out_idx[3]
        
        # 累加器
        acc = 0.0
        
        # 遍历输入通道和卷积核
        for ic in range(in_channels):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    if reverse:
                        # 右下对齐 (用于反向传播)
                        input_h = h - kh_i
                        input_w = w - kw_i
                    else:
                        # 左上对齐 (用于前向传播)
                        input_h = h + kh_i - (kh - 1)
                        input_w = w + kw_i - (kw - 1)
                    
                    # 边界检查
                    if 0 <= input_h < height and 0 <= input_w < width:
                        # 计算输入位置
                        input_pos = (
                            b * s1_0 +
                            ic * s1_1 +
                            input_h * s1_2 +
                            input_w * s1_3
                        )
                        
                        # 计算权重位置
                        weight_pos = (
                            oc * s2_0 +
                            ic * s2_1 +
                            kh_i * s2_2 +
                            kw_i * s2_3
                        )
                        
                        acc += input[input_pos] * weight[weight_pos]
        
        # 写入输出
        out_pos = (
            b * s_out_0 +
            oc * s_out_1 +
            h * s_out_2 +
            w * s_out_3
        )
        out[out_pos] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
