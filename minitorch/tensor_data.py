from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # fast indexing: 利用strides, 把元组形式的 index 转化为一维数组的下标
    # 例如步长为(5, 1), 我们的storage有15个元素, 
    # 那么传入index = (1, 0)时, 下标就应该是 5*1+1*0 (直观理解为dim1的5步走了1次)
    # 传入index = (1, 2)时, 下标就应该是 5*1+1*2 (直观理解为dim1的5步走了1次, dim2的1步走了2次)

    # position = index dot strides
    # position: int = 0
    # for i in range(len(index)):
    #     position += index[i] * strides[i]
    # return position

    position = 0
    for ind, strides in zip(index, strides):
        position += ind * strides
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # 用于tensor_data的indices方法, 从ordinal \in {0, ..., size - 1}
    # 到index, 比如输出 (0, 0), (0, 1), ..., (2, 4)的某一个

    # 0->(0,0) 5->(1,0) 10->(2,0) 14->(2,4), 当 shape=(3,5)
    # 0->(0,0,0) 3->(0,0,3) 4->(0,1,0) 12->(1,0,0) 当shape=(2,3,4)
    # out_index[-1] = ordinal % shape[-1]  
    # out_index[-2] = ordinal // shape[-1] % shape[-2]
    # out_index[-3] = ordinal // shape[-1] // shape[-2] % shape[-3]

    # cur_pos = ordinal + 0 # 为了Module3并行, 不能修改循环变量的trick
    # for i in range(len(shape) - 1, -1, -1): # 倒序
    #     out_index[i] = cur_pos % shape[i]
    #     cur_pos = int(cur_pos / shape[i])
    cur_pos = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_pos % sh)
        cur_pos = cur_pos // sh


    # 在学习了stride之后可以有直觉的理解:
    # ordinal是向strides的各个方向走了index步得到的,
    # 那么对于"最精细"(strides最右侧)的小步子走了多少? (对最右侧shape的直接取模)
    # 在不考虑小步子(直接整体整除strides最右侧), 稍微粗一点的步子走了多少? (对第二个shape的取模)
    # storage[s1 * index1 + s2 * index2 + s3 * index3 ... ]

def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # eg. big_shape = (2, 3, 4), shape = (3, 4), 
    # big_index = (1, 2, 3) -> out_index = (2, 3), 
    # 即去掉最外层值为1的维度, 保留内层
    # eg2. big_shape = (2, 3), shape = (2, 1)
    # 则将维度为1处全部映射到0
    # eg3. big_shape = (2, 4), shape = (2, ) # WRONG!
    # 报错, 无法正常映射, 本函数不作检查, 默认传入的都合法

    # for i in range(len(shape) - 1, -1, -1):
    #     # big_index和index都从最右往左对齐
    #     dimension = i + len(big_shape) - len(shape)
    #     if big_shape[dimension] == 1 or shape[i] == 1:
    #         out_index[i] = 0
    #     else:
    #         out_index[i] = big_index[dimension]
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # shape: UserShape = ()
    # if (len(shape1) < len(shape2)):
    #     shape1 = (1,) * (len(shape2) - len(shape1)) + tuple(shape1)
    # if (len(shape2) < len(shape1)):
    #     shape2 = (1,) * (len(shape1) - len(shape2)) + tuple(shape2)
    # for i in range(len(shape1)):
    #     if (min(shape1[i], shape2[i]) == 1 or shape1[i] == shape2[i]):
    #         shape += (max(shape1[i], shape2[i]),)
    #     else:
    #         raise IndexingError()
    # return shape

    a, b = shape1, shape2
    m = max(len(a), len(b))
    # print("m",m)
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError("Broadcast failure")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError("Broadcast failure")
    return tuple(reversed(c_rev))


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.
        strides降序

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        return TensorData(self._storage, tuple([self.shape[i] for i in order]), tuple([self.strides[i] for i in order]))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
