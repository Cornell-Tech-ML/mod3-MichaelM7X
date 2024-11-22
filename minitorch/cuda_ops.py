# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003, D103
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003, D103
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # noqa: D102
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(  # noqa: D102
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:  # noqa: D102
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return

        # Convert the linear thread index `i` into a multi-dimensional index for the output tensor.
        to_index(i, out_shape, out_index)

        # Broadcasting ensures compatibility between tensors of different shapes.
        broadcast_index(out_index, out_shape, in_shape, in_index)

        out_pos = index_to_position(out_index, out_strides)

        out[out_pos] = fn(in_storage[index_to_position(in_index, in_strides)])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Calculate the global thread index based on block and thread positions.
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Exit if the thread index exceeds the total output size to prevent out-of-bounds access.
        if i >= out_size:
            return

        # Convert the linear index `i` to a multi-dimensional index for the output tensor.
        to_index(i, out_shape, out_index)

        # Compute the flat storage position for the output tensor using its strides.
        o = index_to_position(out_index, out_strides)

        # Map the output tensor's index to the corresponding input tensor `a` index, considering broadcasting.
        broadcast_index(out_index, out_shape, a_shape, a_index)

        # Map the output tensor's index to the corresponding input tensor `b` index, considering broadcasting.
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Compute the flat storage position for input tensor 'a' 'b' using its strides.
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)

        # Apply the function `fn` to elements from both input tensors and store the result in the output tensor.
        out[o] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """  # noqa: D301, D404
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Skip processing if thread index exceeds size
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0  # Ensure out-of-bounds threads store 0
    cuda.syncthreads()

    step = 1
    while step < BLOCK_DIM:
        # Only threads at even multiples of 'step' perform summation
        neighbor = pos + step
        if neighbor < BLOCK_DIM and (i + step) < size:
            cache[pos] += cache[neighbor]
        step *= 2
        cuda.syncthreads()

    # Write the final block sum to the output
    if pos == 0:
        block_index = cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x
        out[block_index] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:  # noqa: D103
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if out_pos >= out_size:
            return
        reduce_dim_size = a_shape[reduce_dim]
        to_index(out_pos, out_shape, out_index)
        to_index(out_pos, out_shape, a_index)
        a_index[reduce_dim] = pos
        if pos < reduce_dim_size:
            cache[pos] = a_storage[index_to_position(a_index, a_strides)]
        else:
            cache[pos] = 0.0
        cuda.syncthreads()
        # naive
        if pos == 0:
            for i in range(1, reduce_dim_size):
                a_index[reduce_dim] = i
                cache[0] = fn(
                    a_storage[index_to_position(a_index, a_strides)], cache[0]
                )
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """  # noqa: D404
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # Compute global thread indices
    row_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Allocate shared memory for tiles of matrices
    tile_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    tile_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Skip threads that are out of bounds
    if row_idx >= size or col_idx >= size:
        return

    # Load data from global memory to shared memory tiles
    tile_a[cuda.threadIdx.x, cuda.threadIdx.y] = a[row_idx * size + col_idx]
    tile_b[cuda.threadIdx.x, cuda.threadIdx.y] = b[row_idx * size + col_idx]

    # Synchronize threads to ensure shared memory is fully populated
    cuda.syncthreads()

    # Perform the dot product for the corresponding row and column
    result = 0.0
    for k in range(size):
        result += tile_a[cuda.threadIdx.x, k] * tile_b[k, cuda.threadIdx.y]

    # Store the computed value in the output matrix
    out[row_idx * size + col_idx] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:  # noqa: D103
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    M, N, K = out_shape[1], out_shape[2], a_shape[-1]
    result = 0.0  # Temporary accumulator for the thread's computation

    # Iterate through the blocks of the shared dimension
    for offset in range(0, K, BLOCK_DIM):
        # Load a block of matrix A into shared memory
        if i < M and (offset + pj) < K:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + (offset + pj) * a_strides[2]
            ]
        else:
            a_shared[pi, pj] = 0.0  # Handle out-of-bounds threads

        # Load a block of matrix B into shared memory
        if (offset + pi) < K and j < N:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + (offset + pi) * b_strides[1] + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0.0  # Handle out-of-bounds threads

        # Synchronize to ensure shared memory is fully loaded
        cuda.syncthreads()

        # Compute the partial dot product for this block
        for inner in range(BLOCK_DIM):
            if (offset + inner) < K:
                result += a_shared[pi, inner] * b_shared[inner, pj]

        # Synchronize before loading the next block
        cuda.syncthreads()

    # Write the final result to the global output array
    if i < M and j < N:
        global_index = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[global_index] = result


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
