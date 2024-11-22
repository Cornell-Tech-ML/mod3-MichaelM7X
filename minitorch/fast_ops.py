from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:  # noqa: D103
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
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

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

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
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check if output and input tensors are aligned in shape and strides
        if np.array_equal(out_shape, in_shape) and np.array_equal(
            out_strides, in_strides
        ):
            # Directly apply the mapping function if shapes and strides align
            for output_idx in prange(len(out)):
                out[output_idx] = fn(in_storage[output_idx])
        else:
            # Handle the case where shapes and strides do not align
            for flat_idx in prange(len(out)):
                # Initialize index arrays for input and output tensors
                output_coords = np.zeros(MAX_DIMS, np.int32)
                input_coords = np.zeros(MAX_DIMS, np.int32)
                
                # Convert flat index to multi-dimensional coordinates for output
                to_index(flat_idx, out_shape, output_coords)
                
                # Adjust coordinates for broadcasting between output and input tensors
                broadcast_index(output_coords, out_shape, in_shape, input_coords)
                
                # Determine flat index for input tensor using strides
                input_pos = index_to_position(input_coords, in_strides)
                
                # Determine flat index for output tensor using strides
                output_pos = index_to_position(output_coords, out_strides)
                
                # Apply the mapping function and store the result in the output tensor
                out[output_pos] = fn(in_storage[input_pos])

    # Use Numba's Just-In-Time compilation for optimized parallel execution
    return njit(_map, parallel=True)  # type: ignore



def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Check if output, input A, and input B tensors are aligned in shape and strides
        if (
            np.array_equal(out_shape, a_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, b_strides)
        ):
            # Directly apply the function element-wise when shapes and strides align
            for out_idx in prange(len(out)):
                out[out_idx] = fn(a_storage[out_idx], b_storage[out_idx])
        else:
            # Handle the case where shapes and strides do not align
            for flat_idx in prange(len(out)):
                # Initialize index arrays for output, input A, and input B tensors
                output_coords = np.zeros(MAX_DIMS, np.int32)
                a_coords = np.zeros(MAX_DIMS, np.int32)
                b_coords = np.zeros(MAX_DIMS, np.int32)
                
                # Convert flat index to multi-dimensional coordinates for output
                to_index(flat_idx, out_shape, output_coords)
                
                # Adjust coordinates for broadcasting for input tensors A and B
                broadcast_index(output_coords, out_shape, a_shape, a_coords)
                broadcast_index(output_coords, out_shape, b_shape, b_coords)
                
                # Determine flat indices for input tensors A and B using strides
                a_flat_idx = index_to_position(a_coords, a_strides)
                b_flat_idx = index_to_position(b_coords, b_strides)
                
                # Apply the function and store the result in the output tensor
                out[flat_idx] = fn(a_storage[a_flat_idx], b_storage[b_flat_idx])

    # Use Numba's Just-In-Time compilation for optimized parallel execution
    return njit(_zip, parallel=True)  # type: ignore

        


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
                # Loop over each position in the output tensor
        for out_flat_idx in prange(out.size):
            # Get the size of the dimension being reduced
            reduction_axis_size = a_shape[reduce_dim]

            # Initialize a multi-dimensional index for the output tensor
            out_coords = np.zeros(MAX_DIMS, dtype=np.int32)
            
            # Convert the flat index to a multi-dimensional index for output
            to_index(out_flat_idx, out_shape, out_coords)

            # Create a copy of the output coordinates for indexing the input tensor
            input_coords = out_coords.copy()

            # Initialize the reduction result with the first element along the reduction axis
            input_coords[reduce_dim] = 0
            initial_position = index_to_position(input_coords, a_strides)
            accumulated_value = a_storage[initial_position]

            # Iterate over the remaining elements along the reduction axis
            for reduction_idx in range(1, reduction_axis_size):
                # Update the coordinate for the current position in the reduction axis
                input_coords[reduce_dim] = reduction_idx

                # Convert the coordinates to a flat index for the input tensor
                current_position = index_to_position(input_coords, a_strides)

                # Perform the reduction operation
                accumulated_value = fn(accumulated_value, a_storage[current_position])

            # Store the final reduced value in the output tensor
            out[out_flat_idx] = accumulated_value

    # Use Numba's Just-In-Time compilation for optimized parallel execution
    return njit(_reduce, parallel=True)  # type: ignore



def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
        # Calculate batch strides for tensors A and B
    a_batch_step = a_strides[0] if a_shape[0] > 1 else 0  # noqa: F841
    b_batch_step = b_strides[0] if b_shape[0] > 1 else 0  # noqa: F841

    # Ensure the inner dimensions match for matrix multiplication
    assert a_shape[-1] == b_shape[-2]

    # Iterate over each position in the output tensor
    for flat_out_idx in prange(len(out)):
        # Initialize a multi-dimensional index for the output tensor
        output_coords: Index = np.zeros(len(out_shape), dtype=np.int32)
        
        # Convert the flat index to a multi-dimensional index
        to_index(flat_out_idx, out_shape, output_coords)
        
        # Get the size of the inner loop (shared dimension in matrix multiplication)
        shared_dim_size = a_shape[-1]
        
        # Perform the inner product for the current output position
        for shared_idx in range(shared_dim_size):
            # Initialize multi-dimensional indices for tensors A and B
            a_coords: Index = np.zeros(len(a_shape), dtype=np.int32)
            b_coords: Index = np.zeros(len(b_shape), dtype=np.int32)
            
            # Copy the output coordinates to adjust for broadcasting
            a_temp_coords = output_coords.copy()
            b_temp_coords = output_coords.copy()
            
            # Update indices to reflect the current shared dimension position
            a_temp_coords[-1] = shared_idx
            b_temp_coords[-2] = shared_idx
            
            # Broadcast the coordinates for tensors A and B
            broadcast_index(a_temp_coords, out_shape, a_shape, a_coords)
            broadcast_index(b_temp_coords, out_shape, b_shape, b_coords)
            
            # Compute the flat positions in tensors A and B
            a_flat_idx = index_to_position(a_coords, a_strides)
            b_flat_idx = index_to_position(b_coords, b_strides)
            
            # Accumulate the product into the output storage
            out[flat_out_idx] += a_storage[a_flat_idx] * b_storage[b_flat_idx]


# Use Numba's Just-In-Time compilation for optimized parallel execution
tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
