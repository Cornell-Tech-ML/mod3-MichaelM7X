# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```
# Map

```
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py (163) 
------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                             | 
        out: Storage,                                                                     | 
        out_shape: Shape,                                                                 | 
        out_strides: Strides,                                                             | 
        in_storage: Storage,                                                              | 
        in_shape: Shape,                                                                  | 
        in_strides: Strides,                                                              | 
    ) -> None:                                                                            | 
        # Check if output and input tensors are aligned in shape and strides              | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(                        | 
            out_strides, in_strides                                                       | 
        ):                                                                                | 
            # Directly apply the mapping function if shapes and strides align             | 
            for output_idx in prange(len(out)):-------------------------------------------| #2
                out[output_idx] = fn(in_storage[output_idx])                              | 
        else:                                                                             | 
            # Handle the case where shapes and strides do not align                       | 
            for flat_idx in prange(len(out)):---------------------------------------------| #3
                # Initialize index arrays for input and output tensors                    | 
                output_coords = np.zeros(MAX_DIMS, np.int32)------------------------------| #0
                input_coords = np.zeros(MAX_DIMS, np.int32)-------------------------------| #1
                                                                                          | 
                # Convert flat index to multi-dimensional coordinates for output          | 
                to_index(flat_idx, out_shape, output_coords)                              | 
                                                                                          | 
                # Adjust coordinates for broadcasting between output and input tensors    | 
                broadcast_index(output_coords, out_shape, in_shape, input_coords)         | 
                                                                                          | 
                # Determine flat index for input tensor using strides                     | 
                input_pos = index_to_position(input_coords, in_strides)                   | 
                                                                                          | 
                # Determine flat index for output tensor using strides                    | 
                output_pos = index_to_position(output_coords, out_strides)                | 
                                                                                          | 
                # Apply the mapping function and store the result in the output tensor    | 
                out[output_pos] = fn(in_storage[input_pos])                               | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(182) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: output_coords = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(183) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: input_coords = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
```

# Zip

```
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(227)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py (227) 
--------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                               | 
        out: Storage,                                                                       | 
        out_shape: Shape,                                                                   | 
        out_strides: Strides,                                                               | 
        a_storage: Storage,                                                                 | 
        a_shape: Shape,                                                                     | 
        a_strides: Strides,                                                                 | 
        b_storage: Storage,                                                                 | 
        b_shape: Shape,                                                                     | 
        b_strides: Strides,                                                                 | 
    ) -> None:                                                                              | 
        # Check if output, input A, and input B tensors are aligned in shape and strides    | 
        if (                                                                                | 
            np.array_equal(out_shape, a_shape)                                              | 
            and np.array_equal(out_strides, a_strides)                                      | 
            and np.array_equal(out_shape, b_shape)                                          | 
            and np.array_equal(out_strides, b_strides)                                      | 
        ):                                                                                  | 
            # Directly apply the function element-wise when shapes and strides align        | 
            for out_idx in prange(len(out)):------------------------------------------------| #7
                out[out_idx] = fn(a_storage[out_idx], b_storage[out_idx])                   | 
        else:                                                                               | 
            # Handle the case where shapes and strides do not align                         | 
            for flat_idx in prange(len(out)):-----------------------------------------------| #8
                # Initialize index arrays for output, input A, and input B tensors          | 
                output_coords = np.zeros(MAX_DIMS, np.int32)--------------------------------| #4
                a_coords = np.zeros(MAX_DIMS, np.int32)-------------------------------------| #5
                b_coords = np.zeros(MAX_DIMS, np.int32)-------------------------------------| #6
                                                                                            | 
                # Convert flat index to multi-dimensional coordinates for output            | 
                to_index(flat_idx, out_shape, output_coords)                                | 
                                                                                            | 
                # Adjust coordinates for broadcasting for input tensors A and B             | 
                broadcast_index(output_coords, out_shape, a_shape, a_coords)                | 
                broadcast_index(output_coords, out_shape, b_shape, b_coords)                | 
                                                                                            | 
                # Determine flat indices for input tensors A and B using strides            | 
                a_flat_idx = index_to_position(a_coords, a_strides)                         | 
                b_flat_idx = index_to_position(b_coords, b_strides)                         | 
                                                                                            | 
                # Apply the function and store the result in the output tensor              | 
                out[flat_idx] = fn(a_storage[a_flat_idx], b_storage[b_flat_idx])            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(252) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: output_coords = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(253) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_coords = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(254) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_coords = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
```

# Reduce

```
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(295)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py (295) 
-------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                 | 
        out: Storage,                                                                            | 
        out_shape: Shape,                                                                        | 
        out_strides: Strides,                                                                    | 
        a_storage: Storage,                                                                      | 
        a_shape: Shape,                                                                          | 
        a_strides: Strides,                                                                      | 
        reduce_dim: int,                                                                         | 
    ) -> None:                                                                                   | 
        # Loop over each position in the output tensor                                           | 
        for out_flat_idx in prange(out.size):----------------------------------------------------| #10
            # Get the size of the dimension being reduced                                        | 
            reduction_axis_size = a_shape[reduce_dim]                                            | 
                                                                                                 | 
            # Initialize a multi-dimensional index for the output tensor                         | 
            out_coords = np.zeros(MAX_DIMS, dtype=np.int32)--------------------------------------| #9
                                                                                                 | 
            # Convert the flat index to a multi-dimensional index for output                     | 
            to_index(out_flat_idx, out_shape, out_coords)                                        | 
                                                                                                 | 
            # Create a copy of the output coordinates for indexing the input tensor              | 
            input_coords = out_coords.copy()                                                     | 
                                                                                                 | 
            # Initialize the reduction result with the first element along the reduction axis    | 
            input_coords[reduce_dim] = 0                                                         | 
            initial_position = index_to_position(input_coords, a_strides)                        | 
            accumulated_value = a_storage[initial_position]                                      | 
                                                                                                 | 
            # Iterate over the remaining elements along the reduction axis                       | 
            for reduction_idx in range(1, reduction_axis_size):                                  | 
                # Update the coordinate for the current position in the reduction axis           | 
                input_coords[reduce_dim] = reduction_idx                                         | 
                                                                                                 | 
                # Convert the coordinates to a flat index for the input tensor                   | 
                current_position = index_to_position(input_coords, a_strides)                    | 
                                                                                                 | 
                # Perform the reduction operation                                                | 
                accumulated_value = fn(accumulated_value, a_storage[current_position])           | 
                                                                                                 | 
            # Store the final reduced value in the output tensor                                 | 
            out[out_flat_idx] = accumulated_value                                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(310) is hoisted out of the parallel loop labelled #10 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_coords = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(341)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py (341) 
----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                            | 
    out: Storage,                                                                       | 
    out_shape: Shape,                                                                   | 
    out_strides: Strides,                                                               | 
    a_storage: Storage,                                                                 | 
    a_shape: Shape,                                                                     | 
    a_strides: Strides,                                                                 | 
    b_storage: Storage,                                                                 | 
    b_shape: Shape,                                                                     | 
    b_strides: Strides,                                                                 | 
) -> None:                                                                              | 
    """NUMBA tensor matrix multiply function.                                           | 
                                                                                        | 
    Should work for any tensor shapes that broadcast as long as                         | 
                                                                                        | 
    ```                                                                                 | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
    ```                                                                                 | 
                                                                                        | 
    Optimizations:                                                                      | 
                                                                                        | 
    * Outer loop in parallel                                                            | 
    * No index buffers or function calls                                                | 
    * Inner loop should have no global writes, 1 multiply.                              | 
                                                                                        | 
                                                                                        | 
    Args:                                                                               | 
    ----                                                                                | 
        out (Storage): storage for `out` tensor                                         | 
        out_shape (Shape): shape for `out` tensor                                       | 
        out_strides (Strides): strides for `out` tensor                                 | 
        a_storage (Storage): storage for `a` tensor                                     | 
        a_shape (Shape): shape for `a` tensor                                           | 
        a_strides (Strides): strides for `a` tensor                                     | 
        b_storage (Storage): storage for `b` tensor                                     | 
        b_shape (Shape): shape for `b` tensor                                           | 
        b_strides (Strides): strides for `b` tensor                                     | 
                                                                                        | 
    Returns:                                                                            | 
    -------                                                                             | 
        None : Fills in `out`                                                           | 
                                                                                        | 
    """                                                                                 | 
    # Calculate batch strides for tensors A and B                                       | 
    a_batch_step = a_strides[0] if a_shape[0] > 1 else 0  # noqa: F841                  | 
    b_batch_step = b_strides[0] if b_shape[0] > 1 else 0  # noqa: F841                  | 
                                                                                        | 
    # Ensure the inner dimensions match for matrix multiplication                       | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
                                                                                        | 
    # Iterate over each position in the output tensor                                   | 
    for flat_out_idx in prange(len(out)):-----------------------------------------------| #14
        # Initialize a multi-dimensional index for the output tensor                    | 
        output_coords: Index = np.zeros(len(out_shape), dtype=np.int32)-----------------| #11
                                                                                        | 
        # Convert the flat index to a multi-dimensional index                           | 
        to_index(flat_out_idx, out_shape, output_coords)                                | 
                                                                                        | 
        # Get the size of the inner loop (shared dimension in matrix multiplication)    | 
        shared_dim_size = a_shape[-1]                                                   | 
                                                                                        | 
        # Perform the inner product for the current output position                     | 
        for shared_idx in range(shared_dim_size):                                       | 
            # Initialize multi-dimensional indices for tensors A and B                  | 
            a_coords: Index = np.zeros(len(a_shape), dtype=np.int32)--------------------| #12
            b_coords: Index = np.zeros(len(b_shape), dtype=np.int32)--------------------| #13
                                                                                        | 
            # Copy the output coordinates to adjust for broadcasting                    | 
            a_temp_coords = output_coords.copy()                                        | 
            b_temp_coords = output_coords.copy()                                        | 
                                                                                        | 
            # Update indices to reflect the current shared dimension position           | 
            a_temp_coords[-1] = shared_idx                                              | 
            b_temp_coords[-2] = shared_idx                                              | 
                                                                                        | 
            # Broadcast the coordinates for tensors A and B                             | 
            broadcast_index(a_temp_coords, out_shape, a_shape, a_coords)                | 
            broadcast_index(b_temp_coords, out_shape, b_shape, b_coords)                | 
                                                                                        | 
            # Compute the flat positions in tensors A and B                             | 
            a_flat_idx = index_to_position(a_coords, a_strides)                         | 
            b_flat_idx = index_to_position(b_coords, b_strides)                         | 
                                                                                        | 
            # Accumulate the product into the output storage                            | 
            out[flat_out_idx] += a_storage[a_flat_idx] * b_storage[b_flat_idx]          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #14, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--14 is a parallel loop
   +--11 --> rewritten as a serial loop
   +--12 --> rewritten as a serial loop
   +--13 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--11 (parallel)
   +--12 (parallel)
   +--13 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--11 (serial)
   +--12 (serial)
   +--13 (serial)


 
Parallel region 0 (loop #14) had 0 loop(s) fused and 3 loop(s) serialized as 
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(394) is hoisted out of the parallel loop labelled #14 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: output_coords: Index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(405) is hoisted out of the parallel loop labelled #14 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_coords: Index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/mzc-7x/Desktop/CT_Fall24/CS 5781/mod3-MichaelM7X/minitorch/fast_ops.py 
(406) is hoisted out of the parallel loop labelled #14 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_coords: Index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```


* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py