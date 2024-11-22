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

# Train a tensor Model

# CPU (simple)
```
cpu --HIDDEN 10 --DATASET simple --RATE 0.05
```

```
Epoch  0  loss  8.537481551217823 correct 20
Epoch  10  loss  6.5920359636868175 correct 30
Epoch  20  loss  6.82832185681613 correct 30
Epoch  30  loss  5.363042578580208 correct 39
Epoch  40  loss  4.192663528994036 correct 44
Epoch  50  loss  3.6499824320812606 correct 48
Epoch  60  loss  2.247505349138385 correct 49
Epoch  70  loss  0.7836732604357564 correct 49
Epoch  80  loss  1.4495136290799966 correct 49
Epoch  90  loss  1.032054907227122 correct 49
Epoch  100  loss  1.1533655097153654 correct 50
Epoch  110  loss  0.5377713017771707 correct 50
Epoch  120  loss  1.3697630481266736 correct 50
Epoch  130  loss  0.27820704453639483 correct 50
Epoch  140  loss  0.40542695381661603 correct 50
Epoch  150  loss  1.1141855389247055 correct 49
Epoch  160  loss  0.9213852063803193 correct 50
Epoch  170  loss  0.7639247036229935 correct 50
Epoch  180  loss  0.20882111828184124 correct 50
Epoch  190  loss  0.9473706069894743 correct 50
Epoch  200  loss  0.3150232632090167 correct 50
Epoch  210  loss  0.5686536286154946 correct 50
Epoch  220  loss  0.13869760727509095 correct 50
Epoch  230  loss  0.0462924993891584 correct 50
Epoch  240  loss  0.08763910771340043 correct 50
Epoch  250  loss  0.25115385740314383 correct 50
Epoch  260  loss  0.3884447997138279 correct 50
Epoch  270  loss  0.7475541688363863 correct 50
Epoch  280  loss  0.04211548204978375 correct 50
Epoch  290  loss  0.00786959243674853 correct 50
Epoch  300  loss  0.026316523414555124 correct 50
Epoch  310  loss  0.7102621767054599 correct 50
Epoch  320  loss  0.20466280693074546 correct 50
Epoch  330  loss  0.06498177152846002 correct 50
Epoch  340  loss  0.240315408960746 correct 50
Epoch  350  loss  0.017840483387547835 correct 50
Epoch  360  loss  0.04556372107925764 correct 50
Epoch  370  loss  0.1531256746164094 correct 50
Epoch  380  loss  0.5213259991359637 correct 50
Epoch  390  loss  0.40437997072895077 correct 50
Epoch  400  loss  0.1241857384013598 correct 50
Epoch  410  loss  0.012585644647884586 correct 50
Epoch  420  loss  0.7135065088371455 correct 50
Epoch  430  loss  0.023378762624154687 correct 50
Epoch  440  loss  0.5370793552228482 correct 50
Epoch  450  loss  0.04597886435137009 correct 50
Epoch  460  loss  0.009769440930467003 correct 50
Epoch  470  loss  0.11235645433099918 correct 50
Epoch  480  loss  0.03268731108345065 correct 50
Epoch  490  loss  0.039037080695301866 correct 50
Mean time taken 10 epochs: 0.7389841032028198
```

# CPU (split)
```
cpu --HIDDEN 10 --DATASET split --RATE 0.05
```

```
poch  0  loss  6.840855287358149 correct 19
Epoch  10  loss  7.299017817687951 correct 31
Epoch  20  loss  7.553013690563964 correct 31
Epoch  30  loss  7.143129170118831 correct 31
Epoch  40  loss  7.177124503430172 correct 31
Epoch  50  loss  5.797380248448376 correct 31
Epoch  60  loss  6.165813273046274 correct 31
Epoch  70  loss  7.521463388398309 correct 31
Epoch  80  loss  7.096440491122324 correct 31
Epoch  90  loss  6.5760445561098475 correct 31
Epoch  100  loss  6.2161178036424936 correct 31
Epoch  110  loss  6.813353376327122 correct 31
Epoch  120  loss  5.591875769178205 correct 31
Epoch  130  loss  7.042664558102522 correct 31
Epoch  140  loss  5.439913080953999 correct 31
Epoch  150  loss  5.720892433050513 correct 31
Epoch  160  loss  6.2739038557868465 correct 31
Epoch  170  loss  7.020283054320542 correct 32
Epoch  180  loss  5.674167230371817 correct 33
Epoch  190  loss  6.47262308951744 correct 34
Epoch  200  loss  6.218057715058046 correct 35
Epoch  210  loss  6.5744972672146575 correct 37
Epoch  220  loss  5.354808438054503 correct 38
Epoch  230  loss  4.953335196641689 correct 39
Epoch  240  loss  5.089563283340948 correct 39
Epoch  250  loss  4.127677668392 correct 39
Epoch  260  loss  4.521904139128981 correct 42
Epoch  270  loss  3.896711166149383 correct 42
Epoch  280  loss  3.94474323651939 correct 44
Epoch  290  loss  2.766389536564566 correct 46
Epoch  300  loss  4.978339678620543 correct 49
Epoch  310  loss  4.045695251648958 correct 49
Epoch  320  loss  1.8574579711574444 correct 48
Epoch  330  loss  2.7221786356926083 correct 49
Epoch  340  loss  3.026079012558058 correct 49
Epoch  350  loss  1.703940925887803 correct 47
Epoch  360  loss  3.034478823682339 correct 50
Epoch  370  loss  2.528497179069366 correct 49
Epoch  380  loss  2.734711532105517 correct 49
Epoch  390  loss  1.1918430077880318 correct 49
Epoch  400  loss  1.7764240502886823 correct 49
Epoch  410  loss  0.6035500167890773 correct 50
Epoch  420  loss  1.1367010068209122 correct 49
Epoch  430  loss  1.4929458205278432 correct 48
Epoch  440  loss  1.8355692485475217 correct 48
Epoch  450  loss  2.2063529328506006 correct 50
Epoch  460  loss  1.112311528783176 correct 47
Epoch  470  loss  0.3439034706502018 correct 48
Epoch  480  loss  1.3390799768368988 correct 49
Epoch  490  loss  1.0078187314229266 correct 50
Mean time taken 10 epochs: 0.7114020729064942
```

# CPU (xor)
```
cpu --HIDDEN 10 --DATASET xor --RATE 0.05
```

```
Epoch  0  loss  6.958866756302016 correct 27
Epoch  10  loss  6.948631389462087 correct 27
Epoch  20  loss  7.052400994748039 correct 27
Epoch  30  loss  6.892922365593571 correct 27
Epoch  40  loss  6.587012096403664 correct 27
Epoch  50  loss  6.8491285973599005 correct 27
Epoch  60  loss  6.617163306203103 correct 27
Epoch  70  loss  6.706241615429118 correct 27
Epoch  80  loss  6.648024251255037 correct 27
Epoch  90  loss  6.467359849729381 correct 33
Epoch  100  loss  6.38399938625423 correct 36
Epoch  110  loss  6.277517704111605 correct 40
Epoch  120  loss  6.223660645748489 correct 41
Epoch  130  loss  5.8686638211268995 correct 41
Epoch  140  loss  5.748987833405481 correct 40
Epoch  150  loss  5.263271104830226 correct 43
Epoch  160  loss  4.668564554360304 correct 45
Epoch  170  loss  4.832495423825911 correct 46
Epoch  180  loss  5.065155101621791 correct 46
Epoch  190  loss  5.016851899930266 correct 45
Epoch  200  loss  2.6334562569512765 correct 45
Epoch  210  loss  4.154157181410358 correct 45
Epoch  220  loss  1.3959162105473188 correct 45
Epoch  230  loss  0.673512936251872 correct 46
Epoch  240  loss  2.8673476100917963 correct 45
Epoch  250  loss  1.5169251778524862 correct 45
Epoch  260  loss  0.9383404687208275 correct 46
Epoch  270  loss  5.036692054811256 correct 46
Epoch  280  loss  3.255105512499377 correct 46
Epoch  290  loss  2.4943199807968433 correct 46
Epoch  300  loss  3.5980697521736844 correct 46
Epoch  310  loss  2.191625529274684 correct 46
Epoch  320  loss  1.4239365991184707 correct 46
Epoch  330  loss  1.0227146871167299 correct 46
Epoch  340  loss  2.0684722773039503 correct 46
Epoch  350  loss  0.4508734712949401 correct 48
Epoch  360  loss  2.8026123194151795 correct 46
Epoch  370  loss  1.4247942696644267 correct 46
Epoch  380  loss  2.97030217599812 correct 48
Epoch  390  loss  1.6102899288724086 correct 48
Epoch  400  loss  2.379675166429937 correct 48
Epoch  410  loss  2.118410878203552 correct 47
Epoch  420  loss  1.6472598458129295 correct 49
Epoch  430  loss  2.16230500988517 correct 47
Epoch  440  loss  1.919233476457526 correct 50
Epoch  450  loss  0.6026104524175064 correct 50
Epoch  460  loss  1.0039180347521062 correct 50
Epoch  470  loss  1.5723484806702277 correct 50
Epoch  480  loss  0.2099781713566333 correct 50
Epoch  490  loss  0.6140910630096137 correct 50
Mean time taken 10 epochs: 0.721669054031372
```

# GPU (simple)
```
gpu --HIDDEN 10 --DATASET simple --RATE 0.05
```

```
0  loss  7.044935436207029 correct 23
Epoch  10  loss  6.930603179827086 correct 27
Epoch  20  loss  6.65234956126439 correct 27
Epoch  30  loss  6.638392792444535 correct 27
Epoch  40  loss  7.293009811103818 correct 27
Epoch  50  loss  6.865348718859507 correct 27
Epoch  60  loss  6.764666494409285 correct 27
Epoch  70  loss  6.432693892679716 correct 27
Epoch  80  loss  6.798020191676201 correct 27
Epoch  90  loss  6.724467684313425 correct 29
Epoch  100  loss  6.827325268076103 correct 30
Epoch  110  loss  6.050006900176301 correct 32
Epoch  120  loss  6.3132247278207885 correct 33
Epoch  130  loss  6.310386593446962 correct 34
Epoch  140  loss  5.74349813134314 correct 39
Epoch  150  loss  5.824336012568243 correct 42
Epoch  160  loss  5.29665240839792 correct 44
Epoch  170  loss  6.427457284429488 correct 44
Epoch  180  loss  5.392424745515231 correct 43
Epoch  190  loss  4.881640717664716 correct 44
Epoch  200  loss  5.930751867190987 correct 44
Epoch  210  loss  4.359493565893841 correct 44
Epoch  220  loss  2.7580321265840646 correct 44
Epoch  230  loss  3.991368646635463 correct 45
Epoch  240  loss  3.262938718941988 correct 45
Epoch  250  loss  1.4936723934767322 correct 45
Epoch  260  loss  2.2863548371283717 correct 45
Epoch  270  loss  2.1361021466366235 correct 46
Epoch  280  loss  2.133812228854766 correct 46
Epoch  290  loss  0.7465460749798715 correct 45
Epoch  300  loss  3.1181546893977585 correct 44
Epoch  310  loss  1.9874201656052577 correct 45
Epoch  320  loss  3.887253816842566 correct 45
Epoch  330  loss  4.507036972435004 correct 46
Epoch  340  loss  2.426221924899381 correct 46
Epoch  350  loss  0.9811952690939679 correct 47
Epoch  360  loss  3.79435959760894 correct 47
Epoch  370  loss  3.034072715024819 correct 48
Epoch  380  loss  2.824232166948509 correct 48
Epoch  390  loss  1.7560359861003298 correct 49
Epoch  400  loss  0.649136724185023 correct 48
Epoch  410  loss  1.0078186121826183 correct 48
Epoch  420  loss  3.611691230733178 correct 49
Epoch  430  loss  0.3597018047706425 correct 48
Epoch  440  loss  2.182815727366284 correct 49
Epoch  450  loss  1.4013886564070883 correct 48
Epoch  460  loss  1.2942158750440074 correct 50
Epoch  470  loss  2.266191099259206 correct 50
Epoch  480  loss  0.6257386803753154 correct 50
Epoch  490  loss  1.5158654715383064 correct 50
Mean time taken 10 epochs: 11.996146087646485
```

# GPU (split)
```
gpu --HIDDEN 10 --DATASET split --RATE 0.05
```

```
Epoch  0  loss  6.972197184885792 correct 19
Epoch  10  loss  6.6277853076045625 correct 31
Epoch  20  loss  5.796292502242937 correct 31
Epoch  30  loss  7.0112082661691 correct 31
Epoch  40  loss  6.411501240005519 correct 31
Epoch  50  loss  7.687988511038705 correct 31
Epoch  60  loss  7.3908755507286354 correct 31
Epoch  70  loss  5.699059952431044 correct 31
Epoch  80  loss  6.199301453351044 correct 33
Epoch  90  loss  5.710225553884311 correct 33
Epoch  100  loss  5.954663654119772 correct 34
Epoch  110  loss  5.4298305666007805 correct 35
Epoch  120  loss  6.3187592843693245 correct 39
Epoch  130  loss  5.47830667456502 correct 39
Epoch  140  loss  5.972557828661541 correct 40
Epoch  150  loss  6.5249774256314605 correct 42
Epoch  160  loss  5.728344375309774 correct 42
Epoch  170  loss  5.608688983970237 correct 43
Epoch  180  loss  4.553239794408045 correct 43
Epoch  190  loss  4.197175865016703 correct 43
Epoch  200  loss  3.813877846246914 correct 43
Epoch  210  loss  4.242268249994455 correct 43
Epoch  220  loss  4.649656828155082 correct 43
Epoch  230  loss  3.921142969482723 correct 43
Epoch  240  loss  4.5656993644006425 correct 42
Epoch  250  loss  4.637503090595362 correct 43
Epoch  260  loss  5.7287263249596405 correct 43
Epoch  270  loss  3.255050724155943 correct 44
Epoch  280  loss  1.8871476427737042 correct 43
Epoch  290  loss  3.816254958443105 correct 44
Epoch  300  loss  2.978423447468954 correct 45
Epoch  310  loss  3.423624169402843 correct 46
Epoch  320  loss  2.890617316513224 correct 47
Epoch  330  loss  2.8186608600642518 correct 48
Epoch  340  loss  3.4617635919948024 correct 46
Epoch  350  loss  2.5408154102396368 correct 47
Epoch  360  loss  2.515317184872408 correct 47
Epoch  370  loss  2.7162242493867828 correct 48
Epoch  380  loss  1.334610351396247 correct 48
Epoch  390  loss  1.7560359861003298 correct 48
Epoch  400  loss  0.649136724185023 correct 49
Epoch  410  loss  1.0078186121826183 correct 49
Epoch  420  loss  3.611691230733178 correct 50
Epoch  430  loss  0.3597018047706425 correct 50
Epoch  440  loss  2.182815727366284 correct 50
Epoch  450  loss  1.4013886564070883 correct 50
Epoch  460  loss  1.2942158750440074 correct 50
Epoch  470  loss  2.266191099259206 correct 50
Epoch  480  loss  0.6257386803753154 correct 50
Epoch  490  loss  1.5158654715383064 correct 50
Mean time taken 10 epochs: 12.000146087646485

```

# GPU (xor)
```
gpu --HIDDEN 10 --DATASET xor --RATE 0.05
```

```
Epoch 0 loss 7.021874653829472 correct 23
Epoch 10 loss 7.105239142773651 correct 29
Epoch 20 loss 6.917203491728091 correct 31
Epoch 30 loss 5.882354219193101 correct 30
Epoch 40 loss 5.742129887439104 correct 31
Epoch 50 loss 6.291843612573182 correct 30
Epoch 60 loss 6.081743726193572 correct 34
Epoch 70 loss 6.638495293751231 correct 35
Epoch 80 loss 6.672421587293112 correct 36
Epoch 90 loss 5.315609481943521 correct 37
Epoch 100 loss 4.382011345882992 correct 36
Epoch 110 loss 4.931749231097528 correct 37
Epoch 120 loss 5.091035984183715 correct 37
Epoch 130 loss 3.972453291543771 correct 37
Epoch 140 loss 4.904329101892547 correct 38
Epoch 150 loss 4.558231905198814 correct 38
Epoch 160 loss 4.412018376251930 correct 39
Epoch 170 loss 3.893294615202991 correct 39
Epoch 180 loss 4.326512872832917 correct 40
Epoch 190 loss 7.075341924893731 correct 39
Epoch 200 loss 5.185981232798451 correct 39
Epoch 210 loss 3.298542180721841 correct 39
Epoch 220 loss 4.002719838254201 correct 39
Epoch 230 loss 4.067315502191872 correct 40
Epoch 240 loss 3.153081452892739 correct 40
Epoch 250 loss 5.129572846109021 correct 41
Epoch 260 loss 3.926581812439074 correct 42
Epoch 270 loss 2.849792306573218 correct 41
Epoch 280 loss 2.189029453092771 correct 43
Epoch 290 loss 3.284931212051301 correct 44
Epoch 300 loss 2.5062357634281773 correct 44
Epoch 310 loss 2.821094123847893 correct 47
Epoch 320 loss 3.206571092013112 correct 47
Epoch 330 loss 2.202345912940162 correct 49
Epoch 340 loss 3.999711265089487 correct 49
Epoch 350 loss 2.491821342109732 correct 49
Epoch 360 loss 4.063572194183601 correct 45
Epoch 370 loss 1.689091873452109 correct 48
Epoch 380 loss 1.039283231497179 correct 49
Epoch 390 loss 1.419674841215893 correct 48
Epoch 400 loss 3.002391241761932 correct 47
Epoch 410 loss 2.563491172307218 correct 49
Epoch 420 loss 2.158934621793801 correct 48
Epoch 430 loss 2.487691021037812 correct 47
Epoch 440 loss 0.384097123241729 correct 49
Epoch 450 loss 2.296521892091381 correct 50
Epoch 460 loss 2.172509124681217 correct 49
Epoch 470 loss 2.036731145739582 correct 50
Epoch 480 loss 2.124893601092517 correct 50
Epoch 490 loss 1.095837621398372 correct 50
Mean time taken 10 epochs: 12.239502158920731

```

# Large Model

CPU(200 split)
```
cpu --HIDDEN 200 --DATASET split --RATE 0.05
```

```
Epoch 0 loss 5.912483201739417 correct 36
Epoch 10 loss 4.685714587392761 correct 43
Epoch 20 loss 4.287019752319641 correct 46
Epoch 30 loss 2.401982741123047 correct 47
Epoch 40 loss 4.408122045678935 correct 42
Epoch 50 loss 2.7293159834267915 correct 44
Epoch 60 loss 3.2829410511836213 correct 46
Epoch 70 loss 1.158793120589476 correct 47
Epoch 80 loss 1.4129036428903925 correct 48
Epoch 90 loss 2.985123740672158 correct 45
Epoch 100 loss 1.5419283190563194 correct 47
Epoch 110 loss 0.746290374182467 correct 46
Epoch 120 loss 1.3291021762484568 correct 49
Epoch 130 loss 1.2654704892833417 correct 50
Epoch 140 loss 0.2043748598372163 correct 50
Epoch 150 loss 1.042938104729372 correct 50
Epoch 160 loss 0.8269457482937196 correct 49
Epoch 170 loss 1.7530186729381878 correct 48
Epoch 180 loss 1.3102027937162056 correct 50
Epoch 190 loss 1.5028471392763457 correct 49
Epoch 200 loss 3.1027496218637843 correct 46
Epoch 210 loss 1.804922731928695 correct 46
Epoch 220 loss 0.6978131243928616 correct 50
Epoch 230 loss 0.980453618394832 correct 49
Epoch 240 loss 1.1531205429730196 correct 50
Epoch 250 loss 0.9778296341825639 correct 48
Epoch 260 loss 0.15294783904754253 correct 49
Epoch 270 loss 0.6121049372101484 correct 50
Epoch 280 loss 0.3679023058159401 correct 49
Epoch 290 loss 0.24789302783013296 correct 49
Epoch 300 loss 0.08919457462830715 correct 50
Epoch 310 loss 0.4937518426821686 correct 50
Epoch 320 loss 0.8205649237605983 correct 49
Epoch 330 loss 0.2427658201904061 correct 50
Epoch 340 loss 1.0629132049176015 correct 49
Epoch 350 loss 1.3827428301972542 correct 49
Epoch 360 loss 1.3921028340912787 correct 50
Epoch 370 loss 0.2473051921782062 correct 48
Epoch 380 loss 0.18403927851924337 correct 50
Epoch 390 loss 0.9214836751925398 correct 49
Epoch 400 loss 1.1174938273921804 correct 50
Epoch 410 loss 0.6483210981294724 correct 50
Epoch 420 loss 1.0348293651824976 correct 50
Epoch 430 loss 0.2718459372915743 correct 50
Epoch 440 loss 0.773290156708954 correct 50
Epoch 450 loss 0.5243298370920351 correct 50
Epoch 460 loss 0.3521932047194515 correct 50
Epoch 470 loss 0.7805421349812496 correct 50
Epoch 480 loss 1.2781092646173265 correct 50
Epoch 490 loss 0.03017927530421519 correct 50
Mean time taken 10 epochs: 21.270482915918544
```

GPU(200 split)
```
gpu --HIDDEN 200 --DATASET split --RATE 0.05
```

```
Epoch 0 loss 27.102938412593845 correct 18
Epoch 10 loss 6.834272839118712 correct 28
Epoch 20 loss 3.502011672831145 correct 37
Epoch 30 loss 3.132648991274502 correct 43
Epoch 40 loss 2.518743290468729 correct 46
Epoch 50 loss 5.869302491573812 correct 43
Epoch 60 loss 2.341079488293412 correct 47
Epoch 70 loss 0.6234990283740918 correct 45
Epoch 80 loss 2.3421498397537824 correct 49
Epoch 90 loss 1.4281213457082756 correct 50
Epoch 100 loss 0.712938472901278 correct 49
Epoch 110 loss 0.8120914739201831 correct 50
Epoch 120 loss 0.4599385814723009 correct 50
Epoch 130 loss 0.2923840719374163 correct 50
Epoch 140 loss 0.9972834812907624 correct 50
Epoch 150 loss 3.0123489174357623 correct 49
Epoch 160 loss 0.8640191748305729 correct 50
Epoch 170 loss 0.881129175928102 correct 50
Epoch 180 loss 0.6937482192841378 correct 50
Epoch 190 loss 0.8234562110911925 correct 50
Epoch 200 loss 0.6752935827635198 correct 49
Epoch 210 loss 0.726492837462872 correct 50
Epoch 220 loss 1.724839102312034 correct 48
Epoch 230 loss 1.3910281937274355 correct 46
Epoch 240 loss 0.5810293747198724 correct 50
Epoch 250 loss 0.8927341993725135 correct 50
Epoch 260 loss 2.689038472493716 correct 48
Epoch 270 loss 0.1792019342841631 correct 47
Epoch 280 loss 0.7852387239427153 correct 50
Epoch 290 loss 2.619174932038172 correct 50
Epoch 300 loss 0.17389240293649572 correct 50
Epoch 310 loss 8.457198310284616 correct 46
Epoch 320 loss 0.5287329118203421 correct 50
Epoch 330 loss 0.1839482172948346 correct 48
Epoch 340 loss 0.8737195439201343 correct 50
Epoch 350 loss 0.7214982938476164 correct 50
Epoch 360 loss 0.6024937137289425 correct 49
Epoch 370 loss 0.5468274939128316 correct 50
Epoch 380 loss 1.285638291927839 correct 46
Epoch 390 loss 0.2627389418383925 correct 48
Epoch 400 loss 0.115302741927502 correct 50
Epoch 410 loss 1.2034981049201832 correct 47
Epoch 420 loss 0.5432391283847163 correct 50
Epoch 430 loss 0.5948920174856349 correct 50
Epoch 440 loss 0.6947382910291735 correct 50
Epoch 450 loss 0.2384923847109384 correct 50
Epoch 460 loss 1.5730123748291376 correct 50
Epoch 470 loss 0.482749301027492 correct 49
Epoch 480 loss 0.5327493291846159 correct 50
Epoch 490 loss 0.14523947384920932 correct 50
Mean time taken 10 epochs: 24.761293847610473

```