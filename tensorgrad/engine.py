"""
This will be like micrograd, but will operate over tensors implemented as python lists (can use NumPy later is wanting to). Also we will be generic with the data types for now but in the future, we will need a field dtype. Allows us to later use numpy and all it's benefits. see: ../speed_of_numpy.md

--> Since python lists are flat, begin my making a function to get a specific index  

 Fields in a tensor:
 - data
 - grad (gradient of loss with respect to each value in tensor, same shape as tensor)
 - shape
 - stride (for movement ops)
 - _children (for the computation graph)
 - _backward (function to backpropagate the gradient to the children)


Features: 
    - easy tensor initialization: ones(), zeros(), randn()

    Foundational Ops:
        - elemetwise ops: +, -, / , **, log, exp, relu
        - reduce ops: sum, mean, max, min
        - movement ops (change the way data is viewed): reshape, transpose, broadcast
        - Like pytorch and tinygrad allow shape: (n,) (arbitary vector). Will have to hanle converting it to at least a 
        2D vector though, doing our operations and then converting it back. The key is that our foundational operations/
        kernel should not be responsible for this. The logic for reshaping to 2D then converting back will be handled at a higher level. Here will be the logic:

        - # Case 1: both 1D → inner product → scalar (shape ()), ex: (n,) @ (n,) --> scalar
        -  # Case 2: 1D @ 2D → treat a as (1, n), matmul, squeeze leading dim, ex: (n,) @ (n,m) --> (m,)
        - # Case 3: 2D @ 1D → treat b as (n, 1), matmul, squeeze trailing dim, ex: (m, n) @ (n,) → (m,)
        - # Case 4: both >= 2D → standard matmul

        -Broadcasting Rules (Broadcasting should also be implemented using strides):
            **Step 1 — Align ranks from the right** by padding `1`s on the left of the shorter shape:
            ```
            a: (      3, 4)   →   (1, 3, 4)
            b: (2, 1, 4)      →   (2, 1, 4)
            ```
            **Step 2 — For each dimension, the sizes must either match or one of them is `1`:**
            ```
            dim 0:  1 vs 2  → ok, broadcast a → 2
            dim 1:  3 vs 1  → ok, broadcast b → 3
            dim 2:  4 vs 4  → ok, they match  → 4

            output shape: (2, 3, 4)
            
            **Step 3 — If any dimension fails both conditions, raise an error.

            The decision tree for a binary op `a OP b`:
            ```
            1. Are shapes identical?           → operate directly, no broadcast needed
            2. Can shapes be broadcast?        → compute output shape, expand strides
            3. Otherwise                       → raise ShapeError
            4. Do the kernel operation (addition, substraction, multiplication, etc. )

    
    Complex Ops:
        - Matrix Multiplication: @
        - Convolutions (make use of matrix multiplication here)

    Make sure @ is implemented from those foundational ops

"""

# data --> 1D array representing the data
# shape --> how the tensor looks like to the user/ its current state
# strides --> used to obtain sepcific elements for each index in shape , i.e: used to translate between shape and actual data values
import math
class Tensor:
    def __init__(self, data: list, shape: tuple):
        # Always validate entry points of data, most validation needs to happen here
        # not enough elements
        if len(data) != math.prod(shape):
            raise ValueError("insufficient length of data to produce tensor")
        
        if not isinstance(data, (tuple,list)):
            raise TypeError("data should be a list/tuple")
        
        if not isinstance(shape, (tuple,list)):
            raise TypeError("shape should be a list/tuple")
        
        self.data = list(data)
        # defensive programming, even though expecting tuple
        self.shape = tuple(shape)
        self.strides = compute_strides(self.shape)
        self.grad = None
        self._backward = lambda: None
        self._children = []
    

    def _position_from_indices(self, indices) -> int:
        if len(indices) != len(self.shape):
            raise IndexError("wrong number of indices")

        pos = 0
        for i, val in enumerate(indices):
            if val < 0 or val >= self.shape[i]:
                raise IndexError("index out of range")
            pos += val * self.strides[i]
        return pos
    
    # get the data point at indices (respects tensor shape)
    # indices is 0-indexed as usual
    def get(self, indices):
        pos = self._position_from_indices(indices)
        return self.data[pos]

    def set(self, indices, val):
        pos = self._position_from_indices(indices)
        self.data[pos] = val

    # this is not the full optimizations even, partial reshapes can break this contiguity but still not require a copy. Can take a look at numpy to see how they fully do it later. This will be used in the reshape algo
    def _is_contiguous(self) -> bool:
        expected_stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.shape[i] == 1:
                continue                          # stride irrelevant, skip
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        return True
    
    def reshape():
        pass

    # take a look at the .T notation in python
    def transpose():
        pass
        
    def broadcast():
        pass

    def __repr__(self):
        return f"Tensor(shape: {self.shape}, data: {self.data})"


def compute_strides(shape):
    strides = [0] * len(shape)
    cur_stride = 1
    for i in range(len(shape) - 1, -1,-1):
        strides[i] = cur_stride
        cur_stride *= shape[i]

    return tuple(strides)

# this will be a generator, kind of like the range function in python
def ndindex(shape: tuple):
    def helper(dim):
        if dim == len(shape):
            yield ()
            return
        # iterate over all possible indices, return a tuple
        for i in range(shape[dim]):
            for next_tuple in helper(dim + 1):
                yield (i,) + next_tuple

    yield from helper(0)

def zeros(shape: tuple) -> Tensor:
    return Tensor([0] * math.prod(shape), shape)

def ones(shape: tuple) -> Tensor:
    return Tensor([1] * math.prod(shape), shape)


test_T = ones((2,3,4))
print(test_T)
