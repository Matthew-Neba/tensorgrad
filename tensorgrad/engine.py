"""
Architecture Overview: 

Tensor (logical layer)
  ├── shape, strides, grad, _children, _backward   ← always lives here
  └── storage: Storage                              ← backend-specific

Storage (abstract interface)
  ├── ListStorage      ← current impl, will need to abstract away after basics are done
  ├── NumpyStorage     ← next step
  └── CudaStorage / MetalStorage (later)


This will be like micrograd, but will operate over tensors implemented as python lists (can use NumPy later is wanting to). Also we will be generic with the data types for now but in the future, we will need a field dtype. Allows us to later use numpy and all it's benefits. see: ../speed_of_numpy.md

--> Since python lists are flat, begin my making a function to get a specific index  

 Fields in a tensor:
 - data
 - shape
 - stride (for movement ops)
 - offset
 - op
 - grad (gradient of loss with respect to each value in tensor, same shape as tensor)
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

# data --> 1D array representing the data, will be converting to storage later per architecture at start
# shape --> how the tensor looks like to the user/ its current state
# strides --> used to obtain sepcific elements for each index in shape , i.e: used to translate between shape and actual data values
import math
class Tensor:
    """Minimal tensor with flat storage, shape metadata, and autograd placeholders."""
    def __init__(
        self,
        data: list,
        shape: tuple | list | None = None,
    ):
        """Create a tensor from nested data (infer shape) or flat data plus explicit shape."""
        # Always validate entry points of data, most validation needs to happen here
        # not enough elements
        if not isinstance(data, (tuple,list)):
            raise TypeError("data should be a list/tuple")
        
        # we were given a nested list, we need to infer the shape and flatten the array
        if not shape:
            self.data = list(self._flatten(data))
            self.shape = self._infer_shape(data)
        else:
            # if we were given shape, expected to be given data aswell
            if not isinstance(shape, (tuple, list)):
                raise TypeError("shape should be a tuple/list")
            
            if math.prod(shape) != len(data):
                raise ValueError("input data does not have enough elements for the specified shape")
            
            self.data = list(data)
            self.shape = tuple(shape)

        self.offset = 0
        self.op = "create"
        self.strides = compute_strides(self.shape)
        self.grad = None
        self._backward = lambda: None
        self._children = []
    
    # more flexible creation of a tensor
    @classmethod
    def _make_tensor(cls, data: list|tuple, shape: tuple | list, strides: tuple | list, offset: int, op: str) -> "Tensor":
        t_view = cls.__new__(cls)

        t_view.data = data if isinstance(data, list) else list(data)
        t_view.offset = offset
        t_view.shape = tuple(shape)
        t_view.strides = tuple(strides)
        t_view.op = op
        t_view.grad = None
        t_view._backward = lambda: None
        t_view._children = []

        return t_view

    @staticmethod
    def _normalize_shape_args(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Accept either variadic dims or a single tuple/list of dims."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return tuple(shape)

    def _infer_shape(self, data: list | tuple) -> tuple:
        """Infer tensor shape by descending through nested list/tuple levels."""
        shape = []
        current = data
        while isinstance(current, (list, tuple)):
            shape.append(len(current))
            current = current[0]
        
        return tuple(shape)

    def _flatten(self, data: list | tuple):
        """Yield scalar elements from nested list/tuple data in row-major order."""
        for item in data:
            if isinstance(item, (list, tuple)):
                yield from self._flatten(item)
            else:
                yield item
    
    @classmethod
    def zeros(cls, *shape:int) -> "Tensor":
        """Return a tensor of the given shape filled with zeros."""
        shape = cls._normalize_shape_args(shape)
        return cls([0] * math.prod(shape), shape)

    @classmethod
    def ones(cls,*shape:int) -> "Tensor":
        """Return a tensor of the given shape filled with ones."""
        shape = cls._normalize_shape_args(shape)
        return cls([1] * math.prod(shape), shape)

    def _position_from_indices(self, indices) -> int:
        """Convert N-D indices into a flat storage position using strides."""
        if len(indices) != len(self.shape):
            raise IndexError("wrong number of indices")

        pos = self.offset
        for i, val in enumerate(indices):
            if val < 0 or val >= self.shape[i]:
                raise IndexError("index out of range")
            pos += val * self.strides[i]
        return pos
    
    # get the data point at indices (respects tensor shape)
    # indices is 0-indexed as usual
    def get(self, indices):
        """Read one tensor element at the provided N-D index tuple/list."""
        pos = self._position_from_indices(indices)
        return self.data[pos]

    def set(self, indices, val):
        """Write one tensor element at the provided N-D index tuple/list."""
        pos = self._position_from_indices(indices)
        self.data[pos] = val

    # this is not the full optimizations even, partial reshapes can break this contiguity but still not require a copy. Can take a look at numpy to see how they fully do it later. This will be used in the reshape algo
    def _is_contiguous(self) -> bool:
        """Check whether current shape/strides represent contiguous row-major layout."""
        expected_stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.shape[i] == 1:
                continue                          # stride irrelevant, skip
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= self.shape[i]
        return True
    
    # TODO , ALL Movement ops are currently not handling grads and backwards/children. Will later
    # need for backpropagation

    # user facing api for broadcasting
    def broadcast_to(self, to_shape: list | tuple) -> "Tensor":
        if len(to_shape) < len(self.shape):
            raise ValueError("cannot broadcast to less dimensions")
        return self._broadcast(to_shape)

    # internal broadcasting helper
    def _broadcast(self, to_shape: list | tuple) -> "Tensor":
        # 1) Use rules to figure out broadcast dimensions, give them
        # 2) Use those broadcast dimensions with stride 0 in relevant fields to broadcast a tensor
        """Expand tensor view to a broadcast-compatible shape"""
        new_strides = [0] * len(to_shape)

        to_i , from_j = len(to_shape) - 1, len(self.shape) - 1
        while to_i >= 0:
            if from_j < 0:
                new_strides[to_i] = 0
            elif self.shape[from_j] == to_shape[to_i]:
                new_strides[to_i] = self.strides[from_j]
            elif self.shape[from_j] == 1:
                new_strides[to_i] = 0
            else:
                raise ValueError(f"cannot broadcast dim {from_j}: size {self.shape[from_j]} into {to_shape[to_i]}")

            to_i -= 1
            from_j -= 1

        return Tensor._make_tensor(self.data, tuple(to_shape), tuple(new_strides), self.offset, "_broadcast")
    
    def reshape(self, new_shape: tuple|list) -> "Tensor":
        """Return a view/copy with a new shape."""
        if not math.prod(new_shape) == math.prod(self.shape):
            raise ValueError("count of elements in the new shape don't match the count in the current shape")

        if self._is_contiguous():
            view = Tensor._make_tensor(self.data, new_shape, compute_strides(new_shape), self.offset,  "reshape")
            return view
        else:
            # neew to make a copy of the data, get the correct current order
            flat = [self.data[self._position_from_indices(idx)] for idx in ndindex(self.shape)]
            return Tensor._make_tensor(flat, new_shape, compute_strides(new_shape), 0, op="reshape_copy")
        
    @property
    def T(self):
        # call permute with correct arguments to only flip the last two dimensions
        if len(self.shape) != 2:
            raise ValueError("T only valid for 2D Tensors currently")
        
        return self.permute(1, 0)
    
    def permute(self, *permutation):
        """ Given a permutation of the new dimensions, return a new view of the tensor with the axis
        reversed. May or may not copy the underlying data structure. Does not when possible.
        """
        new_shape = [self.shape[x] for x in permutation]
        new_strides = [self.strides[x] for x in permutation]

        op = "transpose" if permutation == (1,0) else "permute"
        view = self._make_tensor(self.data, new_shape, new_strides, self.offset, op)

        return view
        
    def  __len__(self) -> int:
        """Return total number of elements in the tensor."""
        return math.prod(self.shape)
        
    def __repr__(self):
        """Return a debug-friendly string representation."""
        return f"Tensor(shape: {self.shape}, data: {self.data})"


def broadcast_shape(shape1: tuple, shape2: tuple) -> tuple:
    """Return the broadcasted output shape for two input shapes."""
    new_shape = []
    i, j = len(shape1) - 1, len(shape2) - 1
    while i >= 0 or j >= 0:
        a = shape1[i] if i >= 0 else 1
        b = shape2[j] if j >= 0 else 1

        if a != b and a != 1 and b != 1:
            raise ValueError(f"two shapes not compatible on dimensions: (shape1 : {i}, shape2: {j})")

        new_shape.append(max(a, b))
        i -= 1
        j -= 1
    return tuple(reversed(new_shape))
    
    

def compute_strides(shape: tuple) -> tuple:
    """Compute row-major strides for a given shape."""
    strides = [0] * len(shape)
    cur_stride = 1
    for i in range(len(shape) - 1, -1,-1):
        strides[i] = cur_stride
        cur_stride *= shape[i]

    return tuple(strides)

# this will be a generator, kind of like the range function in python
def ndindex(shape: tuple):
    """Yield all valid N-D index tuples for the provided shape."""
    def helper(dim):
        if dim == len(shape):
            yield ()
            return
        # iterate over all possible indices, return a tuple
        for i in range(shape[dim]):
            for next_tuple in helper(dim + 1):
                yield (i,) + next_tuple

    yield from helper(0)


if __name__ == "__main__":
    test_T = Tensor.ones(2, 3, 4)
    print(test_T)
