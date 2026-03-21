import sys
import unittest
from pathlib import Path

# Ensure `tensorgrad` package resolves to tensorgrad/tensorgrad
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from tensorgrad import Tensor, broadcast_shape, compute_strides, ndindex


class TestTensorConstruction(unittest.TestCase):
    def test_constructor_infers_shape_and_flattens_nested_data(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.data, [1, 2, 3, 4, 5, 6])
        self.assertEqual(t.strides, (3, 1))

    def test_constructor_with_explicit_shape_uses_flat_data(self):
        t = Tensor([1, 2, 3, 4], shape=[2, 2])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.data, [1, 2, 3, 4])

    def test_constructor_rejects_invalid_data_type(self):
        with self.assertRaises(TypeError):
            Tensor(123)  # type: ignore[arg-type]

    def test_constructor_rejects_invalid_shape_type(self):
        with self.assertRaises(TypeError):
            Tensor([1, 2, 3, 4], shape=4)  # type: ignore[arg-type]

    def test_constructor_rejects_data_length_shape_mismatch(self):
        with self.assertRaises(ValueError):
            Tensor([1, 2, 3], shape=(2, 2))

    def test_zeros_and_ones_accept_variadic_and_tuple_shapes(self):
        z1 = Tensor.zeros(2, 3)
        z2 = Tensor.zeros((2, 3))
        o1 = Tensor.ones(2, 3)
        o2 = Tensor.ones((2, 3))

        self.assertEqual(z1.shape, (2, 3))
        self.assertEqual(z2.shape, (2, 3))
        self.assertEqual(o1.shape, (2, 3))
        self.assertEqual(o2.shape, (2, 3))
        self.assertTrue(all(v == 0 for v in z1.data + z2.data))
        self.assertTrue(all(v == 1 for v in o1.data + o2.data))

    def test_normalize_shape_args_helper(self):
        self.assertEqual(Tensor._normalize_shape_args((2, 3)), (2, 3))
        self.assertEqual(Tensor._normalize_shape_args(((2, 3),)), (2, 3))
        self.assertEqual(Tensor._normalize_shape_args(([2, 3],)), (2, 3))


class TestIndexingAndMetadata(unittest.TestCase):
    def test_position_from_indices_valid_and_invalid(self):
        t = Tensor([0, 1, 2, 3, 4, 5], shape=(2, 3))
        self.assertEqual(t._position_from_indices((1, 2)), 5)

        with self.assertRaises(IndexError):
            t._position_from_indices((1,))

        with self.assertRaises(IndexError):
            t._position_from_indices((2, 0))

        with self.assertRaises(IndexError):
            t._position_from_indices((-1, 0))

    def test_get_and_set_round_trip(self):
        t = Tensor.zeros(2, 2)
        t.set((1, 1), 9)
        self.assertEqual(t.get((1, 1)), 9)

    def test_len_and_repr(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertEqual(len(t), 4)
        self.assertIn("shape: (2, 2)", repr(t))
        self.assertIn("data: [1, 2, 3, 4]", repr(t))

    def test_is_contiguous_true_for_base_false_for_permuted_view(self):
        t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        self.assertTrue(t._is_contiguous())

        tv = t.permute(1, 0)
        self.assertFalse(tv._is_contiguous())


class TestMovementOps(unittest.TestCase):
    def test_broadcast_to_expands_shape_and_reuses_values(self):
        t = Tensor([10, 20, 30], shape=(1, 3))
        b = t.broadcast_to((2, 3))

        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.get((0, 0)), 10)
        self.assertEqual(b.get((0, 2)), 30)
        self.assertEqual(b.get((1, 0)), 10)
        self.assertEqual(b.get((1, 2)), 30)

    def test_broadcast_to_rejects_lower_rank_target(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        with self.assertRaises(ValueError):
            t.broadcast_to((2,))

    def test_broadcast_rejects_incompatible_dimensions(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        with self.assertRaises(ValueError):
            t.broadcast_to((2, 3))

    def test_reshape_contiguous_behaves_as_view(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        r = t.reshape((4,))

        self.assertEqual(r.shape, (4,))
        t.set((0, 0), 99)
        self.assertEqual(r.get((0,)), 99)

    def test_reshape_non_contiguous_creates_copy_in_logical_order(self):
        t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        tv = t.permute(1, 0)   # shape (3, 2), non-contiguous logical view
        r = tv.reshape((6,))   # triggers copy path

        self.assertEqual(r.shape, (6,))
        self.assertEqual(r.data, [1, 4, 2, 5, 3, 6])

        t.set((0, 0), 99)
        self.assertEqual(r.get((0,)), 1)

    def test_reshape_rejects_mismatched_element_count(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        with self.assertRaises(ValueError):
            t.reshape((3, 2))

    def test_T_transposes_2d_tensor(self):
        t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        tt = t.T

        self.assertEqual(tt.shape, (3, 2))
        self.assertEqual(tt.get((0, 0)), 1)
        self.assertEqual(tt.get((0, 1)), 4)
        self.assertEqual(tt.get((2, 1)), 6)

    def test_T_rejects_non_2d_tensor(self):
        t = Tensor.ones(2, 3, 4)
        with self.assertRaises(ValueError):
            _ = t.T

    def test_permute_reorders_axes(self):
        t = Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        p = t.permute(1, 0)

        self.assertEqual(p.shape, (3, 2))
        self.assertEqual(p.strides, (1, 3))
        self.assertEqual(p.get((2, 1)), 6)


class TestModuleHelpers(unittest.TestCase):
    def test_broadcast_shape_compatible(self):
        self.assertEqual(broadcast_shape((3, 1, 4), (1, 5, 4)), (3, 5, 4))
        self.assertEqual(broadcast_shape((3, 4), (2, 1, 4)), (2, 3, 4))

    def test_broadcast_shape_incompatible_raises(self):
        with self.assertRaises(ValueError):
            broadcast_shape((2, 3), (4, 3))

    def test_compute_strides(self):
        self.assertEqual(compute_strides((2, 3, 4)), (12, 4, 1))
        self.assertEqual(compute_strides((5,)), (1,))
        self.assertEqual(compute_strides(()), ())

    def test_ndindex(self):
        self.assertEqual(list(ndindex((2, 2))), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(list(ndindex((3,))), [(0,), (1,), (2,)])
        self.assertEqual(list(ndindex(())), [()])


if __name__ == "__main__":
    unittest.main()
