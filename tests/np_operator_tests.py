import unittest

import numpy as np

import vexpr as vp
import vexpr.custom.numpy as vcnp
import vexpr.custom.scipy as vcsp
import vexpr.numpy as vnp
import vexpr.scipy as vscipy


from jax.tree_util import tree_map


class TestNumpyOperatorTests(unittest.TestCase):
    def test_multiply_with_np_array(self):
        import vexpr.core as c
        w = vp.symbol("w")
        x = c.constant(np.array([[1.0, 2.0]]))
        self._assert_vexprs_equal(w * x, c.operator_mul(w, x))
        self._assert_vexprs_equal(x * w, c.operator_mul(x, w))

    def test_stack_weights_multiply(self):
        example_inputs = dict(
            x1=np.random.randn(10, 5),
            x2=np.random.randn(10, 5),
            w1=np.array(0.7),
            w2=np.array(0.3),
        )

        @vp.vectorize
        def f(x1, x2, w1, w2):
            return vnp.sum(
                [w1 * vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                                    x2[..., [0, 1, 2]]),
                 w2 * vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                                    x2[..., [0, 3, 4]])],
                axis=0)

        @vp.make_vexpr
        def expected(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vnp.reshape(vnp.stack([w1, w2]), (2, 1, 1))
                * vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3])),
                axis=0)

        self._vectorize_test(example_inputs, f, expected)

        # now flip the multiplication around and test again

        @vp.vectorize
        def f_flipped(x1, x2, w1, w2):
            return vnp.sum(
                [vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                               x2[..., [0, 1, 2]])
                 * w1,
                 vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                               x2[..., [0, 3, 4]])
                 * w2],
                axis=0)

        @vp.make_vexpr
        def expected_flipped(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3]))
                * vnp.reshape(vnp.stack([w1, w2]), (2, 1, 1)),
                axis=0)

        self._vectorize_test(example_inputs, f_flipped, expected_flipped)

    @unittest.skip("Not yet supported")
    def test_stack_multiply_side_by_shape(self):
        example_inputs = dict(
            x1=np.random.randn(10, 5),
            x2=np.random.randn(10, 5),
            w1=np.array(0.7),
            w2=np.array(0.3),
        )

        @vp.vectorize
        def f(x1, x2, w1, w2):
            return vnp.sum(
                [w1 * vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                                    x2[..., [0, 1, 2]]),
                 vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                               x2[..., [0, 3, 4]]) * w2],
                axis=0)

        @vp.make_vexpr
        def expected(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vnp.reshape(vnp.stack([w1, w2]), (2, 1, 1))
                * vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3])),
                axis=0)

        self._vectorize_test(example_inputs, f, expected)

        # now flip the multiplication around and test again

        @vp.vectorize
        def f_flipped(x1, x2, w1, w2):
            return vnp.sum(
                [vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                               x2[..., [0, 1, 2]])
                 * w1,
                 w2 * vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                                    x2[..., [0, 3, 4]])],
                axis=0)

        @vp.make_vexpr
        def expected_flipped(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3]))
                * vnp.reshape(vnp.stack([w1, w2]), (2, 1, 1)),
                axis=0)

        self._vectorize_test(example_inputs, f_flipped, expected_flipped)

    @unittest.skip("Not yet supported")
    def test_stack_multiply_side_by_symbol(self):
        example_inputs = dict(
            x=np.array([42.0, 44.0, 46.0]),
            w1=np.array(0.7),
            w2=np.array(0.8),
        )

        @vp.vectorize
        def f(x, w1, w2):
            return vnp.stack([
                w1 * x[0],
                x[2] * w2,
            ])

        @vp.make_vexpr
        def expected(x, w1, w2):
            return (vnp.stack([w1, w2])
                    * x[np.array([0, 2])])

        self._vectorize_test(example_inputs, f, expected)

        # now flip the multiplication around and test again

        @vp.vectorize
        def f_flipped(x, w1, w2):
            return vnp.stack([
                x[0] * w1,
                w2 * x[2],
            ])

        @vp.make_vexpr
        def expected_flipped(x, w1, w2):
            return (x[np.array([0, 2])]
                    * vnp.stack([w1, w2]))

        self._vectorize_test(example_inputs, f_flipped, expected_flipped)

    @unittest.skip("Not yet supported")
    def test_stack_multiply_identity_side_by_symbol1(self):
        example_inputs = dict(
            x1=np.random.randn(10, 5),
            x2=np.random.randn(10, 5),
            w1=np.array(0.7),
        )

        @vp.vectorize
        def f(x1, x2, w1, w2):
            return vnp.sum(
                [w1 * vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                                    x2[..., [0, 1, 2]]),
                 vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                               x2[..., [0, 3, 4]])],
                axis=0)

        @vp.make_vexpr
        def expected(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vnp.reshape(vnp.stack([w1, 1.0]), (2, 1, 1))
                * vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3])),
                axis=0)

        self._vectorize_test(example_inputs, f, expected)

        # now flip the multiplication around, flip where the identity occurs,
        # and test again

        example_inputs = dict(
            x1=np.random.randn(10, 5),
            x2=np.random.randn(10, 5),
            w2=np.array(0.3),
        )

        @vp.vectorize
        def f_flipped(x1, x2, w1, w2):
            return vnp.sum(
                [vscipy.spatial.distance.cdist(x1[..., [0, 1, 2]],
                                               x2[..., [0, 1, 2]]),
                 vscipy.spatial.distance.cdist(x1[..., [0, 3, 4]],
                                               x2[..., [0, 3, 4]]) * w2],
                axis=0)

        @vp.make_vexpr
        def expected_flipped(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3]))
                *  vnp.reshape(vnp.stack([1.0, w2]), (2, 1, 1)),
                axis=0)

        self._vectorize_test(example_inputs, f_flipped, expected_flipped)

    @unittest.skip("Not yet supported")
    def test_stack_multiply_identity_side_by_symbol2(self):
        example_inputs = dict(
            x=np.array([42.0, 44.0, 46.0]),
            w1=np.array(0.7),
        )

        @vp.vectorize
        def f(x, w1, w2):
            return vnp.stack([
                w1 * x[0],
                x[2],
            ])

        @vp.make_vexpr
        def expected(x, w1, w2):
            return (vnp.stack([w1, 1.0])
                    * x[np.array([0, 2])])

        self._vectorize_test(example_inputs, f, expected)

        # now flip the multiplication around, flip where the identity occurs,
        # and test again

        example_inputs = dict(
            x=np.array([42.0, 44.0, 46.0]),
            w2=np.array(0.8),
        )

        @vp.vectorize
        def f_flipped(x, w1, w2):
            return vnp.stack([
                x[0],
                x[2] * w2,
            ])

        @vp.make_vexpr
        def expected_flipped(x, w1, w2):
            return (x[np.array([0, 2])]
                    * vnp.stack([1.0, w2]))

        self._vectorize_test(example_inputs, f_flipped, expected_flipped)

    def _vectorize_test(self, example_inputs, f, expected_after):
        before_result = f(**example_inputs)
        after = f.vexpr

        self._assert_vexprs_equal(after, expected_after.vexpr)

        after_result = f(**example_inputs)
        np.testing.assert_equal(before_result, after_result)

    def _assert_vexprs_equal(self, vexpr1, vexpr2):
        # Equality checks are a pain when there might be numpy arrays in the
        # objects. Test the types and values separately.
        vexpr1_types, vexpr2_types = tree_map(
            lambda x: (x.dtype
                       if isinstance(x, np.ndarray)
                       else x),
            (vexpr1, vexpr2))
        vexpr1_no_np, vexpr2_no_np = tree_map(
            lambda x: (x.tolist()
                       if isinstance(x, np.ndarray)
                       else x),
            (vexpr1, vexpr2))

        self.assertEqual(vexpr1_types, vexpr2_types)
        self.assertEqual(vexpr1_no_np, vexpr2_no_np)



if __name__ == '__main__':
    unittest.main()
