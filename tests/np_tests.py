import unittest

import numpy as np

import vexpr as vp
import vexpr.custom.numpy as vcnp
import vexpr.custom.scipy as vcsp
import vexpr.numpy as vnp
import vexpr.scipy as vscipy


from jax.tree_util import tree_map


class TestVexprNumpyTests(unittest.TestCase):
    def test_sum_impl(self):
        example_inputs = dict(
            x1=np.full((3, 3), 1.0),
            x2=np.full((3, 3), 2.0),
        )

        @vp.make_vexpr
        def f(x1, x2):
            return vnp.sum([x1, x2], axis=0)

        expected = np.full((3, 3), 3.0)
        result = f(**example_inputs)
        np.testing.assert_equal(result, expected)

    def test_sum_to_add_at(self):
        example_inputs = dict(
            x1=np.full((3, 3), 1.0),
            x2=np.full((3, 3), 2.0),
        )

        @vp.vectorize
        def f(x1, x2):
            return vnp.stack([vnp.sum([x1, x2], axis=0),
                              vnp.sum([x1, x2], axis=0)])

        @vp.make_vexpr
        def expected(x1, x2):
            return vnp.add_at(vnp.zeros((2, 3, 3)),
                              np.array([0, 0, 1, 1]),
                              vnp.stack([x1, x2, x1, x2]))

        self._vectorize_test(example_inputs, f, expected)

    def test_sum_vectorize_single(self):
        example_inputs = dict(a=1, b=2)

        @vp.vectorize
        def f(a, b):
            return vnp.sum([a, b])

        @vp.make_vexpr
        def expected(a, b):
            return vnp.sum(vnp.stack([a, b]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_sum_vectorize_stack_scalars(self):
        example_inputs = dict(a=1, b=2, c=3, d=4)

        @vp.vectorize
        def f(a, b, c, d):
            return vnp.stack([vnp.sum([a, b]),
                              vnp.sum([c, d])])

        @vp.make_vexpr
        def expected(a, b, c, d):
            return vnp.add_at(vnp.zeros((2,)),
                              np.array([0, 0, 1, 1]),
                              vnp.stack([a, b, c, d]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_sum_vectorize_concat_vectors(self):
        example_inputs = dict(a=np.array([1, 2]),
                              b=np.array([2, 3]),
                              c=np.array([2, 3, 4]),
                              d=np.array([3, 4, 5]))

        @vp.vectorize
        def f(a, b, c, d):
            return vnp.concatenate([vnp.sum([a, b], axis=0),
                                    vnp.sum([c, d], axis=0)])

        @vp.make_vexpr
        def expected(a, b, c, d):
            return vnp.add_at(vnp.zeros((5,)),
                              np.array([0, 1, 0, 1, 2, 3, 4, 2, 3, 4]),
                              vnp.concatenate([a, b, c, d]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_select(self):
        example_inputs = dict(x=np.array([10, 11, 12, 13, 14]))

        @vp.vectorize
        def f(x):
            return vnp.stack([vnp.sum(x[..., [0, 1, 2]]),
                              vnp.sum(x[..., [2, 3, 4]])])

        @vp.make_vexpr
        def expected(x):
            return vnp.add_at(vnp.zeros((2,)),
                              np.array([0, 0, 0, 1, 1, 1]),
                              x[..., [0, 1, 2, 2, 3, 4]])

        self._vectorize_test(example_inputs, f, expected)

    def test_kernel(self):
        example_inputs = dict(
            x1=np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.5, 0.5, 0.5, 0.5, 0.5]]),
            x2=np.array([[1.0, 0.9, 1.0, 0.9, 1.0],
                         [0.5, 0.4, 0.5, 0.4, 0.5]]),
            lengthscale=np.array([0.7, 0.8, 0.9, 1.0, 1.1]),
            scale=np.array(1.2),
        )

        @vp.vectorize
        def kernel(x1, x2, scale, lengthscale):
            return scale * (
                vnp.sum([vscipy.spatial.distance.cdist(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices])
                         for indices in ([0, 1, 2],
                                         [2, 3, 4])],
                        axis=0))

        @vp.make_vexpr
        def expected(x1, x2, scale, lengthscale):
            indices = np.array([0, 1, 2, 2, 3, 4])
            return scale * (
                vnp.sum(vcsp.cdist_multi(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices],
                    lengths=np.array([3, 3])),
                        axis=0))

        self._vectorize_test(example_inputs, kernel, expected)

    def test_weighted_additive_kernel(self):
        example_inputs = dict(
            x1=np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.4, 0.5, 0.5, 0.5, 0.5]]),
            x2=np.array([[1.0, 0.9, 1.0, 0.9, 1.0],
                         [0.5, 0.4, 0.5, 0.4, 0.5],
                         [0.4, 0.4, 0.5, 0.4, 0.5]]),
            beta_w=np.array([0.9, 0.1]),
            dirichlet_w=np.array([0.5, 0.2, 0.3]),
            lengthscale=np.array([0.7, 0.8, 0.9, 1.0, 1.1]),
            scale=np.array(1.2),
        )

        indices1 = [0]
        indices2 = [1]
        indices3 = [2]
        joint = indices1 + indices2 + indices3

        @vp.vectorize
        def kernel(x1, x2, scale, beta_w, dirichlet_w, lengthscale):

            def subkernel(indices):
                return vscipy.spatial.distance.cdist(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices]
                )

            additive_kernel = vnp.sum(dirichlet_w
                                      * vnp.stack([subkernel(indices1),
                                                   subkernel(indices2),
                                                   subkernel(indices3)],
                                                  axis=-1),
                                      axis=-1)
            joint_kernel = subkernel(joint)

            return scale * vnp.sum(beta_w *
                                   vnp.stack([additive_kernel,
                                              joint_kernel], axis=-1),
                                   axis=-1)

        @vp.make_vexpr
        def expected(x1, x2, scale, beta_w, dirichlet_w, lengthscale):
            indices = np.array(indices1 + indices2 + indices3 + joint)
            return scale * vnp.sum(
                beta_w
                * vnp.add_at(
                    vnp.zeros((3, 3, 2)),
                    (Ellipsis, np.array([0, 0, 0, 1])),
                    vnp.concatenate([dirichlet_w, np.ones(1)], axis=-1)
                    * vcsp.cdist_multi(
                        x1[..., indices] / lengthscale[indices],
                        x2[..., indices] / lengthscale[indices],
                        lengths=np.array([len(indices1), len(indices2),
                                          len(indices3), len(joint)]),
                        axis=-1)),
                axis=-1)

        self._vectorize_test(example_inputs, kernel, expected)

    @unittest.skip("Not yet supported")
    def test_mul_identity(self):
        inputs = dict(
            x=np.array([[1.0, 1.0, 1.0, 1.0],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.4, 0.5, 0.5, 0.5]]),
            w=np.array([0.7, 0.8, 0.9, 1.1]),
        )

        @vp.vectorize
        def f(w, x):
            return vnp.concatenate([w[[0, 1]] * x[..., [0, 1]],
                                    w[[1, 2]] * x[..., [1, 2]],
                                    x[..., [2]],
                                    x[..., [3, 2]],
                                    w[[3]] * x[..., [3]],
                                    x[..., [0, 3]]],
                                   axis=-1)

        @vp.make_vexpr
        def expected(w, x):
            # TODO note here that there is potential for a shuffle lift. if we
            # lift this shuffle, we can push it onto the x indices. if the final
            # result gets reduced, then all is equal.
            return (vcnp.shuffle(vnp.concatenate([w[[0, 1, 1, 2, 3, 2, 0, 3]],
                                                  np.ones(5)]),
                                 np.array([0, 1, 2, 3, 7, 4, 5, 6, 8, 9])),
                    * x[..., [0, 1, 1, 2, 2, 3, 2, 3, 0, 3]])

        @vp.make_vexpr
        def expected_alt(w, x):
            return (vnp.concatenate([w[[0, 1, 1, 2]],
                                     np.ones(3),
                                     w[[3]],
                                     np.ones(2)]),
                    * x[..., [0, 1, 1, 2, 2, 3, 2, 3, 0, 3]])

        self._vectorize_test(inputs, f, expected)

    def test_vectorized_vexpr_is_used(self):
        example_inputs = dict(
            x1=np.full((3, 3), 1.0),
            x2=np.full((3, 3), 2.0),
        )

        @vp.vectorize
        def f(x1, x2):
            return vnp.stack([vnp.sum([x1, x2], axis=0),
                              vnp.sum([x1, x2], axis=0)])

        @vp.make_vexpr
        def expected(x1, x2):
            return vnp.add_at(vnp.zeros((2, 3, 3)),
                              np.array([0, 0, 1, 1]),
                              vnp.stack([x1, x2, x1, x2]))


        import vexpr.core as core
        import vexpr.numpy.primitives as np_p
        orig = core.vectorize_impls[np_p.stack_p]
        trace = [False]
        def traced_vectorize(shapes, expr):
            trace[0] = True
            return orig(shapes, expr)
        core.vectorize_impls[np_p.stack_p] = traced_vectorize

        # first call should vectorize
        f(**example_inputs)
        self.assertTrue(trace[0])
        self._assert_vexprs_equal(f.vexpr, expected)

        # subsequent calls should not
        assert np_p.add_at_p not in core.vectorize_impls  # update test if this changes
        trace = [False]
        def traced_add_at_vectorize(shapes, expr):
            trace[0] = True
            return expr

        core.vectorize_impls[np_p.add_at_p] = traced_add_at_vectorize
        f(**example_inputs)
        self.assertFalse(trace[0])

    def test_readme(self):
        from vexpr.scipy.spatial.distance import cdist as v_cdist

        @vp.vectorize
        def f(x1, x2, w1, w2):
            return vnp.sum([w1 * v_cdist(x1[..., [0, 1, 2]], x2[..., [0, 1, 2]]),
                            w2 * v_cdist(x1[..., [0, 3, 4]], x2[..., [0, 3, 4]])],
                           axis=0)

        # print(f.vexpr)
        @vp.make_vexpr
        def expected_vectorized(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vnp.reshape(vnp.stack([w1, w2]), (2, 1, 1))
                * vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3])),
                axis=0)

        import vexpr.core

        @vp.make_vexpr
        def expected_pe(x1, x2, w1, w2):
            indices = np.array([0, 1, 2, 0, 3, 4])
            return vnp.sum(
                vexpr.core.operator_mul(
                    np.array([[[0.75]], [[0.25]]]),
                    vcsp.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=np.array([3, 3]))
                ),
                axis=0)

        example_inputs = dict(
            x1=np.random.randn(10, 5),
            x2=np.random.randn(10, 5),
            w1=np.array(0.7),
            w2=np.array(0.3),
        )

        f(**example_inputs)  # first call triggers compilation
        # print(f.vexpr)
        self._assert_vexprs_equal(f.vexpr, expected_vectorized)

        inference_f = vp.partial_evaluate(f, dict(w1=0.75, w2=0.25))
        print(inference_f)
        self._assert_vexprs_equal(inference_f, expected_pe)

    def _vectorize_test(self, example_inputs, f, expected_after):
        before_result = f(**example_inputs)
        after = f.vexpr

        self._assert_vexprs_equal(after, expected_after)

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
