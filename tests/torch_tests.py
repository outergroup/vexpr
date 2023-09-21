import unittest

import torch

import vexpr as vp
import vexpr.custom.torch as vctorch
import vexpr.torch as vtorch


from jax.tree_util import tree_map


class TestVexprTorchTests(unittest.TestCase):
    def test_sum_impl(self):
        example_inputs = dict(
            x1=torch.full((3, 3), 1.0),
            x2=torch.full((3, 3), 2.0),
        )

        @vp.make_vexpr
        def f(x1, x2):
            return vtorch.sum([x1, x2], dim=0)

        expected = torch.full((3, 3), 3.0)
        result = f(**example_inputs)
        torch.testing.assert_close(result, expected)

    def test_sum_to_index_add(self):
        example_inputs = dict(
            x1=torch.full((3, 3), 1.0),
            x2=torch.full((3, 3), 2.0),
        )

        @vp.vectorize
        def f(x1, x2):
            return vtorch.stack([vtorch.sum([x1, x2], dim=0),
                                 vtorch.sum([x1, x2], dim=0)])

        @vp.make_vexpr
        def expected(x1, x2):
            return vctorch.index_add_into_zeros(2, 0,
                                                torch.tensor([0, 0, 1, 1]),
                                                vtorch.stack([x1, x2,
                                                              x1, x2]))

        self._vectorize_test(example_inputs, f, expected)

    def test_prod_to_index_reduce(self):
        example_inputs = dict(
            x1=torch.full((3, 3), 1.0),
            x2=torch.full((3, 3), 2.0),
        )

        @vp.vectorize
        def f(x1, x2):
            return vtorch.stack([vtorch.prod([x1, x2], dim=0),
                                 vtorch.prod([x1, x2], dim=0)])

        @vp.make_vexpr
        def expected(x1, x2):
            return vctorch.index_reduce_into_ones(2, 0,
                                                  torch.tensor([0, 0, 1, 1]),
                                                  vtorch.stack([x1, x2,
                                                                x1, x2]),
                                                  "prod")

        self._vectorize_test(example_inputs, f, expected)

    def test_sum_vectorize_single(self):
        example_inputs = dict(a=1, b=2)

        @vp.vectorize
        def f(a, b):
            return vtorch.sum([a, b])

        @vp.make_vexpr
        def expected(a, b):
            return vtorch.sum(vtorch.stack([a, b]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_sum_vectorize_stack_scalars(self):
        example_inputs = dict(a=1, b=2, c=3, d=4)

        @vp.vectorize
        def f(a, b, c, d):
            return vtorch.stack([vtorch.sum([a, b]),
                              vtorch.sum([c, d])])

        @vp.make_vexpr
        def expected(a, b, c, d):
            return vctorch.index_add_into_zeros(2, 0,
                                                torch.tensor([0, 0, 1, 1]),
                                                vtorch.stack([a, b, c, d]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_sum_vectorize_concat_vectors(self):
        example_inputs = dict(a=torch.tensor([1, 2]),
                              b=torch.tensor([2, 3]),
                              c=torch.tensor([2, 3, 4]),
                              d=torch.tensor([3, 4, 5]))

        @vp.vectorize
        def f(a, b, c, d):
            return vtorch.concat([vtorch.sum([a, b], dim=0),
                                    vtorch.sum([c, d], dim=0)])

        @vp.make_vexpr
        def expected(a, b, c, d):
            return vctorch.index_add_into_zeros(5, 0,
                                                torch.tensor([0, 1, 0, 1, 2, 3, 4, 2, 3, 4]),
                                                vtorch.concat([a, b, c, d]))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_select(self):
        example_inputs = dict(x=torch.tensor([10, 11, 12, 13, 14]))

        @vp.vectorize
        def f(x):
            return vtorch.stack([vtorch.sum(x[..., [0, 1, 2]]),
                              vtorch.sum(x[..., [2, 3, 4]])])

        @vp.make_vexpr
        def expected(x):
            return vctorch.index_add_into_zeros(2, 0,
                                                torch.tensor([0, 0, 0, 1, 1, 1]),
                                                x[..., [0, 1, 2, 2, 3, 4]])

        self._vectorize_test(example_inputs, f, expected)

    def test_kernel(self):
        example_inputs = dict(
            x1=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                             [0.5, 0.5, 0.5, 0.5, 0.5]]),
            x2=torch.tensor([[1.0, 0.9, 1.0, 0.9, 1.0],
                             [0.5, 0.4, 0.5, 0.4, 0.5]]),
            lengthscale=torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1]),
            scale=torch.tensor(1.2),
        )

        @vp.vectorize
        def kernel(x1, x2, scale, lengthscale):
            return scale * (
                vtorch.sum([vtorch.cdist(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices])
                            for indices in ([0, 1, 2],
                                            [2, 3, 4])],
                           dim=0))

        @vp.make_vexpr
        def expected(x1, x2, scale, lengthscale):
            indices = torch.tensor([0, 1, 2, 2, 3, 4])
            return scale * (
                vtorch.sum(vctorch.cdist_multi(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices],
                    lengths=torch.tensor([3, 3]),
                    ps=torch.tensor([2, 2])),
                           dim=0))

        self._vectorize_test(example_inputs, kernel, expected)

    def test_weighted_additive_kernel(self):
        example_inputs = dict(
            x1=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.4, 0.5, 0.5, 0.5, 0.5]]),
            x2=torch.tensor([[1.0, 0.9, 1.0, 0.9, 1.0],
                         [0.5, 0.4, 0.5, 0.4, 0.5],
                         [0.4, 0.4, 0.5, 0.4, 0.5]]),
            beta_w=torch.tensor([0.9, 0.1]),
            dirichlet_w=torch.tensor([0.5, 0.2, 0.3]),
            lengthscale=torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1]),
            scale=torch.tensor(1.2),
        )

        indices1 = [0]
        indices2 = [1]
        indices3 = [2]
        joint = indices1 + indices2 + indices3

        @vp.vectorize
        def kernel(x1, x2, scale, beta_w, dirichlet_w, lengthscale):

            def subkernel(indices):
                return vtorch.cdist(
                    x1[..., indices] / lengthscale[indices],
                    x2[..., indices] / lengthscale[indices]
                )

            additive_kernel = vtorch.sum(dirichlet_w
                                      * vtorch.stack([subkernel(indices1),
                                                   subkernel(indices2),
                                                   subkernel(indices3)],
                                                  dim=-1),
                                      dim=-1)
            joint_kernel = subkernel(joint)

            return scale * vtorch.sum(beta_w *
                                   vtorch.stack([additive_kernel,
                                              joint_kernel], dim=-1),
                                   dim=-1)

        @vp.make_vexpr
        def expected(x1, x2, scale, beta_w, dirichlet_w, lengthscale):
            indices = torch.tensor(indices1 + indices2 + indices3 + joint)
            return scale * vtorch.sum(
                beta_w
                * vctorch.index_add_into_zeros(
                    2, -1,
                    torch.tensor([0, 0, 0, 1]),
                    vtorch.concat([dirichlet_w, torch.ones(1)], dim=-1)
                    * vctorch.cdist_multi(
                        x1[..., indices] / lengthscale[indices],
                        x2[..., indices] / lengthscale[indices],
                        lengths=torch.tensor([len(indices1), len(indices2),
                                          len(indices3), len(joint)]),
                        ps=torch.tensor([2, 2, 2, 2]),
                        dim=-1)),
                dim=-1)

        self._vectorize_test(example_inputs, kernel, expected)

    @unittest.skip("Not yet supported")
    def test_mul_identity(self):
        inputs = dict(
            x=torch.tensor([[1.0, 1.0, 1.0, 1.0],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.4, 0.5, 0.5, 0.5]]),
            w=torch.tensor([0.7, 0.8, 0.9, 1.1]),
        )

        @vp.vectorize
        def f(w, x):
            return vtorch.concat([w[[0, 1]] * x[..., [0, 1]],
                                    w[[1, 2]] * x[..., [1, 2]],
                                    x[..., [2]],
                                    x[..., [3, 2]],
                                    w[[3]] * x[..., [3]],
                                    x[..., [0, 3]]],
                                   dim=-1)

        @vp.make_vexpr
        def expected(w, x):
            # TODO note here that there is potential for a shuffle lift. if we
            # lift this shuffle, we can push it onto the x indices. if the final
            # result gets reduced, then all is equal.
            return (vctorch.shuffle(vtorch.concat([w[[0, 1, 1, 2, 3, 2, 0, 3]],
                                                  torch.ones(5)]),
                                 torch.tensor([0, 1, 2, 3, 7, 4, 5, 6, 8, 9])),
                    * x[..., [0, 1, 1, 2, 2, 3, 2, 3, 0, 3]])

        @vp.make_vexpr
        def expected_alt(w, x):
            return (vtorch.concat([w[[0, 1, 1, 2]],
                                     torch.ones(3),
                                     w[[3]],
                                     torch.ones(2)]),
                    * x[..., [0, 1, 1, 2, 2, 3, 2, 3, 0, 3]])

        self._vectorize_test(inputs, f, expected)

    def test_vectorized_vexpr_is_used(self):
        example_inputs = dict(
            x1=torch.full((3, 3), 1.0),
            x2=torch.full((3, 3), 2.0),
        )

        @vp.vectorize
        def f(x1, x2):
            return vtorch.stack([vtorch.sum([x1, x2], dim=0),
                              vtorch.sum([x1, x2], dim=0)])

        @vp.make_vexpr
        def expected(x1, x2):
            return vctorch.index_add_into_zeros(2, 0,
                                                torch.tensor([0, 0, 1, 1]),
                                                vtorch.stack([x1, x2, x1, x2]))


        import vexpr.vectorization as v
        import vexpr.torch.primitives as torch_p
        orig = v.vectorize_impls[torch_p.stack_p]
        trace = [False]
        def traced_vectorize(expr):
            trace[0] = True
            return orig(expr)
        v.vectorize_impls[torch_p.stack_p] = traced_vectorize

        # first call should vectorize
        f(**example_inputs)
        self.assertTrue(trace[0])
        self._assert_vexprs_equal(f.vexpr, expected.vexpr)

        # subsequent calls should not
        assert torch_p.index_add_p not in v.vectorize_impls  # update test if this changes
        trace = [False]
        def traced_index_add_vectorize(expr):
            trace[0] = True
            return expr

        v.vectorize_impls[torch_p.index_add_p] = traced_index_add_vectorize
        f(**example_inputs)
        self.assertFalse(trace[0])

    def test_readme(self):
        @vp.vectorize
        def f(x1, x2, w1, w2):
            return vtorch.sum([w1 * vtorch.cdist(x1[..., [0, 1, 2]], x2[..., [0, 1, 2]]),
                            w2 * vtorch.cdist(x1[..., [0, 3, 4]], x2[..., [0, 3, 4]])],
                           dim=0)

        # print(f.vexpr)
        @vp.make_vexpr
        def expected_vectorized(x1, x2, w1, w2):
            indices = torch.tensor([0, 1, 2, 0, 3, 4])
            return vtorch.sum(
                vtorch.reshape(vtorch.stack([w1, w2]), (2, 1, 1))
                * vctorch.cdist_multi(
                    x1[..., indices],
                    x2[..., indices],
                    lengths=torch.tensor([3, 3]),
                    ps=torch.tensor([2, 2])),
                dim=0)

        import vexpr.core

        @vp.make_vexpr
        def expected_pe(x1, x2, w1, w2):
            indices = torch.tensor([0, 1, 2, 0, 3, 4])
            return vtorch.sum(
                vexpr.core.operator_mul(
                    torch.tensor([[[0.75]], [[0.25]]]),
                    vctorch.cdist_multi(
                        x1[..., indices],
                        x2[..., indices],
                        lengths=torch.tensor([3, 3]),
                        ps=torch.tensor([2, 2]))
                ),
                dim=0)

        example_inputs = dict(
            x1=torch.randn(10, 5),
            x2=torch.randn(10, 5),
            w1=torch.tensor(0.7),
            w2=torch.tensor(0.3),
        )

        f(**example_inputs)  # first call triggers compilation
        # print(f.vexpr)
        self._assert_vexprs_equal(f.vexpr, expected_vectorized.vexpr)

        inference_f = vp.partial_evaluate(f, dict(w1=0.75, w2=0.25))
        print(inference_f)
        self._assert_vexprs_equal(inference_f.vexpr, expected_pe.vexpr)

    def _enable_matern(self):
        import math
        from functools import partial

        import torch
        import vexpr.core
        import vexpr.torch.primitives as t_p
        import vexpr.vectorization as v
        from vexpr import Vexpr
        from vexpr.torch.register_pushthrough import (
            push_concat_through_unary_elementwise,
            push_stack_through_unary_elementwise,
        )

        matern_p, matern = vexpr.core._p_and_constructor("matern")

        def matern_impl(d, nu=2.5):
            assert nu == 2.5
            exp_component = torch.exp(-math.sqrt(5) * d)
            constant_component = 1. + (math.sqrt(5) * d) + (5. / 3.) * d**2
            return constant_component * exp_component

        vexpr.core.eval_impls[matern_p] = matern_impl
        v.vectorize_impls[matern_p] = v.unary_elementwise_vectorize
        v.pushthrough_impls[(t_p.stack_p, matern_p)] = partial(
            push_stack_through_unary_elementwise, matern_p)
        v.pushthrough_impls[(t_p.concat_p, matern_p)] = partial(
            push_concat_through_unary_elementwise, matern_p)

        return matern

    def test_branching(self):
        matern = self._enable_matern()

        example_inputs = dict(
            x1=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            x2=torch.tensor([[1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9],
                             [0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4]]),
        )

        @vp.vectorize
        def f(x1, x2):
            indices1 = [0, 1]
            indices2 = [2, 3]
            indices3 = [4, 5]
            indices4 = [6, 7]
            return vtorch.stack([
                matern(vtorch.cdist(x1[..., indices1], x2[..., indices1], p=2)),
                matern(vtorch.cdist(x1[..., indices2], x2[..., indices2], p=2)),
                vtorch.exp(-vtorch.cdist(x1[..., indices3], x2[..., indices3], p=1)),
                vtorch.exp(-vtorch.cdist(x1[..., indices4], x2[..., indices4], p=1)),
            ])

        @vp.make_vexpr
        def expected(x1, x2):
            indices12 = torch.tensor([0, 1, 2, 3])
            indices34 = torch.tensor([4, 5, 6, 7])
            return vctorch.shuffle(
                vtorch.concat([
                    matern(
                        vctorch.cdist_multi(
                            x1[..., indices12], x2[..., indices12],
                            lengths=torch.tensor([2, 2]),
                            ps=torch.tensor([2, 2]))),
                    vtorch.exp(
                        -vctorch.cdist_multi(
                            x1[..., indices34], x2[..., indices34],
                            lengths=torch.tensor([2, 2]),
                            ps=torch.tensor([1, 1])))
                ]),
                torch.tensor([0, 1, 2, 3])
            )

        self._vectorize_test(example_inputs, f, expected)

    def _vectorize_test(self, example_inputs, f, expected_after):
        before_result = f(**example_inputs)
        after = f.vexpr

        self._assert_vexprs_equal(after, expected_after.vexpr)

        after_result = f(**example_inputs)
        torch.testing.assert_close(before_result, after_result)

    def _assert_vexprs_equal(self, vexpr1, vexpr2):
        # Equality checks are a pain when there might be numpy arrays in the
        # objects. Test the types and values separately.
        vexpr1_types, vexpr2_types = tree_map(
            lambda x: (x.dtype
                       if isinstance(x, torch.Tensor)
                       else x),
            (vexpr1, vexpr2))
        vexpr1_no_np, vexpr2_no_np = tree_map(
            lambda x: (x.tolist()
                       if isinstance(x, torch.Tensor)
                       else x),
            (vexpr1, vexpr2))

        self.assertEqual(vexpr1_types, vexpr2_types)
        self.assertEqual(vexpr1_no_np, vexpr2_no_np)


if __name__ == '__main__':
    unittest.main()
