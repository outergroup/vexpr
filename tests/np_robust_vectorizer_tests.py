import unittest

import numpy as np
from jax.tree_util import tree_map

import vexpr as vp
import vexpr.numpy as vnp
import vexpr.custom.numpy as vcnp


class TestVexprCore(unittest.TestCase):
    @unittest.skip("Not yet supported")
    def test_choose_good_sum_prod_order(self):
        example_inputs = dict(a=1, b=2, c=3, d=4, e=5, f=6)

        @vp.vectorize
        def f(a, b, c, d, e, f):
            return vnp.stack([vnp.prod([a, b]),
                              vnp.sum([vnp.prod([c, d]),
                                       vnp.prod([e, f])])])

        @vp.make_vexpr
        def expected(a, b, c, d, e, f):
            return vnp.add_at(
                vnp.zeros((2,)),
                np.array([0, 1, 1]),
                vnp.prod_at(
                    vnp.ones((3,)),
                    np.array([0, 0, 1, 1, 2, 2]),
                    vnp.stack([a, b, c, d, e, f])
                ))

        self._vectorize_test(example_inputs, f, expected)

    @unittest.skip("Not yet supported")
    def test_unary_partial(self):
        example_inputs = dict(a=1, b=2, c=3, d=4)

        @vp.vectorize
        def f(a, b, c, d):
            return vnp.prod([-vnp.sum([a, b]),
                             vnp.sum([c, d])])

        @vp.make_vexpr
        def expected(a, b, c, d):
            return vnp.prod(vcnp.shuffle(vnp.concatenate([-vnp.stack([a, b]),
                                                          vnp.stack([c, d])]),
                                       np.array([0, 1])))

        self._vectorize_test(example_inputs, f, expected)


    def _vectorize_test(self, example_inputs, f, expected_after):
        before_result = f(**example_inputs)
        after = f.vexpr

        # Equality checks are a pain when there might be numpy arrays in the
        # objects. Test the types and values separately.
        after_types, expected_after_types = tree_map(
            lambda x: (x.dtype
                       if isinstance(x, np.ndarray)
                       else x),
            (after, expected_after))
        after_no_np, expected_after_no_np = tree_map(
            lambda x: (x.tolist()
                       if isinstance(x, np.ndarray)
                       else x),
            (after, expected_after))

        self.assertEqual(after_types, expected_after_types)
        self.assertEqual(after_no_np, expected_after_no_np)

        after_result = f(**example_inputs)
        np.testing.assert_equal(before_result, after_result)


if __name__ == '__main__':
    unittest.main()
