import unittest

import torch

import vexpr as vp
import vexpr.custom.torch as vctorch
import vexpr.torch as vtorch


class IsolateConstantsTests(unittest.TestCase):
    @unittest.skip("Not yet implemented")
    def test_cdist(self):
        x1_baseline = torch.tensor([[41., 42., 43.],
                                    [42., 43., 44.],
                                    [45., 46., 47.]])
        x2_baseline = torch.tensor([[41., 42., 43.],
                                    [42., 43., 44.],
                                    [45., 46., 47.]])
        x1 = vp.symbol("x1")
        x2 = vp.symbol("x2")

        expr = vtorch.cdist(vtorch.cat([x1_baseline, x1]),
                            vtorch.cat([x2_baseline, x2]))

        refactored = vp.isolate_constants(expr)

        expected = vctorch.matrix_of_three_parts(
            vtorch.cdist(x1_baseline, x2_baseline),
            vtorch.cdist(vtorch.cat([x1_baseline, x1]), x2),
            vtorch.cdist(x1, x2_baseline),
        )

        self.assertEqual(vp.comparable(refactored), vp.comparable(expected))

        precomputed = vp.partial_eval(refactored)

        expected2 = vctorch.matrix_of_three_parts(
            torch.cdist(x1_baseline, x2_baseline),
            vtorch.cdist(vtorch.cat([x1_baseline, x1]), x2),
            vtorch.cdist(x1, x2_baseline),
        )

        self.assertEqual(vp.comparable(precomputed), vp.comparable(expected2))


if __name__ == '__main__':
    unittest.main()
