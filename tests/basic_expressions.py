import numpy as np
import torch

import vexpr.numpy as vp


def test1():
    a = vp.ArrayMean(vp.SelectFromSymbol("x", [1, 3, 5]))
    b = vp.ArrayMean(vp.SelectFromSymbol("x", [0, 1, 2]))
    c = vp.ArrayMean(vp.SelectFromSymbol("x", [0, 1, 4]))
    d = vp.ArrayMean(vp.SelectFromSymbol("x", [2, 4, 5, 6]))
    e = vp.ArrayMean(vp.SelectFromSymbol("x", [0, 1, 2, 3, 4, 6]))
    w1, w2 = (vp.Symbol(name) for name in ["w1", "w2"])

    expr = w1 * (a + b) ** 2 + w2 * (c + d + e) ** 2
    print(expr)
    result = expr({"x": np.array([0, 1, 2, 3, 4, 5, 6]),
                   "w1": 1.0,
                   "w2": 2.0})
    print(result)

def test1_truth():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    a = np.mean(x[[1, 3, 5]])
    b = np.mean(x[[0, 1, 2]])
    c = np.mean(x[[0, 1, 4]])
    d = np.mean(x[[2, 4, 5, 6]])
    e = np.mean(x[[0, 1, 2, 3, 4, 6]])
    w1, w2 = (1.0, 2.0)

    result = w1 * (a + b) ** 2 + w2 * (c + d + e) ** 2
    print(result)


def test_hstack_f():
    expr = vp.Stack(*[vp.VectorizedProduct(vp.SelectFromSymbol("x", indices))
                      for indices in [[0, 1],
                                      [2, 3],
                                      [4, 5],
                                      [6, 7]]])
    print(expr)

    vectorized = expr.vectorize()
    print(vectorized)

    expected = vp.VectorizedProduct(vp.SelectFromSymbol("x", [0,1,2,3,4,5,6,7]),
                                    at_indices=[0, 0, 1, 1, 2, 2, 3, 3])
    print(expected)
    print(vectorized == expected)


def test_hstack_multiple_f_passthrough():
    expr = vp.Stack(vp.VectorizedProduct(vp.SelectFromSymbol("x", [0, 1])),
                    vp.VectorizedSum(vp.SelectFromSymbol("x", [2, 3])),
                    vp.VectorizedProduct(vp.SelectFromSymbol("x", [4, 5])),
                    vp.VectorizedSum(vp.SelectFromSymbol("x", [6, 7])))
    print(expr)

    vectorized = expr.vectorize()
    print(vectorized)

    expected = vp.VectorizedProduct(
        vp.VectorizedSum(
            vp.SelectFromSymbol("x", [0, 1, 2, 3, 4, 5, 6, 7]),
            [0, 1, 2, 2, 3, 4, 5, 5]
        ),
        [0, 0, 1, 2, 2, 3]
    )
    print(expected)
    print(vectorized == expected)


def test_hstack_multiple_f_no_passthrough():
    expr = vp.Stack(
        vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1]),
                           vp.SelectFromSymbol("x2", [0, 1]))),
        vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3]),
                           vp.SelectFromSymbol("x2", [2, 3]))),
        vp.CDistEuclidean((vp.SelectFromSymbol("x1", [4, 5]),
                           vp.SelectFromSymbol("x2", [4, 5]))),
        vp.CDistCityBlock((vp.SelectFromSymbol("x1", [1, 4]),
                           vp.SelectFromSymbol("x2", [1, 4]))),
    )

    print(expr)

    vectorized = expr.vectorize()
    print(vectorized)

    expected = vp.Shuffle(
        vp.Concatenate(
            vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1, 4, 5]),
                               vp.SelectFromSymbol("x2", [0, 1, 4, 5])),
                              split_lengths=[2, 2]),
            vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3, 1, 4]),
                               vp.SelectFromSymbol("x2", [2, 3, 1, 4])),
                              split_lengths=[2, 2]),
        ),
        [0, 2, 1, 3]
    )

    print(expected)
    print(vectorized == expected)


def test_hstack_product_multiple_f_no_passthrough():
    expr = vp.Stack(
        vp.Product(
            vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1]),
                               vp.SelectFromSymbol("x2", [0, 1]))),
            vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3]),
                               vp.SelectFromSymbol("x2", [2, 3]))),
        ),
        vp.Product(
            vp.CDistEuclidean((vp.SelectFromSymbol("x1", [4, 5]),
                               vp.SelectFromSymbol("x2", [4, 5]))),
            vp.CDistCityBlock((vp.SelectFromSymbol("x1", [1, 4]),
                               vp.SelectFromSymbol("x2", [1, 4]),)),
        )
    )

    print(expr)

    vectorized = expr.vectorize()
    print(vectorized)

    expected = vp.VectorizedProduct(
        vp.Concatenate(
            vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1, 4, 5]),
                               vp.SelectFromSymbol("x2", [0, 1, 4, 5])),
                              split_lengths=[2, 2]),
            vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3, 1, 4]),
                               vp.SelectFromSymbol("x2", [2, 3, 1, 4])),
                              split_lengths=[2, 2]),
        ),
        [0, 1, 0, 1]
    )

    print(expected)
    print(vectorized == expected)



def test_shuffle_multiply():
    w1, w2, w3, w4 = (vp.Symbol(name) for name in ("w1", "w2", "w3", "w4"))
    expr = vp.Stack(
        vp.Product(
            w1 * vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1]),
                                    vp.SelectFromSymbol("x2", [0, 1]))),
            w2 * vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3]),
                                    vp.SelectFromSymbol("x2", [2, 3]))),
        ),
         vp.Product(
             w3 * vp.CDistEuclidean((vp.SelectFromSymbol("x1", [4, 5]),
                                     vp.SelectFromSymbol("x2", [4, 5]))),
             w4 * vp.CDistCityBlock((vp.SelectFromSymbol("x1", [1, 4]),
                                     vp.SelectFromSymbol("x2", [1, 4]),)),
         )
    )

    print(expr)

    vectorized = expr.vectorize()
    print(vectorized)

    expected = vp.VectorizedProduct(
        vp.Multiply((
            vp.Stack(w1, w3, w2, w4),
            vp.Concatenate(
                vp.CDistEuclidean((vp.SelectFromSymbol("x1", [0, 1, 4, 5]),
                                   vp.SelectFromSymbol("x2", [0, 1, 4, 5])),
                                  split_lengths=[2, 2]),
                vp.CDistCityBlock((vp.SelectFromSymbol("x1", [2, 3, 1, 4]),
                                   vp.SelectFromSymbol("x2", [2, 3, 1, 4])),
                                  split_lengths=[2, 2]),
            ),
        )),
        [0, 1, 0, 1]
    )

    print(expected)
    print(vectorized == expected)


def test_kernel():
    expr = vp.Symbol("s") * vp.Product(
        vp.Sum(
            vp.Symbol("w1") * vp.CDistEuclidean(
                vp.Divide((
                    (vp.SelectFromSymbol("x1", [0, 1]),
                     vp.SelectFromSymbol("x2", [0, 1]),),
                    vp.SelectFromSymbol("lengthscale", [0, 1])))),
            vp.Symbol("w2") * vp.CDistCityBlock(
                vp.Divide((
                    (vp.SelectFromSymbol("x1", [2, 3]),
                     vp.SelectFromSymbol("x2", [2, 3]),),
                    vp.SelectFromSymbol("lengthscale", [2, 3]))))
        ),
        vp.Sum(
            vp.Symbol("w3") * vp.CDistEuclidean(
                vp.Divide((
                    (vp.SelectFromSymbol("x1", [4, 5]),
                     vp.SelectFromSymbol("x2", [4, 5]),),
                    vp.SelectFromSymbol("lengthscale", [4, 5])))),
            vp.Symbol("w4") * vp.CDistCityBlock(
                vp.Divide((
                    (vp.SelectFromSymbol("x1", [1, 4]),
                     vp.SelectFromSymbol("x2", [1, 4]),),
                    vp.SelectFromSymbol("lengthscale", [1, 4]))))
        )
    )

    vectorized = expr.vectorize()


if __name__ == "__main__":
    test1()
    test1_truth()
    test_hstack_multiple_f_passthrough()
    test_hstack_multiple_f_no_passthrough()
    test_hstack_product_multiple_f_no_passthrough()
    test_shuffle_multiply()
    test_kernel()
