# vexpr

vexpr is a library for writing and transforming tree-like expressions, helping you write fast human-readable code. These transformations include:

- **Vectorization**, given a human-written expression, return an equivalent expression that uses parallel, vectorized operations
- **Partial evaluation**, given an expression and some of its inputs, return a new partially evaluated expression

vexpr supports numpy, pytorch, and JAX.

*Example usage:*

```python
import vexpr.numpy as vp

# Equivalent to
# def f(x1, x2, w1, w2):
#     return (w1 * distance(x1[[0, 1, 2], x2[[0, 1, 2]]])
#             + w2 * distance(x1[[0, 3, 4], x2[[0, 3, 4]]]))
expr = vp.Sum([
    vp.Multiply([
        vp.Symbol("w1"),
        vp.Distance(
            [vp.SelectFromSymbol("x1", [0, 1, 2]),
             vp.SelectFromSymbol("x2", [0, 1, 2])]
        )
    ]),
    vp.Multiply([
        vp.Symbol("w2"),
        vp.Distance(
            [vp.SelectFromSymbol("x1", [0, 3, 4]),
             vp.SelectFromSymbol("x2", [0, 3, 4])]
        )
    ]),
])

# Evaluation
print(expr({
    "w1": 0.75,
    "w2": 0.25,
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
# Output:
# 0.656  TODO actually run this

print(expr)
# Output: a data structure showing the original expression
# Sum([
#     Multiply([
#         Symbol("w1"),
#         Distance(
#             [SelectFromSymbol("x1", [0, 1, 2]),
#              SelectFromSymbol("x2", [0, 1, 2])]
#         )
#     ]),
#     Multiply([
#         Symbol("w2"),
#         Distance(
#             [SelectFromSymbol("x1", [0, 3, 4]),
#              SelectFromSymbol("x2", [0, 3, 4])]
#         )
#     ]),
# ])

expr_vectorized = expr.vectorize()

print(expr_vectorized)
# Output: An equivalent expression with fewer commands.
# This expression would have been error-prone to write manually.
# VectorSum([
#     Multiply([
#         Stack([Symbol("w1"), Symbol("w2")]),
#         Distance(
#             [SelectFromSymbol("x1", [0, 1, 2, 0, 3, 4]),
#              SelectFromSymbol("x2", [0, 1, 2, 0, 3, 4])],
#             lengths=[3, 3],
#         )
#     ]),
# ])

# Partial evaluation
inference_expr = expr_vectorized.partial_evaluate({
    "w1": 0.70,
    "w2": 0.30,
})

print(inference_expr)
# Output: A faster expression that no longer has to build up an np.array on every execution.
# VectorSum([
#     Multiply([
#         np.array([0.70, 0.30]),
#         Distance(
#             [SelectFromSymbol("x1", [0, 1, 2, 0, 3, 4]),
#              SelectFromSymbol("x2", [0, 1, 2, 0, 3, 4])],
#             lengths=[3, 3],
#         )
#     ]),
# ])
```

## Installation

```
pip install vexpr
```


## Use cases

vexpr is useful anywhere where you have:

1. Large tree-like expressions
2. ...with similar logic in different branches
3. ...that are evaluated many times.

One area where this often occurs is in kernel methods like Gaussian Processes and Support Vector Machines.
