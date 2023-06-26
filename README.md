# vexpr

vexpr is a library for writing and transforming tree-like expressions, helping you write fast human-readable code. These transformations include:

- **Vectorization**, given a human-written expression, return an equivalent expression that uses parallel, vectorized operations
- **Partial evaluation**, given an expression and some of its inputs, return a new partially evaluated expression

vexpr supports numpy, pytorch, and JAX. `vexpr.numpy`, `vexpr.torch`, and `vexpr.jax` are essentially three different libraries with the same core vexpr design but designed for the respective vector library.

*Example usage:*

```python
import vexpr.numpy as vp

# Equivalent to
# def f(x1, x2, w1, w2):
#     return (w1 * distance(x1[[0, 1, 2], x2[[0, 1, 2]]])
#             + w2 * distance(x1[[0, 3, 4], x2[[0, 3, 4]]]))
expr = vp.Sum(
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
)

print(expr)
# Output: a data structure showing the original expression
# Sum(
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
# )

# Evaluation
print(expr({
    "w1": 0.75,
    "w2": 0.25,
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
# Output:
# 0.17320508075688773

expr_vectorized = expr.vectorize()

print(expr_vectorized)
# Output: An equivalent expression with fewer commands.
# This expression would have been error-prone to write manually.
# VectorSum(
#     Multiply([
#         Stack([Symbol("w1"), Symbol("w2")]),
#         Distance(
#             [SelectFromSymbol("x1", [0, 1, 2, 0, 3, 4]),
#              SelectFromSymbol("x2", [0, 1, 2, 0, 3, 4])],
#             lengths=[3, 3],
#         )
#     ]),
# )

# Evaluation
print(expr_vectorized({
    "w1": 0.75,
    "w2": 0.25,
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
# Output:
# 0.17320508075688773

# Partial evaluation
inference_expr = expr_vectorized.partial_evaluate({
    "w1": 0.70,
    "w2": 0.30,
})

print(inference_expr)
# Output: A faster expression that no longer has to build up an np.array on every execution.
# VectorSum(
#     Multiply([
#         np.array([0.70, 0.30]),
#         Distance(
#             [SelectFromSymbol("x1", [0, 1, 2, 0, 3, 4]),
#              SelectFromSymbol("x2", [0, 1, 2, 0, 3, 4])],
#             lengths=[3, 3],
#         )
#     ]),
# )

# Evaluation
print(inference_expr({
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
# Output:
# 0.17320508075688773
```

## Installation

Until official release, it is better to clone the repo and run:

```
python setup.py develop
```

But an outdated version is available via `pip install`:

```
pip install vexpr
```


## Use cases

vexpr is useful anywhere where you have:

1. Large tree-like expressions
2. ...with similar logic in different branches
3. ...that are evaluated many times.

One area where this often occurs is in kernel methods like Gaussian Processes and Support Vector Machines.

## Design

vexpr was originally designed to enable fast [gpytorch](https://gpytorch.ai) compositional kernels. In gpytorch, compositional kernels are built up using expressions like `ProductKernel(AdditiveKernel(Kernel1(), Kernel2(), Kernel3()), Kernel4())`. This tree-structured expression is an intuitive, non-error-prone user interface. The problem with this approach is that running such a kernel runs each subkernel in sequence, rather than running them together as a single vectorized kernel. The purpose of vexpr is to let you write code like this without giving up performance. You write a tree-structured expression, then you let vexpr optimize that it an equivalent, much faster one.

Writing a vexpr expression is like using a programming language that doesn't support variables. Every vexpr is a tree. The moment you introduce variables (a.k.a. `let` binding), you switch from executing a tree to executing a direct acyclic graph (DAG). vexpr specifically aims at optimizing trees, which are a subset of DAGs. To handle DAGs with vexpr, choose from the following strategies:

1. Use [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) as functions arguments. For example, `vp.Divide(((Symbol("x1"), Symbol("x2")), Symbol("lengthscale")))` computes `lengthscale` once and then uses it divide both `x1` and `x2`. vexpr thoroughly embraces pytrees as function arguments.
2. Write a custom vexpr function that runs a DAG internally.
3. Chain two vexpr expressions together. Run one, then run a second one using its output as an input. The second vexpr expression is free to use the output of the first expression in as many leaves of the tree as it wants.

vexpr is designed to work alongside compilers like JAX's XLA compiler or pytorch's `torch.compile`. vexpr's job is to compile your program down to an efficient set of numpy/pytorch/JAX operations, and then those frameworks' compilers go further.

vexpr embraces functional programming, which makes it work automatically with `jax.vmap` and `torch.vmap`. vexpr expression are functions with no mutable state, and transformations like `expr.vectorize()` or `expr.partial_evaluate()` return new instances.


## Conventions

Some vexpr functions like `Sum` use convention `f(arg1, arg2)` while others like `Multiply` use the convention of taking a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) of args, for example `f((arg1, arg2))`. This convention indicates what happens to those arguments during vectorization. During vectorization, multiple arguments are vectorized into a single argument, whereas pytrees are always preserved. So vectorizing `f(arg1, arg2)` and `f(arg3, arg4)` might give you `vectorized_f(np.array([arg1, arg2, arg3, arg4]), sizes=[2, 2])`, while vectorizing `f((arg1, arg2))` and `f((arg3, arg4))` might give you `f((np.array([arg1, arg3]), np.array([arg2, arg4])))`. On top of this, in either calling convention the args themselves may be pytrees. For example, `Sum({"a": 42, "b": 43}, {"a": 2, "b": 3})` would be vectorized to `VectorSum({"a": np.array([42, 2]), "b": np.array([43, 3])})`, again following the rule that vectorization preserves pytree structure.
