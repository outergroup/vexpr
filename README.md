# Vexpr

Vexpr (pronounced "vexper") is a library for writing and transforming tree-like expressions, helping you write fast human-readable code. These transformations include:

- **Vectorization:** given a human-written expression, return an equivalent expression that uses parallel, vectorized operations
- **Partial evaluation:** given an expression and some of its inputs, return a new partially evaluated expression

Vexpr essentially contains three different libraries:

- vexpr.numpy
- vexpr.torch
- vexpr.jax

Each of these have same core Vexpr design, but each aim to feel native to each respective vector library ([numpy](https://numpy.org) / [pytorch](https://pytorch.org) / [JAX](https://github.com/google/jax)).


## Example usage

Implement a function by defining a Vexpr expression.

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
```

Output: a data structure representing the expression.
```text
Sum(
  Multiply(
    [Symbol(name='w1'),
     Distance(
      [SelectFromSymbol(name='x1', indices=[0, 1, 2]),
       SelectFromSymbol(name='x2', indices=[0, 1, 2])],
    )],
  ),
  Multiply(
    [Symbol(name='w2'),
     Distance(
      [SelectFromSymbol(name='x1', indices=[0, 3, 4]),
       SelectFromSymbol(name='x2', indices=[0, 3, 4])],
    )],
  ),
)
```

Evaluate the expression.

```python
print(expr({
    "w1": 0.75,
    "w2": 0.25,
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))

```

```text
0.17320508075688773
```

Transform the expression into a vectorized version.

```python
expr_vectorized = expr.vectorize()

print(expr_vectorized)

print(expr_vectorized({
    "w1": 0.75,
    "w2": 0.25,
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
```

```text
VectorizedSum(
  Multiply(
    (Stack(
      Symbol(name='w1'),
      Symbol(name='w2'),
    ),
     Distance(
      (SelectFromSymbol(name='x1', indices=[0, 1, 2, 0, 3, 4]),
       SelectFromSymbol(name='x2', indices=[0, 1, 2, 0, 3, 4])),
      split_indices=[3]
    )),
  ),
)

0.17320508075688773
```

This is an equivalent expression with fewer commands. This vectorized expression would have been error-prone to write manually.

Perform partial evaluation, transforming the expression again.

```python
inference_expr = expr_vectorized.partial_evaluate({
    "w1": 0.70,
    "w2": 0.30,
})

print(inference_expr)

print(inference_expr({
    "x1": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    "x2": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
}))
```

```text
VectorizedSum(
  Multiply(
    (array([0.75, 0.25]),
     Distance(
      (SelectFromSymbol(name='x1', indices=[0, 1, 2, 0, 3, 4]),
       SelectFromSymbol(name='x2', indices=[0, 1, 2, 0, 3, 4])),
      split_indices=[3]
    )),
  ),
)

0.17320508075688773
```

This is a faster expression because it no longer has to build up a np.array on every execution. Partial evaluation proactively runs every part of the expression that depends only on the partial input.

## Installation

Until official release, it is better to clone the repo and run:

```
python setup.py develop
```

But an outdated version is available via `pip install`:

```
pip install vexpr
```

## Motivation

Vexpr was originally designed to enable fast [gpytorch](https://gpytorch.ai) compositional kernels. In gpytorch, compositional kernels are built up using expressions like:

```python
ProductKernel(
    AdditiveKernel(
        Kernel1(), Kernel2(), Kernel3()
    ),
    Kernel4(),
    AdditiveKernel(
        Kernel5(), Kernel6()
    ),
)
```

This tree-structured expression is an intuitive user interface. The code takes the shape of how we think about the math being performed. The problem with this approach is that running such a kernel runs each subkernel in sequence, rather than running them together as a single vectorized kernel. Writing vectorized code is possible, but then you lose the intuitive interface and instead replace it with something tedious and error-prone. The purpose of Vexpr is to let you write code like this without giving up performance. You write a tree-structured expression, then you let Vexpr convert it to an equivalent, much faster one.

Vexpr is useful anywhere where you have:

1. Large tree-like expressions
2. ...with similar logic in different branches
3. ...that are evaluated many times.


## Design

Writing Vexpr expressions is like using a programming language that doesn't support variables. Vexpr specifically aims at optimizing trees, which are a subset of DAGs (direct acyclic graphs). There are a few strategies for bridging the gap to DAGs:

1. Use an expressive technique built into Vexpr: [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) as functions arguments. For example, the following expression computes `lengthscale` once and then uses it divide both `x1` and `x2`.

```python
    vp.Divide((
        (Symbol("x1"), Symbol("x2")),
        Symbol("lengthscale")
    ))
```

2. Write a custom Vexpr function. It can do whatever it wants inside.
3. Chain two Vexpr expressions together. Run one, then run a second one using its output as an input. The second Vexpr expression is free to use the output of the first expression in as many leaves of the tree as it wants.

Vexpr is designed to work alongside compilers like JAX's XLA compiler or pytorch's `torch.compile`. Vexpr's job is to compile your program down to an efficient set of numpy/pytorch/JAX operations, and then those frameworks' compilers go further.

Vexpr embraces functional programming, which makes it work automatically with `jax.vmap` and `torch.vmap`. Vexpr expression are functions with no mutable state, and transformations like `expr.vectorize()` or `expr.partial_evaluate()` return new instances.


### Calling conventions

Some Vexpr functions like `Sum` use calling convention `f(arg1, arg2)` while others like `Multiply` use the convention of taking a [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) of args, for example `f((arg1, arg2))`. This convention indicates what happens to those arguments during vectorization. During vectorization, multiple arguments are vectorized into a single argument, whereas pytrees are always preserved. So

```
vectorize(f(arg1, arg2), f(arg3, arg4))
```

might give you

```
vectorized_f(np.array([arg1, arg2, arg3, arg4]),
             sizes=[2, 2])
```

while

```
vectorize(f((arg1, arg2)), f((arg3, arg4)))
```

might give you

```
f((np.array([arg1, arg3]),
   np.array([arg2, arg4])))
```

On top of this, in either calling convention the args themselves may be pytrees. For example,

```
vectorize(Sum({"a": 42, "b": 43}), Sum{{"a": 2, "b": 3}))
```

would give you

```
VectorizedSum({
    "a": np.array([42, 2]),
    "b": np.array([43, 3])
})
 ```

again following the rule that vectorization preserves pytree structure.
