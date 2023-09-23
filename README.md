# Vexpr

Vexpr ("**V**ectorizable **expr**ession", pronounced "Vexper") is a library that represents expressions as `Vexpr` objects, enabling Lisp-macro-like transformations on those objects. The library provides `Vexpr` interfaces for [numpy](https://numpy.org), [pytorch](https://pytorch.org), and [JAX](https://github.com/google/jax).

Built-in `Vexpr` transformations include:

- **Vectorization:** Given a human-written expression, return an equivalent expression that uses parallel, vectorized operations. The primary use case for Vexpr's vectorization is when writing "wide" compositional expressions, i.e. expressions with multiple parallel tree branches with similar operations occurring on those branches. In these scenarios, Vexpr makes it possible to write the code in a readable, compositional way without giving up the performance benefits of vectorized code. Compositionality creates opportunities for library development.
- **Partial evaluation:** Given an expression and some of its inputs, return a new partially evaluated expression. Partial evaluation enables you to write code that is flexible to more use cases. If you have a function `f(a,b)`, some of your users may want to hold `a` constant and try many different `b` values, while other users may want to hold `b` constant. When you express your `f` as a `Vexpr` object, it is automatically optimized to both of these scenarios.


## Example usage

Consider the function:

```python
import numpy as np
from scipy.spatial.distance import cdist

def f(x1, x2, w1, w2):
    """
    Evaluate pairwise distance between two lists of points, using a custom
    weighted distance metric.
    """
    return np.sum([w1 * cdist(x1[..., [0, 1, 2], x2[..., [0, 1, 2]]]),
                   w2 * cdist(x1[..., [0, 3, 4], x2[..., [0, 3, 4]]])],
                  axis=0)
```

This function is written clearly but not efficiently. It indexes into `x1` and `x2` twice. It computes distances twice. It multiplies with weights twice. Let's instead use Vexpr to implement this function so that we can vectorize it.


```python
import vexpr as vp
import vexpr.numpy as vnp
from vexpr.scipy.spatial.distance import cdist as v_cdist

@vp.vectorize
def f(x1, x2, w1, w2):
    return vnp.sum([w1 * v_cdist(x1[..., [0, 1, 2]], x2[..., [0, 1, 2]]),
                    w2 * v_cdist(x1[..., [0, 3, 4]], x2[..., [0, 3, 4]])],
                   axis=0)
```

The decorator immediately parses the function by passing "symbols" into each argument and tracing the function.

```python
print(f.vexpr)
```

```text
numpy.sum(
  [operator.mul(
    symbol('w1'),
    scipy.spatial.distance.cdist(
      operator.getitem(
        symbol('x1'),
        (Ellipsis, [0, 1, 2]),
      ),
      operator.getitem(
        symbol('x2'),
        (Ellipsis, [0, 1, 2]),
      ),
    ),
  ),
   operator.mul(
    symbol('w2'),
    scipy.spatial.distance.cdist(
      operator.getitem(
        symbol('x1'),
        (Ellipsis, [0, 3, 4]),
      ),
      operator.getitem(
        symbol('x2'),
        (Ellipsis, [0, 3, 4]),
      ),
    ),
  )]
  axis=0
)

```

This `Vexpr` data structure is ready to be executed without any further compilation, but the real magic occurs when you call `f` for the first time, triggering compilation into a vectorized `Vexpr`.

```python
example_inputs = dict(
    x1=np.random.randn(10, 5),
    x2=np.random.randn(10, 5),
    w1=np.array(0.7),
    w2=np.array(0.3),
)

f(**example_inputs)  # first call triggers compilation
f(**example_inputs)  # subsequent calls run fast version
print(f.vexpr)
```

```text
numpy.sum(
  operator.mul(
    numpy.reshape(
      numpy.stack([symbol('w1'), symbol('w2')]),
      (2, 1, 1),
    ),
    custom.scipy.cdist_multi(
      operator.getitem(
        symbol('x1'),
        (Ellipsis, array([0, 1, 2, 0, 3, 4])),
      ),
      operator.getitem(
        symbol('x2'),
        (Ellipsis, array([0, 1, 2, 0, 3, 4])),
      ),
      lengths=array([3, 3])
    ),
  )
  axis=0
)
```

This is an equivalent expression with fewer commands. It indexes into `x1` and `x2` once, not twice, computes distance once, and uses a single vectorized multiply. This vectorized expression would have been error-prone to write manually.

Now we perform partial evaluation on the expression.

```python
inference_f = vp.partial_evaluate(f, dict(w1=0.75, w2=0.25))
print(inference_f.vexpr)
```

```text
numpy.sum(
  operator.mul(
    array([[[0.75]],
    
           [[0.25]]]),
    custom.scipy.cdist_multi(
      operator.getitem(
        symbol('x1'),
        (Ellipsis, array([0, 1, 2, 0, 3, 4])),
      ),
      operator.getitem(
        symbol('x2'),
        (Ellipsis, array([0, 1, 2, 0, 3, 4])),
      ),
      lengths=array([3, 3])
    ),
  )
  axis=0
)

```

This is a faster expression because it no longer has to build up a np.array on every execution. Partial evaluation proactively runs every part of the expression that depends only on the partial input. In this expression, you can see this from the fact that the `numpy.reshape` has already occurred on the array.

<!-- TODO insert timeit calls for the original f, f.vexpr, and the original f.vexpr  -->


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

This tree-structured expression is an intuitive user interface. The code takes the shape of how we think about the math being performed. The problem with this approach is that running such a kernel runs each subkernel in sequence, rather than running them together as a single vectorized kernel. Manually writing your own kernel with vectorized code is possible, but then you give up the intuitive interface and replace it with something tedious and error-prone. The purpose of Vexpr is to let you write code like this without giving up performance. You write a tree-structured expression, then you let Vexpr convert it to an equivalent, much faster one.

Vexpr is useful anywhere where you have:

1. Large tree-like expressions
2. ...with similar logic in different branches
3. ...that are evaluated many times.


## Design

Writing Vexpr expressions is like using a programming language that doesn't support variables. Vexpr specifically aims at optimizing trees, which are a subset of DAGs (direct acyclic graphs). There are a couple strategies for bridging the gap to DAGs:

1. Write a custom Vexpr function. It can do whatever it wants inside.
2. Chain two Vexpr expressions together. Run one, then run a second one using its output as an input. The second Vexpr expression is free to use the output of the first expression in as many leaves of the tree as it wants.

Vexpr is designed to work alongside compilers like JAX's XLA compiler or pytorch's `torch.compile`. Vexpr's job is to compile your program down to an efficient set of numpy/pytorch/JAX operations, and then those frameworks' compilers go further.

Vexpr embraces functional programming, which makes it work automatically with `jax.vmap` and `torch.vmap`. Vexpr expressions are functions with no mutable state, and transformations like `expr.vectorize()` or `expr.partial_evaluate()` return new instances.
