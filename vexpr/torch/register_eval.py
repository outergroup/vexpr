import torch

import vexpr.vectorization as v
from vexpr import core
from . import primitives as p

def shape(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    else:
        return ()

v.shape_impls.update({
    torch.Tensor: shape,
})

v.shape_impls.update({
    dtype: shape
    for dtype in [torch.float16, torch.float32, torch.float64, torch.int8,
                  torch.int16, torch.int32, torch.int64, torch.uint8,
                  torch.bool]
})

def allow_listlike_arg0(torch_func):
    def wrapper(arg0, *args, **kwargs):
        with torch.profiler.record_function("listlike_of_tensors"):
            if isinstance(arg0, (list, tuple)):
                if len(arg0) > 1 and isinstance(arg0[0], torch.Tensor):
                    return torch_func(torch.stack(arg0), *args, **kwargs)
                else:
                    return torch_func(torch.tensor(arg0), *args, **kwargs)
            else:
                return torch_func(arg0, *args, **kwargs)
    return wrapper

def listlike_of_tensors(torch_func):
    def wrapper(arg0, *args, **kwargs):
        with torch.profiler.record_function("listlike_of_tensors"):
            if isinstance(arg0, (list, tuple)):
                return torch_func([torch.as_tensor(x)
                                   for x in arg0], *args, **kwargs)
            else:
                return torch_func(arg0, *args, **kwargs)
    return wrapper

core.eval_impls.update({
    p.zeros_p: torch.zeros,
    p.ones_p: torch.ones,
    p.stack_p: listlike_of_tensors(torch.stack),
    p.cat_p: torch.cat,
    p.moveaxis_p: torch.moveaxis,
    p.reshape_p: torch.reshape,
    p.exp_p: torch.exp,
    p.log_p: torch.log,
    p.sum_p: allow_listlike_arg0(torch.sum),
    p.prod_p: allow_listlike_arg0(torch.prod),
    p.cdist_p: torch.cdist,
    p.scatter_p: torch.scatter,
})


def index_add_impl(a, dim, index, source, *args, **kwargs):
    return a.index_add(dim, index, source, *args, **kwargs)

def index_reduce_impl(a, dim, index, source, *args, **kwargs):
    return a.index_reduce(dim, index, source, *args, **kwargs)

core.eval_impls.update({
    p.index_add_p: index_add_impl,
    p.index_reduce_p: index_reduce_impl,
})
