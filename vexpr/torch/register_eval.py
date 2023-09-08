import torch

from vexpr import core
from . import primitives as p

def shape(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    else:
        return ()

core.shape_impls.update({
    torch.Tensor: shape,
})

core.shape_impls.update({
    dtype: shape
    for dtype in [torch.float16, torch.float32, torch.float64, torch.int8,
                  torch.int16, torch.int32, torch.int64, torch.uint8,
                  torch.bool]
})

def allow_listlike_arg0(torch_func):
    def wrapper(arg0, *args, **kwargs):
        if isinstance(arg0, (list, tuple)):
            if len(arg0) > 1 and isinstance(arg0[0], torch.Tensor):
                return torch_func(torch.stack(arg0), *args, **kwargs)
            else:
                return torch_func(torch.tensor(arg0), *args, **kwargs)
        else:
            return torch_func(arg0, *args, **kwargs)
    return wrapper

core.eval_impls.update({
    p.zeros_p: torch.zeros,
    p.ones_p: torch.ones,
    p.stack_p: torch.stack,
    p.concat_p: torch.concat,
    p.moveaxis_p: torch.moveaxis,
    p.reshape_p: torch.reshape,
    p.exp_p: torch.exp,
    p.log_p: torch.log,
    p.sum_p: torch.sum,
    p.prod_p: allow_listlike_arg0(torch.prod),
    p.cdist_p: torch.cdist,
})
