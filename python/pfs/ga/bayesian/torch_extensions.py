import torch
from torch import (
    contiguous_format,
    Generator,
    inf,
    memory_format,
    strided,
    SymInt,
    Tensor,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
)

def torch_select(
    tensors: tuple[Tensor, ...] | list[Tensor] | None,
    index: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    select(tensors, index, dim=0, out=None) -> Tensor

    Selects values from a list of tensors along an axis specified by `dim` using
    the indices specified in `index`.
    """

    return torch.gather(torch.stack(tensors, dim=-1), dim=-1, index=index.unsqueeze(-1), out=out).squeeze(-1)

torch.select = torch_select