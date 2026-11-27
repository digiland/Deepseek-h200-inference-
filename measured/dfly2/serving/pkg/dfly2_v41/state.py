"""Process-wide state shared between the patched target attention and the DFly2 drafter.

The target's index-source layers 28/32/36 copy their per-token indexer top-k rows into
``TOPK_BUFFERS[layer]`` (vLLM shares one top-k buffer across all layers, so later index sources
overwrite earlier ones). The drafter reads the row of the last accepted token.
"""

import os

import torch

TOPK_BUFFERS: dict[int, torch.Tensor] = {}


def topk_layers() -> tuple[int, ...]:
    raw = os.environ.get("DFLY2_TOPK_LAYERS", "")
    return tuple(int(x) for x in raw.split(",") if x.strip())


def normed_hidden_for_drafter() -> bool:
    return os.environ.get("DFLY2_NORMED_HIDDEN", "0") == "1"
