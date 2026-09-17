"""Editable baseline: the exact routed AITER MXFP4 expert-chain boundary."""
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def moe_forward(inputs):
    args = dict(inputs)
    args['activation'] = ActivationType(args['activation'])
    args['quant_type'] = QuantType(args['quant_type'])
    if isinstance(args.get('dtype'), str):
        import torch
        args['dtype'] = getattr(torch, args['dtype'])
    return fused_moe(**args)
