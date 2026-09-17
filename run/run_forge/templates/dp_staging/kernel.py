"""Offset/size-driven device memcpy kernel, migrated from
``sglang.srt.layers.dp_attention`` (RFC #29630, Phase 2.5).
"""

import functools

import triton
import triton.language as tl


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src: tl.constexpr,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


# Editable local staging anchor. The code above is copied byte-for-byte from
# SGLang's original memcpy_triton.py (see source_identity.json). Both wrappers
# retain the production fill_(0) followed by memcpy_triton baseline. Collective
# execution, residual add, RMSNorm, and quantization are outside this subtask.

def gather(global_tokens, local_tokens, local_start_pos, local_num_tokens):
    """Zero global output, then copy valid local rows at device-scalar offset."""
    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )


def scatter(local_tokens, global_tokens, local_start_pos, local_num_tokens):
    """Zero local output, then copy valid global rows at device-scalar offset."""
    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )
