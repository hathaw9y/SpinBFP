import torch
from tqdm import tqdm


def convert2fp16(
    x: torch.Tensor,
    block_size: int = 128,
    mbits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    flat = x.reshape(*x.shape[:-1], -1, block_size).half()

    int_bits = flat.view(torch.int16)
    elem_exp = (int_bits >> 10) & 0x1F
    shared_exp = elem_exp.max(dim=-1, keepdim=True).values

    shift = (shared_exp - elem_exp).clamp(min=0, max=10)

    mantissa = (int_bits & 0x03FF) | 0x0400
    mantissa_shifted = mantissa >> shift

    truncate_bits = 11 - mbits + 1
    round_bit = (mantissa_shifted >> (truncate_bits - 1)) & 1
    mantissa_truncated = (mantissa_shifted >> truncate_bits) + round_bit

    max_mantissa = (1 << (mbits - 1)) - 1
    mantissa_truncated = mantissa_truncated.clamp(max=max_mantissa)

    sign = ((int_bits >> 15) & 0x1).half()
    mantissa_signed = mantissa_truncated.to(torch.int16) * (1 - 2 * sign)

    restored = (
        (1 - 2 * sign).half()
        * (mantissa_truncated << truncate_bits).half()
        / 1024.0
        * (2.0 ** (shared_exp - 15)).half()
    )

    real_exp = (2 ** (shared_exp - 15).half()).expand(*shared_exp.shape[:-1], block_size)
    return (
        restored.reshape(shape),
        mantissa_signed.reshape(shape),
        real_exp.reshape(shape),
        shared_exp,
        shift,
        mantissa_truncated,
    )


def bfp_quantize_activation(
    x: torch.Tensor,
    block_size: int = 128,
    mbits: int = 8,
    stat_hook=None,
    stat_name: str | None = None,
) -> torch.Tensor:
    if x.shape[-1] % block_size != 0:
        block_size = x.shape[-1]
    restored, _, _, shared_exp, shift, mantissa_truncated = convert2fp16(
        x,
        block_size=block_size,
        mbits=mbits,
    )
    if stat_hook is not None and stat_name is not None:
        stat_hook.record_bfp_shared_exponent(stat_name, shared_exp)
        stat_hook.record_bfp_shift(stat_name, shift)
        stat_hook.record_bfp_quantization_rates(
            stat_name, mantissa_truncated, shift, mbits,
        )
    return restored.to(x.dtype)


def bfp_quantize_weight_transpose(
    w: torch.Tensor,
    block_size: int = 128,
    mbits: int = 8,
    stat_hook=None,
    stat_name: str | None = None,
) -> torch.Tensor:
    wt = w.T.contiguous()
    if wt.shape[-1] % block_size != 0:
        block_size = wt.shape[-1]
    restored, _, _, shared_exp, shift, mantissa_truncated = convert2fp16(
        wt,
        block_size=block_size,
        mbits=mbits,
    )
    if stat_hook is not None and stat_name is not None:
        stat_hook.record_bfp_shared_exponent(stat_name, shared_exp)
        stat_hook.record_bfp_shift(stat_name, shift)
        stat_hook.record_bfp_quantization_rates(
            stat_name, mantissa_truncated, shift, mbits,
        )
    return restored.T.contiguous().to(w.dtype)


@torch.no_grad()
def eval_ppl_wikitext(model, tokenizer, seq_len=2048, device="cuda"):
    from datasets import load_dataset

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    n_samples = input_ids.shape[1] // seq_len
    nlls = []
    for i in tqdm(range(n_samples)):
        batch = input_ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(batch, labels=batch)
        nlls.append(out.loss.float() * seq_len)
    return torch.exp(torch.stack(nlls).sum() / (n_samples * seq_len)).item()
