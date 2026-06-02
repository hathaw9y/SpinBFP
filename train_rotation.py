import argparse
import datetime
import os
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import datasets
import torch
import torch.distributed as dist
import transformers
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as PT_FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, Trainer, TrainingArguments, default_data_collator

REFACTOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = REFACTOR_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from bfp_llama.bfp import bfp_quant_dequant
from bfp_llama.config import ExperimentConfig
from bfp_llama.modeling import add_attention_matmul_bfp, rotation_filename
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import hadamard_matrix, random_hadamard_matrix
from utils.rotation_utils import apply_rotation_left, apply_rotation_right
from utils.utils import get_global_rank, get_local_rank, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train BFP rotations for LLaMA models.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--w-bfp-group-size", type=int, default=None)
    parser.add_argument("--a-bfp-group-size", type=int, default=None)
    parser.add_argument("--kv-bfp-group-size", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--rotation-compute-dtype", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--online-had-group-size", type=int, default=32)
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument("--qk-matmul-bits", type=int, default=None)
    parser.add_argument("--av-matmul-bits", type=int, default=None)
    parser.add_argument("--qk-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--av-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--rotation-block-size", type=int, default=0)
    parser.add_argument("--rotation-init", choices=["random_hadamard", "hadamard"], default="random_hadamard")
    parser.add_argument("--fp32-had", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default=None)
    return parser.parse_args()


class RotateModule(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.weight = nn.Parameter(matrix.to(dtype=torch.float32, device=torch.cuda.current_device()))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        return self.weight @ x


class BlockDiagRotateModule(nn.Module):
    def __init__(self, size, block_size, init_fn):
        super().__init__()
        if block_size <= 0 or size % block_size != 0:
            raise ValueError(f"rotation size {size} must be divisible by block size {block_size}")
        self.blocks = nn.ParameterList(
            [
                nn.Parameter(
                    init_fn(block_size, "cuda").to(
                        dtype=torch.float32,
                        device=torch.cuda.current_device(),
                    )
                )
                for _ in range(size // block_size)
            ]
        )

    @property
    def weight(self):
        return torch.stack(tuple(self.blocks), dim=0)


def rotation_init_fn(rotation_init):
    if rotation_init == "hadamard":
        return hadamard_matrix
    if rotation_init == "random_hadamard":
        return random_hadamard_matrix
    raise ValueError(f"Unsupported rotation init: {rotation_init}")


def make_rotation_module(size, block_size, rotation_init="random_hadamard"):
    init_fn = rotation_init_fn(rotation_init)
    if block_size and block_size > 0:
        return BlockDiagRotateModule(size, block_size, init_fn)
    return RotateModule(init_fn(size, "cuda"))


def rotation_module_params(*modules):
    params = []
    for module in modules:
        params.extend(module.parameters())
    return params


class OptRotationLinear(nn.Module):
    def __init__(self, linear, role, cfg, model, compute_dtype, r2_module=None):
        super().__init__()
        self.linear = linear
        self.role = role
        self.cfg = cfg
        object.__setattr__(self, "_model_ref", weakref.ref(model))
        self.compute_dtype = compute_dtype
        object.__setattr__(self, "_r2_ref", weakref.ref(r2_module) if r2_module is not None else None)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _r1(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("OPT parent model reference is gone")
        return model.R1.weight

    def _r2(self):
        r2_ref = getattr(self, "_r2_ref", None)
        r2_module = r2_ref() if r2_ref is not None else None
        return None if r2_module is None else r2_module.weight

    def _apply_r2_to_weight(self, weight, transpose=False):
        r2 = self._r2()
        if r2 is None:
            return weight
        had_dim = r2.shape[0] * r2.shape[-1] if r2.dim() == 3 else r2.shape[0]
        dtype = weight.dtype
        compute_dtype = self.compute_dtype
        if transpose:
            shape = weight.shape
            temp = weight.reshape(-1, shape[-1] // had_dim, had_dim)
            temp = apply_rotation_right(temp, r2, compute_dtype)
            return temp.reshape(shape).to(dtype)
        wt = weight.t()
        shape = wt.shape
        temp = wt.reshape(-1, shape[-1] // had_dim, had_dim)
        temp = apply_rotation_right(temp, r2, compute_dtype)
        return temp.reshape(shape).t().to(dtype)

    def _effective_weight(self):
        weight = self.linear.weight
        r1 = self._r1()
        dtype = weight.dtype
        compute_dtype = self.compute_dtype
        if self.role in ["q_proj", "k_proj", "v_proj", "fc1"]:
            weight = apply_rotation_right(weight, r1, compute_dtype)
        elif self.role in ["out_proj", "fc2"]:
            weight = apply_rotation_left(weight, r1, compute_dtype, transpose=True)
        if self.role == "v_proj":
            weight = self._apply_r2_to_weight(weight, transpose=False)
        elif self.role == "out_proj":
            weight = self._apply_r2_to_weight(weight, transpose=True)
        return weight

    def forward(self, x):
        x_dtype = x.dtype
        if self.role == "lm_head":
            r1 = self._r1()
            x = apply_rotation_right(x, r1, self.compute_dtype, transpose=True)
            return self.linear(x).to(x_dtype)

        x = bfp_quant_dequant(x, self.cfg.a_bits, self.cfg.a_bfp_group_size)
        weight = self._effective_weight()
        weight = bfp_quant_dequant(weight, self.cfg.w_bits, self.cfg.w_bfp_group_size)
        out = nn.functional.linear(x, weight, self.linear.bias).to(x_dtype)
        if self.role == "v_proj":
            out = bfp_quant_dequant(out, self.cfg.kv_bits, self.cfg.kv_bfp_group_size)
        return out


def _replace_opt_linears(model, cfg, compute_dtype):
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        r2 = attn.R2
        attn.q_proj = OptRotationLinear(attn.q_proj, "q_proj", cfg, model, compute_dtype)
        attn.k_proj = OptRotationLinear(attn.k_proj, "k_proj", cfg, model, compute_dtype)
        attn.v_proj = OptRotationLinear(attn.v_proj, "v_proj", cfg, model, compute_dtype, r2_module=r2)
        attn.out_proj = OptRotationLinear(attn.out_proj, "out_proj", cfg, model, compute_dtype, r2_module=r2)
        layer.fc1 = OptRotationLinear(layer.fc1, "fc1", cfg, model, compute_dtype)
        layer.fc2 = OptRotationLinear(layer.fc2, "fc2", cfg, model, compute_dtype)
    model.lm_head = OptRotationLinear(model.lm_head, "lm_head", cfg, model, compute_dtype)


def setup_opt_rotation_model(model, cfg, compute_dtype):
    for param in model.parameters():
        param.requires_grad = False
    model.R1 = make_rotation_module(model.config.hidden_size, cfg.rotation_block_size, cfg.rotation_init)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for layer in model.model.decoder.layers:
        layer.self_attn.R2 = make_rotation_module(head_dim, cfg.rotation_block_size, cfg.rotation_init)
    _replace_opt_linears(model, cfg, compute_dtype)

    if hasattr(model, "_bfp_opt_input_rotation_hook"):
        return model

    def rotate_input(module, args, kwargs):
        if not hasattr(model, "R1"):
            return args, kwargs

        def rotate(x):
            return apply_rotation_right(x, model.R1.weight, compute_dtype)

        if len(args) > 0:
            return (rotate(args[0]),) + args[1:], kwargs
        kwargs["hidden_states"] = rotate(kwargs["hidden_states"])
        return args, kwargs

    model._bfp_opt_input_rotation_hook = model.model.decoder.layers[0].register_forward_pre_hook(
        rotate_input, with_kwargs=True
    )
    return model


def save_custom_opt_rotations(model, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {"R1": model.R1.weight.detach().cpu()}
    for idx, layer in enumerate(model.model.decoder.layers):
        state[f"model.decoder.layers.{idx}.self_attn.R2"] = (
            layer.self_attn.R2.weight.detach().cpu()
        )
    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def resolve_dtype(dtype_arg, hf_config):
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16

    config_dtype = getattr(hf_config, "torch_dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = config_dtype.replace("torch.", "")
        if config_dtype in ["bfloat16", "bf16"]:
            return torch.bfloat16
        if config_dtype in ["float16", "fp16", "half"]:
            return torch.float16
    if config_dtype == torch.bfloat16:
        return torch.bfloat16
    if config_dtype == torch.float16:
        return torch.float16
    return torch.float16


def resolve_rotation_compute_dtype(dtype_arg):
    if dtype_arg == "fp32":
        return torch.float32
    return torch.float64


def set_rotation_compute_dtype(model, compute_dtype):
    model.rotation_compute_dtype = compute_dtype
    if hasattr(model, "model"):
        model.model.rotation_compute_dtype = compute_dtype
    for module in model.modules():
        if hasattr(module, "rotation_compute_dtype") or module.__class__.__name__ == "QuantizeLinear":
            module.rotation_compute_dtype = compute_dtype


def build_bfp_args(args, cfg):
    return SimpleNamespace(
        seed=args.seed,
        fp32_had=args.fp32_had,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        k_bits=args.kv_bits,
        v_bits=args.kv_bits,
        w_groupsize=-1,
        a_groupsize=-1,
        k_groupsize=-1,
        v_groupsize=-1,
        w_asym=False,
        a_asym=False,
        k_asym=False,
        v_asym=False,
        w_clip=False,
        a_clip_ratio=1.0,
        k_clip_ratio=1.0,
        v_clip_ratio=1.0,
        w_quant_method="bfp",
        a_quant_method="bfp",
        k_quant_method="bfp",
        v_quant_method="bfp",
        w_bfp_groupsize=cfg.w_bfp_group_size,
        a_bfp_groupsize=cfg.a_bfp_group_size,
        k_bfp_groupsize=cfg.kv_bfp_group_size,
        v_bfp_groupsize=cfg.kv_bfp_group_size,
        int8_down_proj=False,
        k_pre_rope=False,
        online_had_groupsize=cfg.online_had_group_size,
        w_down_had_groupsize=cfg.w_down_had_group_size,
        qk_had_groupsize=cfg.qk_had_group_size,
        qk_matmul_bits=cfg.qk_matmul_bits,
        qk_matmul_bfp_groupsize=cfg.qk_matmul_bfp_group_size,
        qk_matmul_clip_ratio=1.0,
        av_matmul_bits=cfg.av_matmul_bits,
        av_matmul_bfp_groupsize=cfg.av_matmul_bfp_group_size,
        av_matmul_clip_ratio=1.0,
    )


def save_custom_rotations(model, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {"R1": model.R1.weight.detach().cpu()}
    for idx, layer in enumerate(model.model.layers):
        state[f"model.layers.{idx}.self_attn.R2"] = (
            layer.self_attn.R2.weight.detach().cpu()
        )
    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def save_rotations_from_state_dict(state_dict, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key.replace(".weight", ""): value
        for key, value in state_dict.items()
        if "R1.weight" in key or "self_attn.R2" in key
    }
    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def unwrap_fsdp(module):
    return module.module if isinstance(module, PT_FSDP) else module


def fsdp_full_params_context(module):
    if isinstance(module, PT_FSDP):
        return PT_FSDP.summon_full_params(
            module,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        )
    return nullcontext()


def save_custom_rotations_fsdp(model, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = get_global_rank()
    state = {}

    with fsdp_full_params_context(model):
        if rank == 0:
            base_model = unwrap_fsdp(model)
            state["R1"] = base_model.R1.weight.detach().cpu().clone()

    base_model = unwrap_fsdp(model)
    for idx, layer in enumerate(base_model.model.layers):
        with fsdp_full_params_context(layer):
            if rank == 0:
                layer_module = unwrap_fsdp(layer)
                state[f"model.layers.{idx}.self_attn.R2"] = (
                    layer_module.self_attn.R2.weight.detach().cpu().clone()
                )

    if rank != 0:
        return None

    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def init_distributed_if_needed():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))


def main():
    args = parse_args()
    init_distributed_if_needed()
    local_rank = get_local_rank() if dist.is_initialized() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        model=args.model,
        max_length=args.max_length,
        seed=args.seed,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        kv_bits=args.kv_bits,
        bfp_group_size=args.bfp_group_size,
        w_bfp_group_size=args.w_bfp_group_size or args.bfp_group_size,
        a_bfp_group_size=args.a_bfp_group_size or args.bfp_group_size,
        kv_bfp_group_size=args.kv_bfp_group_size or args.bfp_group_size,
        online_had_group_size=args.online_had_group_size,
        w_down_had_group_size=args.w_down_had_group_size,
        qk_had_group_size=args.qk_had_group_size,
        qk_matmul_bits=args.qk_matmul_bits or args.kv_bits,
        av_matmul_bits=args.av_matmul_bits or args.kv_bits,
        qk_matmul_bfp_group_size=args.qk_matmul_bfp_group_size,
        av_matmul_bfp_group_size=args.av_matmul_bfp_group_size,
        rotation_block_size=args.rotation_block_size,
        rotation_init=args.rotation_init,
        rotate=True,
        fp32_had=args.fp32_had,
    )

    hf_config = transformers.AutoConfig.from_pretrained(args.model, token=args.access_token)
    dtype = resolve_dtype(args.dtype, hf_config)
    if local_rank == 0:
        print(f"Using compute dtype: {dtype}")
    rotation_compute_dtype = resolve_rotation_compute_dtype(args.rotation_compute_dtype)
    if local_rank == 0:
        print(f"Using rotation compute dtype: {rotation_compute_dtype}")
        print(f"Using rotation block size: {cfg.rotation_block_size or 'full'}")
        print(f"Using rotation init: {cfg.rotation_init}")

    if hf_config.model_type == "llama":
        if cfg.qk_matmul_bits < 16 or cfg.av_matmul_bits < 16:
            hf_config._attn_implementation = "eager"

        clone_lm_head = False
        if getattr(hf_config, "tie_word_embeddings", False):
            hf_config.tie_word_embeddings = False
            clone_lm_head = True

        model = LlamaForCausalLMQuant.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=hf_config,
            torch_dtype=dtype,
            token=args.access_token,
        )
        if clone_lm_head:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

        model = prepare_model(build_bfp_args(args, cfg), model)
        set_rotation_compute_dtype(model, rotation_compute_dtype)
        add_attention_matmul_bfp(model, cfg)
        for param in model.parameters():
            param.requires_grad = False
        model.R1 = make_rotation_module(
            model.config.hidden_size,
            cfg.rotation_block_size,
            cfg.rotation_init,
        )
        for i in range(model.config.num_hidden_layers):
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            model.model.layers[i].self_attn.R2 = make_rotation_module(
                head_dim,
                cfg.rotation_block_size,
                cfg.rotation_init,
            )
        rotation_params = rotation_module_params(model.R1) + [
            param
            for i in range(model.config.num_hidden_layers)
            for param in model.model.layers[i].self_attn.R2.parameters()
        ]
        tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=args.model,
            model_max_length=cfg.max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=args.access_token,
        )
    elif hf_config.model_type == "opt":
        if args.fsdp != "" and args.fsdp != []:
            raise ValueError("OPT rotation training is supported only with train_rotation.sh (non-FSDP) for now.")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=hf_config,
            torch_dtype=dtype,
            token=args.access_token,
        )
        model = setup_opt_rotation_model(model, cfg, rotation_compute_dtype)
        rotation_params = rotation_module_params(model.R1) + [
            param
            for layer in model.model.decoder.layers
            for param in layer.self_attn.R2.parameters()
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model,
            model_max_length=cfg.max_length,
            padding_side="right",
            use_fast=True,
            token=args.access_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"bfp_refactor supports LLaMA and OPT configs, got {hf_config.model_type}")

    model.config.use_cache = False
    use_gradient_checkpointing = args.gradient_checkpointing
    if hf_config.model_type == "opt" and use_gradient_checkpointing:
        use_gradient_checkpointing = False
        if local_rank == 0:
            print("Disabling gradient checkpointing for OPT rotation training because frozen inputs can hide rotation gradients from reentrant checkpointing.")

    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    dataset = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(cfg.max_length, 2048),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        logging_dir=str(output_dir / "logs"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        gradient_checkpointing=use_gradient_checkpointing,
        save_safetensors=False,
        save_strategy="no",
        logging_steps=1,
        log_on_each_node=False,
        report_to=[],
        fsdp=args.fsdp,
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
        ddp_find_unused_parameters=False,
    )

    optimizer = SGDG(rotation_params, lr=args.learning_rate, stiefel=True)
    trainer_cls = FSDPTrainer if training_args.fsdp != "" and training_args.fsdp != [] else Trainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    trainer.train()

    if training_args.fsdp != "" and training_args.fsdp != []:
        path = save_custom_rotations_fsdp(trainer.model, output_dir, cfg)
        if local_rank == 0:
            print(f"Saved rotations to {path}")
    else:
        if local_rank == 0:
            if hf_config.model_type == "opt":
                path = save_custom_opt_rotations(model, output_dir, cfg)
            else:
                path = save_custom_rotations(model, output_dir, cfg)
            print(f"Saved rotations to {path}")
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
