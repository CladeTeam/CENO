#!/usr/bin/env python3
# Copyright (c) 2025-2026, CENO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Patch vLLM for DNA CENO inference.

This script does three things:
  1. Patches shared model code (ceno_tokenizer, tokenizer_config) — once
  2. Prepares a checkpoint directory (symlinks shared files, ensures auto_map)
  3. Patches vLLM source code in the container

Current attention behavior:
  - Keep vLLM's native contiguous q/k/v split for correctly reconverted HF
    checkpoints.
  - Do NOT reintroduce the older interleaved attention forward patch, which
    only compensated for legacy HF exports with incorrectly split q/k/v.

Usage:
    python3 generation/patch_vllm_for_dna.py <checkpoint_dir>

Example:
    python3 generation/patch_vllm_for_dna.py /path/to/ceno_checkpoint
"""

import json
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────
# Shared model code (tokenizer, config class, modeling).
# Defaults to this repo's own ceno_model/ceno_hf package; override with
# the CENO_SHARED_CODE_DIR env var if you keep the model code elsewhere.
# ─────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED_CODE_DIR = os.environ.get(
    "CENO_SHARED_CODE_DIR",
    os.path.join(_REPO_ROOT, "ceno_model", "ceno_hf"),
)
SUPPORTED_VLLM_VERSION = "0.8.5.post1"

# Files to synchronize from SHARED_CODE_DIR into each checkpoint.  The copy is
# deliberate: a published checkpoint must remain standalone-loadable after the
# local repository is removed.
SHARED_FILES = [
    "configuration_ceno.py",
    "modeling_ceno.py",
    "ceno_tokenizer.py",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "generation_config.json",
]


# ═════════════════════════════════════════════
# STEP 1: Patch shared code (run once, idempotent)
# ═════════════════════════════════════════════
def patch_shared_code():
    print(f"[1/3] Patching shared code in {SHARED_CODE_DIR} ...")

    # All shipped source files already use native Python token IDs.  Do not
    # mutate shared source at inference time; checkpoint preparation below
    # synchronizes the checked-in implementation instead.
    print("  OK: shared tokenizer is release-ready")

    # ── Fix tokenizer_config.json: AutoTokenizer format ──
    tc_path = os.path.join(SHARED_CODE_DIR, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path, "r") as f:
            tc = json.load(f)
        auto_map = tc.get("auto_map", {})
        at = auto_map.get("AutoTokenizer")
        if isinstance(at, str):
            # Fix: string → [string, null] (vLLM requires this format)
            auto_map["AutoTokenizer"] = [at, None]
            tc["auto_map"] = auto_map
            with open(tc_path, "w") as f:
                json.dump(tc, f, indent=2)
                f.write("\n")
            print("  Fixed: tokenizer_config.json (AutoTokenizer format)")
        else:
            print("  OK: tokenizer_config.json (already fixed)")
    else:
        print(f"  WARNING: {tc_path} not found")


# ═════════════════════════════════════════════
# STEP 2: Prepare checkpoint directory
# ═════════════════════════════════════════════
def prepare_checkpoint(model_dir):
    print(f"\n[2/3] Preparing checkpoint: {model_dir} ...")

    if not os.path.isdir(model_dir):
        print(f"  ERROR: {model_dir} does not exist!")
        sys.exit(1)

    # ── Synchronize checked-in shared files ──
    for fname in SHARED_FILES:
        src = os.path.join(SHARED_CODE_DIR, fname)
        dst = os.path.join(model_dir, fname)

        if not os.path.exists(src):
            print(f"  WARNING: shared file {src} not found, skipping")
            continue

        if os.path.islink(dst):
            os.unlink(dst)
        shutil.copy2(src, dst)
        print(f"  Synced: {fname}")

    # ── Ensure config.json has auto_map ──
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if cfg.get("architectures") == ["CENOPForCausalLM"]:
        raise ValueError(
            "CENO-P is an MSA scoring model and is supported only by the HuggingFace VEP path, not vLLM generation."
        )

    changed = False
    if "auto_map" not in cfg:
        cfg["auto_map"] = {}
        changed = True

    if "AutoConfig" not in cfg["auto_map"]:
        cfg["auto_map"]["AutoConfig"] = "configuration_ceno.CENOConfig"
        changed = True
    if "AutoModelForCausalLM" not in cfg["auto_map"]:
        cfg["auto_map"]["AutoModelForCausalLM"] = "modeling_ceno.CENOForCausalLM"
        changed = True
    if "AutoTokenizer" not in cfg["auto_map"]:
        cfg["auto_map"]["AutoTokenizer"] = ["ceno_tokenizer.CENOCharLevelTokenizer", None]
        changed = True

    # vLLM resolves the architecture in a fresh process.  Registering an alias
    # in this patcher's Python process is therefore insufficient.  Persistently
    # select vLLM's supported Nemotron-H implementation while retaining CENO's
    # model_type and HF auto_map for normal Transformers loading.
    if cfg.get("architectures") != ["NemotronHForCausalLM"]:
        cfg["architectures"] = ["NemotronHForCausalLM"]
        changed = True

    # ── Fix stale vLLM alias fields ──
    # The conversion script may save moe_intermediate_size with the default value
    # (21504 for the full model) even when intermediate_size was updated to the
    # actual model dimension.  vLLM's built-in config reads these fields directly
    # from config.json, so they MUST match intermediate_size.
    isize = cfg.get("intermediate_size")
    if isize is not None:
        for alias in ("moe_intermediate_size", "moe_shared_expert_intermediate_size"):
            if cfg.get(alias) != isize:
                print(f"  Fixing {alias}: {cfg.get(alias)} -> {isize}")
                cfg[alias] = isize
                changed = True

    # ── Fix head_dim for smaller models ──
    # The conversion script may leave the default head_dim (128) from the full
    # model config, but smaller models (e.g. 600M with hidden_size=1024,
    # num_attention_heads=16) need head_dim = hidden_size / num_attention_heads.
    # vLLM's NemotronHAttention reads config.head_dim directly, and if it's
    # wrong, QKV weight loading will crash with a dimension mismatch.
    hidden_size = cfg.get("hidden_size")
    num_heads = cfg.get("num_attention_heads")
    if hidden_size is not None and num_heads is not None and num_heads > 0:
        correct_head_dim = hidden_size // num_heads
        for key in ("head_dim", "attention_head_dim"):
            val = cfg.get(key)
            if val is not None and val != correct_head_dim:
                print(f"  Fixing {key}: {val} -> {correct_head_dim}"
                      f"  (hidden_size={hidden_size} / num_attention_heads={num_heads})")
                cfg[key] = correct_head_dim
                changed = True
            elif val is None and key == "head_dim":
                # Ensure head_dim is always explicitly set
                print(f"  Setting {key}: {correct_head_dim}"
                      f"  (hidden_size={hidden_size} / num_attention_heads={num_heads})")
                cfg[key] = correct_head_dim
                changed = True

    # ── Force MoE router math to stay in model dtype (bf16), not fp32 ──
    # Earlier debugging showed that fp32 router logits can change routing behavior
    # enough to create a mismatch against native Megatron. For vLLM comparison we
    # want the router to follow the model dtype path unless explicitly overridden
    # elsewhere, so pin the checkpoint config to bfloat16 here.
    router_dtype = str(cfg.get("moe_router_dtype", "") or "").lower()
    if router_dtype != "bfloat16":
        print(f"  Fixing moe_router_dtype: {cfg.get('moe_router_dtype')} -> bfloat16")
        cfg["moe_router_dtype"] = "bfloat16"
        changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print("  Updated: config.json")
    else:
        print("  OK: config.json (no changes needed)")


# ═════════════════════════════════════════════
# STEP 3: Patch vLLM source code
# ═════════════════════════════════════════════
def find_vllm_nemotron_h():
    """Find the vLLM nemotron_h.py file."""
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nemotron_h.py",
        "/usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/nemotron_h.py",
        "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/nemotron_h.py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    result = subprocess.run(
        ["python3", "-c",
         "import vllm.model_executor.models.nemotron_h as m; print(m.__file__)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print("ERROR: Cannot find vLLM's nemotron_h.py!")
    sys.exit(1)


def find_vllm_mamba_mixer2():
    """Find the vLLM mamba_mixer2.py file."""
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py",
        "/usr/local/lib/python3.11/dist-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py",
        "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    result = subprocess.run(
        ["python3", "-c",
         "import vllm.model_executor.layers.mamba.mamba_mixer2 as m; print(m.__file__)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    print("ERROR: Cannot find vLLM's mamba_mixer2.py!")
    sys.exit(1)


def safe_replace(source, old, new, desc=""):
    if old not in source:
        print(f"  WARNING: Could not find target for '{desc}' — skipping")
        return source
    count = source.count(old)
    if count > 1:
        print(f"  WARNING: Multiple matches ({count}) for '{desc}' — replacing first only")
    result = source.replace(old, new, 1)
    print(f"  Patched: {desc}")
    return result


def patch_vllm_source():
    try:
        installed_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("vLLM is not installed. Install requirements-vllm.txt first.") from exc
    if installed_version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"This patcher supports vLLM {SUPPORTED_VLLM_VERSION}; found {installed_version}. "
            "Do not apply source patches to an unvalidated vLLM release."
        )
    vllm_path = find_vllm_nemotron_h()
    print(f"\n[3/3] Patching vLLM source: {vllm_path} ...")

    backup_path = vllm_path + ".orig"
    if os.path.exists(backup_path):
        print(f"  Restoring from backup: {backup_path}")
        shutil.copy2(backup_path, vllm_path)
    else:
        shutil.copy2(vllm_path, backup_path)
        print(f"  Saved original to: {backup_path}")

    with open(vllm_path, "r") as f:
        source = f.read()

    source = patch_mamba_decoder_layer(source)
    # Keep vLLM's native contiguous q/k/v split when loading correctly
    # converted HF checkpoints. The old interleaved forward patch only
    # compensated for legacy HF exports with incorrectly split attention
    # weights and breaks newly reconverted checkpoints.
    source = patch_attention_decoder_layer(source)
    source = patch_moe_decoder_layer(source)
    source = patch_mlp_decoder_layer(source)
    source = patch_moe_class(source)
    source = patch_expert_mapping(source)
    source = patch_non_gated_moe_flag(source)
    source = patch_model_final_norm(source)
    source = patch_get_max_n_routed_experts(source)

    with open(vllm_path, "w") as f:
        f.write(source)

    try:
        compile(source, vllm_path, "exec")
        print("\n  Syntax check: OK")
    except SyntaxError as e:
        print(f"\n  ERROR: Syntax error in patched file: {e}")
        print("  Restoring original...")
        shutil.copy2(backup_path, vllm_path)
        sys.exit(1)

    mamba_mixer2_path = find_vllm_mamba_mixer2()
    print(f"\n[3/3b] Patching vLLM Mamba mixer: {mamba_mixer2_path} ...")

    mixer_backup_path = mamba_mixer2_path + ".orig"
    if os.path.exists(mixer_backup_path):
        print(f"  Restoring from backup: {mixer_backup_path}")
        shutil.copy2(mixer_backup_path, mamba_mixer2_path)
    else:
        shutil.copy2(mamba_mixer2_path, mixer_backup_path)
        print(f"  Saved original to: {mixer_backup_path}")

    with open(mamba_mixer2_path, "r") as f:
        mixer_source = f.read()

    mixer_source = patch_mamba_mixer2_norm_semantics(mixer_source)

    with open(mamba_mixer2_path, "w") as f:
        f.write(mixer_source)

    try:
        compile(mixer_source, mamba_mixer2_path, "exec")
        print("\n  Mamba mixer syntax check: OK")
    except SyntaxError as e:
        print(f"\n  ERROR: Syntax error in patched mixer file: {e}")
        print("  Restoring original...")
        shutil.copy2(mixer_backup_path, mamba_mixer2_path)
        sys.exit(1)

    print("  OK: checkpoint architecture was persisted as NemotronHForCausalLM for vLLM dispatch")


def patch_vllm_registry():
    """Register the "ceno" model_type + CENO architecture names as aliases.

    Our open-source checkpoints carry ``model_type: "ceno"`` and
    ``architectures: ["CENOForCausalLM"]`` (base) / ``["CENOPForCausalLM"]`` (CENO-P).
    vLLM's model registry, however, knows the Nemotron-H implementation under its
    original names (``"nemotron_h"`` / ``"NemotronHForCausalLM"``) — that file is
    vLLM's own, so we patch its *body* but do NOT rename its classes. To let a
    "ceno" checkpoint dispatch to that same (already-patched) Nemotron-H code, we
    register CENO as an alias pointing at vLLM's NemotronHForCausalLM.

    Defensive across vLLM versions: tries the known registry mechanisms and warns
    (does not abort) if none is found, since the rest of the patch is still useful.
    """
    print("\n[3/3c] Registering CENO aliases in vLLM model registry ...")
    try:
        from vllm.model_executor.models import nemotron_h as vllm_nemotron_h
        NemotronHForCausalLM = getattr(vllm_nemotron_h, "NemotronHForCausalLM", None)
        if NemotronHForCausalLM is None:
            print("  WARNING: NemotronHForCausalLM not found in vLLM's nemotron_h module; skipping registry alias")
            return
    except Exception as e:
        print(f"  WARNING: could not import vLLM's nemotron_h module ({e}); skipping registry alias")
        return

    aliases = [
        ("ceno", "CENOForCausalLM"),          # base model_type / architecture
        ("CENOForCausalLM", "CENOForCausalLM"),
        ("cenop", "CENOPForCausalLM"),        # CENO-P (no separate vLLM class —
        ("CENOPForCausalLM", "CENOPForCausalLM"),  #  MSA path is HF-only; alias to base)
    ]

    registered = 0

    # Mechanism 1: vLLM >= 0.6 ModelRegistry.register_model (architecture -> class).
    try:
        from vllm.model_executor.models.registry import ModelRegistry  # noqa: WPS433
        for arch, _ in aliases:
            try:
                ModelRegistry.register_model(arch, NemotronHForCausalLM)
                registered += 1
            except Exception:
                pass
        if registered:
            print(f"  Registered via ModelRegistry.register_model: {[a for a, _ in aliases][:registered]}")
    except Exception:
        pass

    # Mechanism 2: vLLM's _VLLM_MODELS / _REGISTRY_MODELS architecture->class dict.
    if not registered:
        for dict_name in ("_VLLM_MODELS", "_REGISTRY_MODELS"):
            try:
                import vllm.model_executor.models.registry as reg  # noqa: WPS433
                models = getattr(reg, dict_name, None)
                if isinstance(models, dict):
                    for arch, _ in aliases:
                        models[arch] = NemotronHForCausalLM
                    registered += len(aliases)
                    print(f"  Registered via {dict_name}: {[a for a, _ in aliases]}")
                    break
            except Exception:
                continue

    # Mechanism 3: model_type string alias (NemotronHConfig.model_type is "nemotron_h").
    # Point our CENOConfig (loaded from the checkpoint) at the same registry entry by
    # temporarily exposing model_type "nemotron_h" — handled at load time by the
    # architecture aliases above, so nothing to do here.
    if not registered:
        print("  WARNING: no vLLM registry mechanism found; CENO dispatch may require "
              "setting architectures to [\"NemotronHForCausalLM\"] in config.json.")
    else:
        print("  OK: CENO aliases registered (vLLM will dispatch 'ceno'/'CENOForCausalLM' "
              "to the patched Nemotron-H implementation).")


# ──────────────────────────────────────────────────
# Individual vLLM patches (same proven logic as before)
# ──────────────────────────────────────────────────

def patch_mamba_decoder_layer(source):
    pattern = re.compile(
        r"(class NemotronHMambaDecoderLayer\(nn\.Module\):.*?)(?=\nclass )",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHMambaDecoderLayer")
        return source

    replacement = r'''class HFCompatRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.has_in_proj_norm = getattr(config, 'mamba_in_proj_layernorm', False)

        self.mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.ssm_state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.mamba_num_heads * config.mamba_head_dim,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.mamba_hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        if self.has_in_proj_norm:
            self.mixer.in_proj_norm = HFCompatRMSNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
            )
            self.norm = None
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is not None:
            hidden_states = hidden_states + residual

        if self.has_in_proj_norm:
            normed = self.mixer.in_proj_norm(hidden_states)
            output = self.mixer(normed)
            return hidden_states + output, None
        else:
            normed = self.norm(hidden_states)
            output = self.mixer(normed)
            return hidden_states + output, None

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHMambaDecoderLayer")
    return source


def patch_mamba_mixer2_norm_semantics(source):
    pattern = re.compile(
        r"(    def forward_native\(\n.*?\n)(?=def mamba_v2_sharded_weight_loader\()",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find Mixer2RMSNormGated.forward_native")
        return source

    replacement = '''    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        gate = gate.to(torch.float32)
        x = x * nn.functional.silu(gate)
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        out = x * self.weight.to(torch.float32)
        return out.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            return x * nn.functional.silu(gate.to(torch.float32)).to(input_dtype)

        # For tp_size==1, vLLM's fused kernel already supports group RMSNorm.
        # The upstream guard is too conservative for our NemotronH case.
        if self.tp_size == 1:
            return rms_norm_gated(
                x,
                self.weight.data,
                bias=None,
                z=gate,
                eps=self.variance_epsilon,
                group_size=self.group_size,
                norm_before_gate=False,
            )

        if (self.n_groups % self.tp_size) != 0:
            return self.forward_native(x, gate)

        return rms_norm_gated(
            x,
            self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            group_size=self.group_size,
            norm_before_gate=False,
        )
'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: Mixer2RMSNormGated (prefer vLLM fused kernel, native fallback)")
    return source


def patch_attention_decoder_layer(source):
    pattern = re.compile(
        r"(class NemotronHAttentionDecoderLayer\(nn\.Module\):.*?)(?=\n\nALL_DECODER_LAYER_TYPES)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHAttentionDecoderLayer")
        return source

    replacement = r'''class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config

        self.mixer = NemotronHAttention(
            layer_config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        if getattr(config, 'qkv_layernorm', False):
            self.mixer.qkv_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is not None:
            hidden_states = hidden_states + residual

        input_ln = self.norm(hidden_states)
        if hasattr(self.mixer, 'qkv_norm'):
            input_ln = self.mixer.qkv_norm(input_ln)
        attn_out = self.mixer(hidden_states=input_ln)
        hidden_after_attn = hidden_states + attn_out
        output = self.pre_mlp_norm(hidden_after_attn)

        return output, None

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHAttentionDecoderLayer")
    return source


def patch_attention_module(source):
    pattern = re.compile(
        r"(class NemotronHAttention\(nn\.Module\):.*?)(?=\nclass NemotronHAttentionDecoderLayer)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHAttention")
        return source

    replacement = r'''class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        sliding_window = getattr(config, "sliding_window", None)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        # Megatron stores fused QKV in group-interleaved order:
        # [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...].
        # vLLM's default contiguous split assumes [Q_all, K_all, V_all], which
        # is incorrect for this checkpoint family and changes the real inference
        # result. Reconstruct grouped Q/K/V first, then flatten back to the
        # layout expected by vLLM's Attention layer.
        group_q_size = (self.num_heads // self.num_kv_heads) * self.head_dim
        group_kv_size = self.head_dim
        group_dim = group_q_size + group_kv_size + group_kv_size
        qkv_grouped = qkv.view(qkv.shape[0], self.num_kv_heads, group_dim)
        q, k, v = qkv_grouped.split(
            [group_q_size, group_kv_size, group_kv_size], dim=-1
        )
        q = q.reshape(qkv.shape[0], -1)
        k = k.reshape(qkv.shape[0], -1)
        v = v.reshape(qkv.shape[0], -1)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHAttention")
    return source


def patch_moe_decoder_layer(source):
    pattern = re.compile(
        r"(class NemotronHMoEDecoderLayer\(nn\.Module\):.*?)(?=\nclass )",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHMoEDecoderLayer")
        return source

    replacement = r'''class NemotronHMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config

        self.mixer = NemotronHMoE(
            layer_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is not None:
            hidden_states = hidden_states + residual

        input_ln = self.norm(hidden_states)
        pre_mlp = self.pre_mlp_norm(input_ln)
        moe_out = self.mixer(pre_mlp)
        output = input_ln + moe_out

        return output, None

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHMoEDecoderLayer")
    return source


def patch_mlp_decoder_layer(source):
    pattern = re.compile(
        r"(class NemotronHMLPDecoderLayer\(nn\.Module\):.*?)(?=\nclass )",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHMLPDecoderLayer")
        return source

    replacement = r'''class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.mixer = NemotronHMLP(
            config,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
        )

        # fc1_norm: matches Megatron TELayerNormColumnParallelLinear fused norm
        # before the MLP fc1/up_proj. Weight key: layers.{i}.mixer.fc1_norm.weight
        self.mixer.fc1_norm = RMSNorm(config.hidden_size,
                                       eps=config.layer_norm_epsilon)

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is not None:
            hidden_states = hidden_states + residual

        input_ln = self.norm(hidden_states)
        pre_mlp = self.pre_mlp_norm(input_ln)
        # Apply fc1_norm before the MLP (matches HF NemotronHMLP._maybe_apply_fc1_norm)
        pre_mlp = self.mixer.fc1_norm(pre_mlp)
        mlp_out = self.mixer(pre_mlp)
        output = input_ln + mlp_out

        return output, None

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHMLPDecoderLayer")
    return source


def patch_moe_class(source):
    pattern = re.compile(
        r"(class NemotronHMoE\(nn\.Module\):.*?)(?=\nclass )",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print("  WARNING: Could not find NemotronHMoE")
        return source

    replacement = r'''class NemotronHMoE(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = getattr(config, 'n_routed_experts',
                                              getattr(config, 'num_experts', 8))
        self.n_shared_experts: int = getattr(config, 'n_shared_experts', 0) or 0
        self.use_latent_moe: bool = getattr(config, "moe_latent_size", None) is not None
        self.moe_hidden_size: int = (
            config.moe_latent_size if self.use_latent_moe else config.hidden_size
        )

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # Detect MoE style: Mixtral (no shared experts) vs DeepSeek (shared experts)
        self.is_mixtral_style = (self.n_shared_experts == 0)

        router_dtype_name = str(getattr(config, 'moe_router_dtype', '') or '').lower()
        if router_dtype_name == 'fp32':
            router_logits_dtype = torch.float32
        elif router_dtype_name == 'fp64':
            router_logits_dtype = torch.float64
        else:
            router_logits_dtype = None
        self.router_logits_dtype = router_logits_dtype

        gate_params_dtype = getattr(config, 'torch_dtype', None)
        if isinstance(gate_params_dtype, str):
            gate_params_dtype = getattr(torch, gate_params_dtype, None)
        if gate_params_dtype is None:
            gate_params_dtype = torch.bfloat16
        if router_logits_dtype is not None:
            gate_params_dtype = router_logits_dtype

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            params_dtype=gate_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        if not self.is_mixtral_style:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )

        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if self.n_shared_experts > 0:
            shared_intermediate = (
                getattr(config, 'moe_shared_expert_intermediate_size',
                        config.intermediate_size) * self.n_shared_experts
            )
            self.shared_experts = NemotronHMLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        if self.use_latent_moe:
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=self.moe_hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                disable_tp=self.is_sequence_parallel,
                prefix=f"{prefix}.fc1_latent_proj",
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=config.hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                disable_tp=self.is_sequence_parallel,
                prefix=f"{prefix}.fc2_latent_proj",
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

        top_k = getattr(config, 'num_experts_per_tok',
                         getattr(config, 'moe_top_k', 2))
        intermediate_size = getattr(config, 'moe_intermediate_size',
                                     config.intermediate_size)
        uses_swiglu = getattr(config, 'mlp_use_swiglu', False)

        if self.is_mixtral_style:
            self.experts = FusedMoE(
                num_experts=self.n_routed_experts,
                top_k=top_k,
                hidden_size=self.moe_hidden_size,
                intermediate_size=intermediate_size,
                reduce_results=False,
                renormalize=getattr(config, 'norm_topk_prob', True),
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                activation=activation_without_mul(config.mlp_hidden_act),
                is_act_and_mul=uses_swiglu,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )
        else:
            self.experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=self.n_routed_experts,
                top_k=top_k,
                hidden_size=self.moe_hidden_size,
                intermediate_size=intermediate_size,
                reduce_results=False,
                renormalize=getattr(config, 'norm_topk_prob', False),
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=getattr(config, 'n_group', 1),
                topk_group=getattr(config, 'topk_group', 1),
                prefix=f"{prefix}.experts",
                scoring_func="sigmoid",
                e_score_correction_bias=self.gate.e_score_correction_bias,
                activation=activation_without_mul(config.mlp_hidden_act),
                is_act_and_mul=uses_swiglu,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
                router_logits_dtype=(self.router_logits_dtype or gate_params_dtype),
                routed_input_transform=self.fc1_latent_proj,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_input = hidden_states
        if self.router_logits_dtype is not None:
            router_input = hidden_states.to(dtype=self.router_logits_dtype)
        router_logits, _ = self.gate(router_input)

        if self.is_mixtral_style:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        else:
            shared_output, final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
            if hidden_states.dtype != torch.float16:
                final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
            if self.use_latent_moe:
                final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)
            if self.shared_experts is not None:
                final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: NemotronHMoE (Mixtral-style)")
    return source


def patch_expert_mapping(source):
    """Patch get_expert_mapping to handle SwiGLU (gated) MoE experts.

    The original vLLM code hardcodes non-gated expert weight mapping
    (ckpt_gate_proj_name="up_proj", ckpt_up_proj_name=""). For SwiGLU
    models the checkpoint contains separate gate_proj and up_proj weights.
    """
    old = 'ckpt_gate_proj_name="up_proj",'
    if old not in source:
        print("  INFO: get_expert_mapping gate_proj_name not found (OK if already patched)")
        return source

    old_block = (
        '            expert_params_mapping = FusedMoE.make_expert_params_mapping(\n'
    )
    if old_block not in source:
        print("  WARNING: Could not locate make_expert_params_mapping call")
        return source

    idx = source.index(old_block)
    end_marker = "            return expert_params_mapping"
    end_idx = source.index(end_marker, idx)

    old_section = source[idx:end_idx]

    new_section = (
        '            uses_swiglu = getattr(self.config, "mlp_use_swiglu", False)\n'
        '            _n_experts = getattr(self.config, "n_routed_experts",\n'
        '                                 getattr(self.config, "num_experts", 8))\n'
        '            _n_redundant = getattr(self, "num_redundant_experts", 0)\n'
        '            if uses_swiglu:\n'
        '                expert_params_mapping = FusedMoE.make_expert_params_mapping(\n'
        '                    self,\n'
        '                    ckpt_gate_proj_name="gate_proj",\n'
        '                    ckpt_down_proj_name="down_proj",\n'
        '                    ckpt_up_proj_name="up_proj",\n'
        '                    num_experts=_n_experts,\n'
        '                    num_redundant_experts=_n_redundant,\n'
        '                )\n'
        '            else:\n'
        '                expert_params_mapping = FusedMoE.make_expert_params_mapping(\n'
        '                    self,\n'
        '                    ckpt_gate_proj_name="up_proj",\n'
        '                    ckpt_down_proj_name="down_proj",\n'
        '                    ckpt_up_proj_name="",\n'
        '                    num_experts=_n_experts,\n'
        '                    num_redundant_experts=_n_redundant,\n'
        '                )\n'
    )

    source = source[:idx] + new_section + source[end_idx:]
    print("  Patched: get_expert_mapping (SwiGLU gated MoE support)")
    return source


def patch_non_gated_moe_flag(source):
    """Patch is_non_gated_moe to be dynamic based on config.mlp_use_swiglu.

    The original class attribute is hardcoded True (non-gated). For SwiGLU
    models the MoE experts ARE gated, so this must be overridden per-instance.
    """
    old = "    is_non_gated_moe: bool = True"
    if old not in source:
        print("  INFO: is_non_gated_moe not found (OK if already patched)")
        return source

    new = "    is_non_gated_moe: bool = True  # overridden in __init__ for SwiGLU"
    source = source.replace(old, new, 1)

    init_marker = "        self.config = config"
    if init_marker in source:
        count = source.count(init_marker)
        if count > 1:
            idx = source.rfind(init_marker)
        else:
            idx = source.index(init_marker)
        insert_after = idx + len(init_marker)
        patch_line = (
            '\n        self.is_non_gated_moe = not getattr(config, "mlp_use_swiglu", False)'
        )
        source = source[:insert_after] + patch_line + source[insert_after:]
        print("  Patched: is_non_gated_moe (dynamic per config)")
    else:
        print("  WARNING: Could not find self.config = config in NemotronHForCausalLM.__init__")

    return source


def patch_model_final_norm(source):
    old_norm = "        hidden_states, _ = self.norm_f(hidden_states, residual)"
    new_norm = """        if residual is not None:
            hidden_states, _ = self.norm_f(hidden_states, residual)
        else:
            hidden_states = self.norm_f(hidden_states)"""
    source = safe_replace(source, old_norm, new_norm,
                          "NemotronHModel.forward final norm_f")
    return source


def patch_get_max_n_routed_experts(source):
    """Patch _get_max_n_routed_experts to also check 'num_experts' from HF config.

    NOTE: If configuration_ceno.py already sets self.n_routed_experts = num_experts,
    this patch is redundant (the original function will find it). We apply it as a safety net.
    """
    # Try multiple regex patterns to handle different vLLM versions/formatting
    patterns = [
        # Pattern A: 4-space indent, followed by next 4-space def
        re.compile(r"(    def _get_max_n_routed_experts\(self\)[^\n]*\n.*?)(?=\n    def )", re.DOTALL),
        # Pattern B: any whitespace indent
        re.compile(r"(\s+def _get_max_n_routed_experts\(self\)[^\n]*\n.*?)(?=\n\s+def )", re.DOTALL),
    ]

    match = None
    for pat in patterns:
        match = pat.search(source)
        if match:
            break

    if not match:
        # Not critical: config fix (n_routed_experts alias) should make this work anyway
        print("  INFO: _get_max_n_routed_experts not found (OK if config sets n_routed_experts)")
        return source

    # Detect indentation from matched text
    first_line = match.group(0).split('\n')[0]
    indent = first_line[:len(first_line) - len(first_line.lstrip())]

    replacement = f'''{indent}def _get_max_n_routed_experts(self) -> int:
{indent}    """Get max n_routed_experts, with fallback to HF num_experts."""
{indent}    # Check vLLM-native attribute first
{indent}    n_routed_experts = getattr(self.config, "n_routed_experts", None)
{indent}    if n_routed_experts is not None:
{indent}        return n_routed_experts
{indent}
{indent}    # Fallback: HF NemotronHConfig uses 'num_experts'
{indent}    num_experts = getattr(self.config, "num_experts", None)
{indent}    if num_experts is not None:
{indent}        return num_experts
{indent}
{indent}    # For puzzle models, get MAX from all MoE blocks in block_configs
{indent}    max_experts = 0
{indent}    block_configs = getattr(self.config, "block_configs", None)
{indent}    if block_configs:
{indent}        for block in block_configs:
{indent}            if isinstance(block, dict):
{indent}                if block.get("block_type") == "moe":
{indent}                    max_experts = max(max_experts,
{indent}                                      block.get("n_routed_experts", 0))
{indent}            else:
{indent}                if getattr(block, "block_type", "") == "moe":
{indent}                    max_experts = max(max_experts,
{indent}                                      getattr(block, "n_routed_experts", 0))
{indent}    return max_experts

'''
    source = source[:match.start()] + replacement + source[match.end():]
    print("  Patched: _get_max_n_routed_experts (HF num_experts fallback)")
    return source


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generation/patch_vllm_for_dna.py <checkpoint_dir>")
        print("Example: python3 generation/patch_vllm_for_dna.py /path/to/ceno_checkpoint")
        sys.exit(1)

    model_dir = sys.argv[1]

    print("=" * 60)
    print("DNA CENO vLLM Patcher v6")
    print("=" * 60)
    print(f"Checkpoint:  {model_dir}")
    print(f"Shared code: {SHARED_CODE_DIR}")
    print()

    patch_shared_code()
    prepare_checkpoint(model_dir)
    patch_vllm_source()

    print("\n" + "=" * 60)
    print("Done! Run inference with:")
    print("=" * 60)
    print(f"""
python3 generation/vllm_offline_infer.py {model_dir}
""")
