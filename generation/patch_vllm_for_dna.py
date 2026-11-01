#!/usr/bin/env python3
"""
Patch vLLM for DNA NemotronH inference.

This script does three things:
  1. Patches shared model code (evo2_tokenizer, tokenizer_config) — once
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
    python3 generation/patch_vllm_for_dna.py \
        /mnt/shared-storage-user/dnacoding/nemotron_models_v2/hf/ablation/300M_A100M_stage2_v1_c2
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────
# Shared model code (tokenizer, config class, modeling)
# ─────────────────────────────────────────────
SHARED_CODE_DIR = os.environ.get(
    "CENO_SHARED_CODE_DIR", os.path.join(os.path.dirname(__file__), "..", "ceno_model", "ceno_hf")
)
SUPPORTED_VLLM_VERSION = "0.15.1"


def validate_vllm_version():
    version = importlib.metadata.version("vllm")
    if version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"This CENO patch requires vLLM {SUPPORTED_VLLM_VERSION}; found {version}")

# Files to symlink from SHARED_CODE_DIR into each checkpoint
SHARED_FILES = [
    "training_precision.py",
    "ceno_norm.py",
    "ceno_cp_math.py",
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

    # ── Fix ceno_tokenizer.py: numpy.uint8 → Python int ──
    tok_path = os.path.join(SHARED_CODE_DIR, "ceno_tokenizer.py")
    if os.path.exists(tok_path):
        with open(tok_path, "r") as f:
            src = f.read()
        old_tok = "token_ids = list(np.fromstring(text, dtype=np.uint8))"
        new_tok = ("# .tolist() converts numpy.uint8 → native Python int\n"
                   "        # (required for vLLM's msgspec serialization)\n"
                   "        token_ids = np.fromstring(text, dtype=np.uint8).tolist()")
        if old_tok in src:
            src = src.replace(old_tok, new_tok)
            with open(tok_path, "w") as f:
                f.write(src)
            print("  Fixed: ceno_tokenizer.py (numpy.uint8 → int)")
        else:
            print("  OK: ceno_tokenizer.py (already fixed)")
    else:
        print(f"  WARNING: {tok_path} not found")

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
def prepare_checkpoint(model_dir, ssm_cache_dtype="float32", mamba_gate_mode="checkpoint",
                       rmsnorm_mode="checkpoint", reference_cp_size="checkpoint",
                       reference_context_length=None):
    print(f"\n[2/3] Preparing checkpoint: {model_dir} ...")

    if not os.path.isdir(model_dir):
        print(f"  ERROR: {model_dir} does not exist!")
        sys.exit(1)

    # ── Symlink shared files ──
    for fname in SHARED_FILES:
        src = os.path.join(SHARED_CODE_DIR, fname)
        dst = os.path.join(model_dir, fname)

        if not os.path.exists(src):
            print(f"  WARNING: shared file {src} not found, skipping")
            continue

        if fname.endswith(".py") and os.path.abspath(src) != os.path.abspath(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            shutil.copy2(src, dst)
            print(f"  Updated runtime code: {fname}")
            continue
        if os.path.islink(dst):
            # Already a symlink — check if it points to the right place
            if os.readlink(dst) == src:
                continue
            else:
                os.remove(dst)
                os.symlink(src, dst)
                print(f"  Relinked: {fname}")
        elif os.path.exists(dst):
            # Real file exists — skip (don't overwrite checkpoint files)
            continue
        else:
            os.symlink(src, dst)
            print(f"  Linked: {fname}")

    # ── Ensure config.json has auto_map ──
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    if cfg.get("architectures") == ["CENOPForCausalLM"]:
        raise ValueError("CENO-P uses the HF MSA path, not vLLM generation")
    changed = cfg.get("architectures") != ["NemotronHForCausalLM"]
    cfg["architectures"] = ["NemotronHForCausalLM"]
    if rmsnorm_mode not in ("checkpoint", "torch", "te-general", "te-tuned"):
        raise ValueError(f"Unknown RMSNorm mode: {rmsnorm_mode}")
    selected_norm = cfg.get("ceno_rmsnorm_mode", "torch") if rmsnorm_mode == "checkpoint" else rmsnorm_mode
    if selected_norm not in ("torch", "te-general", "te-tuned"):
        raise ValueError(f"Unsupported checkpoint RMSNorm mode: {selected_norm}")
    if selected_norm.startswith("te-") and cfg.get("hidden_size") not in (512, 1024):
        raise ValueError("The validated TE RMSNorm specialization requires hidden_size=512 or 1024")
    if rmsnorm_mode != "checkpoint" and cfg.get("ceno_rmsnorm_mode") != rmsnorm_mode:
        cfg["ceno_rmsnorm_mode"] = rmsnorm_mode
        changed = True
    if mamba_gate_mode not in ("checkpoint", "training", "inference"):
        raise ValueError(f"Unknown Mamba gate mode: {mamba_gate_mode}")
    # Training's memory-efficient scan gates before the BF16 output boundary.
    # Older exports omit this numerical contract, so require an explicit choice
    # rather than guessing the original training recipe for every checkpoint.
    if mamba_gate_mode != "checkpoint":
        gate_in_ssm = mamba_gate_mode == "training"
        if cfg.get("ceno_mamba_gate_in_ssm") != gate_in_ssm:
            cfg["ceno_mamba_gate_in_ssm"] = gate_in_ssm
            changed = True
    if reference_cp_size not in ("checkpoint", "1", "8", 1, 8):
        raise ValueError("Reference CP size must be checkpoint, 1, or 8")
    cp = cfg.get("ceno_reference_cp_size", 1)
    if reference_cp_size != "checkpoint":
        cp = int(reference_cp_size)
        if cfg.get("ceno_reference_cp_size") != cp:
            cfg["ceno_reference_cp_size"] = cp
            changed = True
        if cp == 1 and "ceno_reference_context_length" in cfg:
            del cfg["ceno_reference_context_length"]
            changed = True
    if reference_context_length is not None:
        if cp != 8:
            raise ValueError("An explicit reference context is only used for CP8")
        if cfg.get("ceno_reference_context_length") != reference_context_length:
            cfg["ceno_reference_context_length"] = reference_context_length
            changed = True
    if cp not in (1, 8):
        raise ValueError(f"Unsupported checkpoint reference CP size: {cp}")
    if cp == 8:
        length = cfg.get("ceno_reference_context_length")
        if not isinstance(length, int) or length <= 0 or length % 1024:
            raise ValueError("CP8 needs an explicit reference context divisible by 1024")
        if selected_norm != "te-tuned" or not cfg.get("ceno_mamba_gate_in_ssm", False):
            raise ValueError("CP8 requires te-tuned RMSNorm and the training Mamba gate")
    if "auto_map" not in cfg:
        cfg["auto_map"] = {}
        changed = True

    if "AutoConfig" not in cfg["auto_map"]:
        cfg["auto_map"]["AutoConfig"] = "configuration_ceno.CENOConfig"
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

    # Router precision is checkpoint semantics, not a loader compatibility alias.
    # In particular, replacing an explicit fp32 router with BF16 can change the
    # selected experts. Missing/null values already mean model dtype in HF.
    print(f"  Preserving moe_router_dtype: {cfg.get('moe_router_dtype')}")

    # Unlike weights/router math, this is an inference cache policy. HF keeps
    # the final Mamba scan state in FP32. BF16 recurrent updates accumulate
    # rounding error over thousands of decode steps, even with identical input.
    if ssm_cache_dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"Unsupported SSM cache dtype: {ssm_cache_dtype}")
    if cfg.get("mamba_ssm_cache_dtype") != ssm_cache_dtype:
        print(f"  Setting mamba_ssm_cache_dtype: {cfg.get('mamba_ssm_cache_dtype')}"
              f" -> {ssm_cache_dtype}")
        cfg["mamba_ssm_cache_dtype"] = ssm_cache_dtype
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
    validate_vllm_version()
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
    source = patch_hf_rmsnorm_semantics(source)
    source = patch_get_max_n_routed_experts(source)

    for helper_name in ("ceno_moe.py", "ceno_mamba.py", "ceno_norm.py",
                        "ceno_chunk.py", "ceno_attention.py", "ceno_precision.py",
                        "ceno_cp_math.py", "ceno_cp_decode.py", "ceno_cp_attention.py"):
        helper = os.path.join(os.path.dirname(__file__), helper_name)
        with open(helper) as helper_file:
            helper_source = helper_file.read()
        compile(helper_source, helper, "exec")
        helper_target = os.path.join(os.path.dirname(vllm_path), "..", "layers", helper_name)
        with open(helper_target, "w") as helper_file:
            helper_file.write(helper_source)

    license_dir = os.path.join(os.path.dirname(vllm_path), "..", "layers", "third_party")
    os.makedirs(license_dir, exist_ok=True)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "third_party", "TransformerEngine-LICENSE"),
                 os.path.join(license_dir, "TransformerEngine-LICENSE"))

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
    mixer_source = patch_mamba_gate_boundary(mixer_source)

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

    conv_path = os.path.join(os.path.dirname(mamba_mixer2_path), "ops", "causal_conv1d.py")
    conv_backup = conv_path + ".orig"
    if not os.path.exists(conv_backup):
        shutil.copy2(conv_path, conv_backup)
    with open(conv_backup, "r") as f:
        conv_source = patch_causal_conv1d_precision(f.read())
    compile(conv_source, conv_path, "exec")
    with open(conv_path, "w") as f:
        f.write(conv_source)
    print("  Patched: causal_conv1d prefill/decode FP32 products")
    scan_targets = {
        "ssd_bmm.py": {"_bmm_chunk_fwd_kernel": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}},
        "ssd_chunk_scan.py": {"_chunk_scan_fwd_kernel": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}},
        "ssd_chunk_state.py": {
            "_chunk_cumsum_fwd_kernel": {"BLOCK_SIZE_H": 4},
            "_chunk_state_fwd_kernel": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        },
        "ssd_state_passing.py": {"_state_passing_fwd_kernel": {"BLOCK_SIZE": 256}},
    }
    for name, targets in scan_targets.items():
        path = os.path.join(os.path.dirname(conv_path), name)
        backup = path + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        with open(backup) as scan_file:
            fixed = patch_scan_autotuning(scan_file.read(), targets)
        compile(fixed, path, "exec")
        with open(path, "w") as scan_file:
            scan_file.write(fixed)
        print(f"  Patched: fixed CENO scan configuration in {name}")


def patch_scan_autotuning(source, targets):
    """Pin reduction tiles in source so eager AND compiled execution use them."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    found = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in targets:
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "autotune"):
                continue
            configs = next((kw.value for kw in decorator.keywords if kw.arg == "configs"), None)
            if not isinstance(configs, ast.List):
                raise RuntimeError(f"Unsupported autotune list: {node.name}")
            matches = [config for config in configs.elts if isinstance(config, ast.Call)
                       and config.args and ast.literal_eval(config.args[0]) == targets[node.name]]
            if len(matches) != 1:
                raise RuntimeError(f"Missing unique CENO scan configuration: {node.name}")
            begin = offsets[configs.lineno-1] + configs.col_offset
            end = offsets[configs.end_lineno-1] + configs.end_col_offset
            edits.append((begin, end, "[" + ast.get_source_segment(source, matches[0]) + "]"))
            found.add(node.name)
    if found != set(targets):
        raise RuntimeError(f"Missing native scan kernels: {set(targets)-found}")
    for begin, end, replacement in sorted(edits, reverse=True):
        source = source[:begin] + replacement + source[end:]
    return source


def patch_causal_conv1d_precision(source):
    """Avoid BF16 product rounding before the FP32 convolution accumulator."""
    old = "acc += matrix_x * matrix_w  # [BLOCK_N]"
    new = "acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)  # [BLOCK_N]"
    if source.count(new) == 2 and old not in source:
        return source
    if source.count(old) != 2:
        raise RuntimeError(
            "Unsupported causal_conv1d source: expected both prefill and decode products"
        )
    return source.replace(old, new)


def patch_hf_rmsnorm_semantics(source):
    """Match CENO's FP32 normalization AND weight multiplication before cast."""
    if "class HFCompatRMSNorm" not in source:
        raise RuntimeError("HF-compatible RMSNorm definition is missing")
    source, count = re.subn(r"(?<![A-Za-z0-9_])RMSNorm\(", "HFCompatRMSNorm(", source)
    if not count and "self.norm_f = HFCompatRMSNorm(" not in source:
        raise RuntimeError("Unsupported NemotronH RMSNorm layout")
    print(f"  Patched: {count} RMSNorm constructors to CENO FP32 weight-multiply semantics")
    return source


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
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config().model_config.hf_config
        self.ceno_rmsnorm_mode = getattr(config, 'ceno_rmsnorm_mode', 'torch')
        if self.ceno_rmsnorm_mode not in ('torch', 'te-general', 'te-tuned'):
            raise ValueError(f'Unsupported RMSNorm mode: {self.ceno_rmsnorm_mode}')
        if self.ceno_rmsnorm_mode.startswith('te-') and hidden_size not in (512, 1024):
            raise ValueError('TE RMSNorm requires hidden_size=512 or 1024')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ceno_rmsnorm_mode != 'torch':
            from vllm.model_executor.layers.ceno_norm import te_rms_norm
            return te_rms_norm(hidden_states, self.weight, self.variance_epsilon,
                               self.ceno_rmsnorm_mode)
        from vllm.model_executor.layers.ceno_moe import ceno_rms_norm
        return ceno_rms_norm(hidden_states, self.weight, self.variance_epsilon)


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
        from vllm.model_executor.layers.ceno_mamba import configure_invariant_scan
        configure_invariant_scan()

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

        self.mixer.ceno_gate_in_ssm = bool(getattr(config, 'ceno_mamba_gate_in_ssm', False))
        self.mixer.ceno_dt_limit = tuple(getattr(config, 'time_step_limit', (0.0, float('inf'))))
        if self.mixer.ceno_gate_in_ssm and self.mixer.tp_size != 1:
            raise ValueError('Training Mamba gate boundary is currently validated for TP=1 only')

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


def patch_mamba_gate_boundary(source):
    """Opt-in training gate placement for CENO, preserving other Mamba models."""
    if "# CENO training gate boundary" in source:
        return source
    replacements = (
        (
            "hidden_states = self.norm(ssm_output, gate)",
            "# CENO training gate boundary\n"
            "        hidden_states = self.norm(\n"
            "            ssm_output, None if getattr(self, 'ceno_gate_in_ssm', False) else gate\n"
            "        )",
        ),
        (
            "gate = gate.to(torch.float32)\n        x = x * nn.functional.silu(gate)",
            "if gate is not None:\n"
            "            gate = gate.to(torch.float32)\n"
            "            x = x * nn.functional.silu(gate)",
        ),
        (
            "z=None,\n                dt_bias=self.dt_bias,",
            "z=(projected_states[num_decodes:num_actual_tokens, :self.tped_intermediate_size]\n"
            "                   .view(num_prefill_tokens, -1, self.head_dim)\n"
            "                   if getattr(self, 'ceno_gate_in_ssm', False) else None),\n"
            "                dt_bias=self.dt_bias,",
        ),
        (
            "z=None,\n                dt_bias=dt_bias,",
            "z=(projected_states[:num_decodes, :self.tped_intermediate_size]\n"
            "                   .view(num_decodes, -1, self.head_dim)\n"
            "                   if getattr(self, 'ceno_gate_in_ssm', False) else None),\n"
            "                dt_bias=dt_bias,",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Unsupported Mamba gate source: expected one {old!r}")
        source = source.replace(old, new)
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
        import os as _os
        self.dna_moe_backend = _os.environ.get(
            'DNANEMOTRON_VLLM_MOE_BACKEND', 'auto'
        ).strip().lower()
        if self.dna_moe_backend not in ('auto', 'ceno', 'fused', 'reference'):
            raise ValueError(
                'DNANEMOTRON_VLLM_MOE_BACKEND must be auto, ceno, fused or reference'
            )

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
        self.top_k = int(top_k)
        intermediate_size = getattr(config, 'moe_intermediate_size',
                                     config.intermediate_size)
        self.reference_intermediate_size = int(intermediate_size)
        self.reference_hidden_size = int(self.moe_hidden_size)
        uses_swiglu = getattr(config, 'mlp_use_swiglu', False)
        self.reference_activation = str(getattr(config, 'mlp_hidden_act', '')).lower()

        if self.dna_moe_backend in ('reference', 'ceno'):
            violations = []
            if not self.is_mixtral_style:
                violations.append('shared experts are unsupported')
            if uses_swiglu:
                violations.append('SwiGLU experts are unsupported')
            if self.reference_activation not in ('relu2', 'squared_relu'):
                violations.append('activation must be relu2/squared_relu')
            if self.tp_size != 1 or self.ep_size != 1:
                violations.append('TP and EP must both equal 1')
            if self.enable_eplb or self.n_redundant_experts != 0:
                violations.append('EPLB/redundant experts are unsupported')
            if self.is_sequence_parallel:
                violations.append('sequence parallelism is unsupported')
            if quant_config is not None:
                violations.append('quantization is unsupported')
            if self.dna_moe_backend == 'ceno' and self.top_k != 2:
                violations.append('CENO fused expert reduction requires top-k=2')
            if self.dna_moe_backend == 'ceno' and getattr(config, 'moe_router_pre_softmax', False):
                violations.append('CENO fused backend requires post-topk softmax')
            if self.dna_moe_backend == 'ceno' and self.use_latent_moe:
                violations.append('CENO fused backend does not support latent projections')
            if violations:
                raise ValueError(
                    'DNANemotron reference MoE contract failed: ' + '; '.join(violations)
                )

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

        if self.dna_moe_backend == 'auto':
            self.dna_moe_backend = 'ceno' if (
                self.is_mixtral_style and not uses_swiglu and not self.use_latent_moe
                and self.reference_activation in ('relu2', 'squared_relu')
                and self.top_k == 2 and self.tp_size == self.ep_size == 1
                and not self.is_sequence_parallel and not self.enable_eplb
                and self.n_redundant_experts == 0 and quant_config is None
                and not getattr(config, 'moe_router_pre_softmax', False)
                and self.experts.w13_weight.dtype == torch.bfloat16
            ) else 'fused'

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_input = hidden_states
        if self.router_logits_dtype is not None:
            router_input = hidden_states.to(dtype=self.router_logits_dtype)
        router_logits, _ = self.gate(router_input)

        if self.is_mixtral_style and self.dna_moe_backend == 'ceno':
            from vllm.model_executor.layers.ceno_moe import bf16_moe
            final_hidden_states = bf16_moe(
                hidden_states, self.experts.w13_weight, self.experts.w2_weight,
                router_logits, self.top_k,
            )
        elif self.is_mixtral_style and self.dna_moe_backend == 'reference':
            w13_weight = getattr(self.experts, 'w13_weight', None)
            w2_weight = getattr(self.experts, 'w2_weight', None)
            expected_w13 = (
                self.n_routed_experts,
                self.reference_intermediate_size,
                self.reference_hidden_size,
            )
            expected_w2 = (
                self.n_routed_experts,
                self.reference_hidden_size,
                self.reference_intermediate_size,
            )
            if w13_weight is None or tuple(w13_weight.shape) != expected_w13:
                raise RuntimeError(
                    f'unexpected reference w13 layout: '
                    f'{None if w13_weight is None else tuple(w13_weight.shape)}; '
                    f'expected {expected_w13}'
                )
            if w2_weight is None or tuple(w2_weight.shape) != expected_w2:
                raise RuntimeError(
                    f'unexpected reference w2 layout: '
                    f'{None if w2_weight is None else tuple(w2_weight.shape)}; '
                    f'expected {expected_w2}'
                )
            if (
                w13_weight.dtype is not torch.bfloat16
                or w2_weight.dtype is not torch.bfloat16
                or not w13_weight.is_contiguous()
                or not w2_weight.is_contiguous()
            ):
                raise RuntimeError(
                    'reference expert weights must be contiguous BF16 tensors'
                )

            router_logits_float = router_logits.to(torch.float32)
            topk_logits, selected_experts = torch.topk(
                router_logits_float, self.top_k, dim=-1
            )
            routing_weights = torch.softmax(topk_logits, dim=-1)
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.n_routed_experts
            ).permute(2, 1, 0)
            final_hidden_states = torch.zeros_like(hidden_states)
            # Match the checkpoint implementation exactly: BF16 up projection,
            # squared ReLU, BF16 route scaling before down projection, then
            # checkpoint-order index_add accumulation.
            for expert_index in range(self.n_routed_experts):
                route_slot, token_index = torch.where(expert_mask[expert_index])
                if token_index.numel() == 0:
                    continue
                current = hidden_states[token_index]
                intermediate = torch.nn.functional.linear(
                    current, w13_weight[expert_index]
                )
                intermediate = torch.pow(torch.relu(intermediate), 2)
                scale = routing_weights[token_index, route_slot].unsqueeze(-1).to(
                    dtype=current.dtype
                )
                intermediate = intermediate * scale
                expert_output = torch.nn.functional.linear(
                    intermediate, w2_weight[expert_index]
                )
                final_hidden_states.index_add_(
                    0, token_index, expert_output.to(dtype=hidden_states.dtype)
                )
        elif self.is_mixtral_style:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Disposable, independent runtime copy of the checkpoint")
    parser.add_argument("--ssm-cache-dtype", choices=("float32", "bfloat16", "float16"),
                        default="float32", help="FP32 is the long-decode default; lower precision is diagnostic")
    parser.add_argument("--mamba-gate-mode", choices=("checkpoint", "training", "inference"),
                        default="checkpoint",
                        help="training: gate inside SSM, for confirmed Megatron memory-efficient "
                             "training checkpoints; checkpoint: preserve stored contract")
    parser.add_argument("--rmsnorm-mode", choices=("checkpoint", "torch", "te-general", "te-tuned"),
                        default="checkpoint",
                        help="Explicit vLLM RMSNorm arithmetic; TE modes require BF16 CENO width 512/1024. "
                             "general: unaligned reference; tuned: aligned training kernel. "
                             "This option alone does not establish whole-model parity.")
    parser.add_argument("--reference-cp-size", choices=("checkpoint", "1", "8"),
                        default="checkpoint", help="Explicit Megatron BF16 reference profile")
    parser.add_argument("--reference-context-length", type=int,
                        help="Fixed CP8 training/reference window; never inferred from a batch")
    args = parser.parse_args()
    model_dir = args.model_dir

    print("=" * 60)
    print("CENO vLLM inference patcher (Megatron BF16 reference)")
    print("=" * 60)
    print(f"Checkpoint:  {model_dir}")
    print(f"Shared code: {SHARED_CODE_DIR}")
    with open(__file__, "rb") as patch_file:
        print(f"Patch SHA256: {hashlib.sha256(patch_file.read()).hexdigest()}")
    print()

    validate_vllm_version()
    patch_shared_code()
    prepare_checkpoint(model_dir, ssm_cache_dtype=args.ssm_cache_dtype,
                       mamba_gate_mode=args.mamba_gate_mode, rmsnorm_mode=args.rmsnorm_mode,
                       reference_cp_size=args.reference_cp_size,
                       reference_context_length=args.reference_context_length)
    patch_vllm_source()

    print("\n" + "=" * 60)
    print("Done! Run inference with:")
    print("=" * 60)
    print(f"""
python3 generation/vllm_offline_infer.py {model_dir}
""")
