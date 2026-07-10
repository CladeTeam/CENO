# coding=utf-8
# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, CENO Team. All rights reserved.
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
"""CENO model configuration"""

import re

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CENOConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CENOModel`]. It is used to instantiate a
    CENO model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CENO-v0.1 model.

    [todo](todo)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the CENO model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CENOModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 52):
            Number of hidden layers in the Transformer encoder.
        hybrid_override_pattern (`str`, *optional*, defaults to `"MMMM*MMMMM*MMMMM*MMMMM*MMMMMM"`):
            The pattern of the hybrid model. The pattern is a string of characters where each character represents one layer: M=Mamba, *=Attention, -=MLP, E=MoE
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        mlp_hidden_act (`str`, *optional*, defaults to "relu2"):
            The non-linear activation function in the MLP layers.
        mlp_use_swiglu (`bool`, *optional*, defaults to `False`):
            Whether to use SwiGLU activation (gated linear unit) in MLP layers.
        num_experts (`int`, *optional*, defaults to 8):
            Number of experts for MoE layers.
        top_k (`int`, *optional*, defaults to 2):
            Number of experts to select for each token in MoE layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*, defaults to None):
            Sliding window attention window size.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device.
        ssm_state_size (`int`, *optional*, defaults to 128):
            The dimension of the mamba state space latents.
        mamba_num_heads (`int`, *optional*, defaults to 128):
            Number of heads in Mamba layers.
        mamba_n_groups (`int`, *optional*, defaults to 8):
            Number of groups in Mamba layers.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each Mamba head.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the mamba intermediate size.
        mamba_hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the Mamba layers.
        mamba_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum value for the time step in Mamba.
        mamba_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum value for the time step in Mamba.
        mamba_dt_limit (`tuple`, *optional*, defaults to (0.0, float("inf"))):
            Limits for the time step in Mamba.
        mamba_dt_init_floor (`float`, *optional*, defaults to 1e-4):
            Floor value for time step initialization in Mamba.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the input and output projections of the mamba mixer block.
        mamba_in_proj_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply an additional RMSNorm before the Mamba in-projection. This matches
            Megatron's TELayerNormColumnParallelLinear fused Mamba in_proj norm when enabled.
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            Size of chunks for Mamba processing.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
            Whether to rescale the pre-normalization residual connections.
        num_experts (`int`, *optional*, defaults to 8):
            Number of experts in MoE layers.
        moe_top_k (`int`, *optional*, defaults to 2):
            Number of experts to route to in MoE layers (renamed from top_k to avoid generation config conflicts).
        qkv_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply an additional RMSNorm before QKV projections in attention layers. This matches
            Megatron's TELayerNormColumnParallelLinear fused QKV norm when enabled.
        intra_encoding_pattern (`str`, *optional*, defaults to `None`):
            Per-layer switch (length must equal `num_hidden_layers`) to control which layers use intra-sequence
            isolation. Characters in `{1, Y, y, S, s, T, t, +}` enable isolation; `{0, N, n, F, f, ., -}` disable.
            If omitted, intra-sequence isolation is disabled (legacy behavior).
    """

    model_type = "ceno"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=52,
        hybrid_override_pattern="MMMM*MMMMM*MMMMM*MMMMM*MMMMMM",
        num_attention_heads=32,
        attention_head_dim=128,
        num_key_value_heads=8,  # nemo: num_query_groups
        mlp_hidden_act="relu2",
        mlp_use_swiglu=False,
        num_experts=8,
        moe_top_k=2,
        moe_router_pre_softmax=False,
        attention_bias=False,
        qkv_layernorm=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02, # nemo: init_method_std
        layer_norm_epsilon=1e-5, # nemo: layernorm_epsilon
        residual_in_fp32=False,  #  Megatron Core default value
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=0,
        sliding_window=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        hidden_dropout=0.0, # * ADDED
        use_mamba_kernels=True,
        ssm_state_size=128, # mamba_state_size
        mamba_num_heads=128,
        mamba_n_groups=8,  # nemo: mamba_ssm_ngroups = num_heads
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_in_proj_layernorm=False,
        mamba_chunk_size=256,
        rescale_prenorm_residual=True,
        intra_encoding_pattern=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hybrid_override_pattern = hybrid_override_pattern
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.qkv_layernorm = qkv_layernorm

        # Validate and process hybrid_override_pattern
        # M: Mamba, *: Attention, -: MLP, E: MoE
        if self.hybrid_override_pattern is None:
            # Default pattern with even distribution of attention layers
            self.hybrid_override_pattern = self._generate_default_pattern()
        
        # Pattern should be a string where each character represents one layer
        # No processing needed since '-' represents MLP layers, not separators
        
        # For the default pattern, adjust it if num_hidden_layers doesn't match
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            # If using default pattern and layers don't match, regenerate
            if self.hybrid_override_pattern == "MMMM*MMMMM*MMMMM*MMMMM*MMMMMM":
                self.hybrid_override_pattern = self._generate_default_pattern()
            else:
                # Only assert if it's a user-provided pattern
                assert len(self.hybrid_override_pattern) == self.num_hidden_layers, f"hybrid_override_pattern length ({len(self.hybrid_override_pattern)}) must match num_hidden_layers ({self.num_hidden_layers})"
        
        assert re.match(r"^[M*\-E]+$", self.hybrid_override_pattern), "hybrid_override_pattern must only contain characters 'M', '*', '-', or 'E'"

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_use_swiglu = mlp_use_swiglu
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        # Megatron MoE routing: default is post-topk softmax (i.e., softmax after selecting top-k experts).
        # This flag mirrors Megatron's `--moe-router-pre-softmax`.
        self.moe_router_pre_softmax = moe_router_pre_softmax
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.use_mamba_kernels = use_mamba_kernels
        self.n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_in_proj_layernorm = mamba_in_proj_layernorm
        self.chunk_size = mamba_chunk_size
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.intra_encoding_pattern = intra_encoding_pattern
        
        # MoE parameters
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _generate_default_pattern(self):
        """Generate default hybrid pattern with even distribution of attention layers"""
        # Default to mostly Mamba with some attention layers evenly distributed
        pattern = ["M"] * self.num_hidden_layers
        
        # Place attention layers at roughly even intervals
        if self.num_hidden_layers >= 4:
            # For larger models, place attention every 4-5 layers
            attention_interval = max(4, self.num_hidden_layers // 8)
            for i in range(attention_interval - 1, self.num_hidden_layers, attention_interval):
                pattern[i] = "*"
        
        return ''.join(pattern)

    @property
    def layers_block_type(self):
        return [
            "mamba" if self.hybrid_override_pattern[i] == "M" else
            "attention" if self.hybrid_override_pattern[i] == "*" else
            "mlp" if self.hybrid_override_pattern[i] == "-" else "moe"
            for i in range(self.num_hidden_layers)]

    @property
    def intra_encoding_mask(self):
        """
        Per-layer boolean mask for intra-sequence isolation.
        If `intra_encoding_pattern` is provided, it is used; otherwise isolation is disabled.
        """
        if getattr(self, "intra_encoding_pattern", None) is not None:
            pat = self.intra_encoding_pattern
            if len(pat) != self.num_hidden_layers:
                raise ValueError(
                    f"intra_encoding_pattern length ({len(pat)}) must match num_hidden_layers ({self.num_hidden_layers})"
                )
            enable_set = {"1", "Y", "y", "S", "s", "T", "t", "+"}
            disable_set = {"0", "N", "n", "F", "f", ".", "-"}
            return [c in enable_set if c in enable_set.union(disable_set) else True for c in pat]

        # Default/legacy: no intra-sequence isolation
        return [False for _ in range(self.num_hidden_layers)]
