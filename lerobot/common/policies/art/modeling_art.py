#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain
from typing import Any, Dict, cast, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import transformers

from lerobot.common.policies.art.configuration_art import ARTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class ARTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ARTConfig
    name = "art"

    def __init__(
        self,
        config: ARTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ART(config)


        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        
        # print everything in the batch, with shape info
        # for k, v in batch.items():
        #     print(f"{k}: {v.shape}")

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )


        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        actions_hat, _ = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item(), "loss": l1_loss}
        
        # loss_dict["loss"] = l1_loss

        # loss = l1_loss

        return loss_dict

class ART(nn.Module):
    """ modified Action Chunking Transformer model, now with AR decoder and pizero style kv sharing.
    cache the kv features from vision encoder for use in decoder.
    deocder maintains its own cache for history states."""

    def __init__(self, config: ARTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config


        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ARTEncoder(config)
        self.decoder = ARTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        # if self.config.robot_state_feature:
        #     n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
    #     """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

    #     `batch` should have the following structure:
    #     {
    #         [robot_state_feature] (optional): (B, state_dim) batch of robot states.

    #         [image_features]: (B, n_cameras, C, H, W) batch of images.
    #             AND/OR
    #         [env_state_feature]: (B, env_dim) batch of environment states.

    #         [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
    #     }

    #     Returns:
    #         (B, chunk_size, action_dim) batch of action sequences
    #         Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
    #         latent dimension.
    #     """
    #     if self.config.use_vae and self.training:
    #         assert (
    #             "action" in batch
    #         ), "actions must be provided when using the variational objective in training mode."

    #     batch_size = (
    #         batch["observation.images"]
    #         if "observation.images" in batch
    #         else batch["observation.environment_state"]
    #     ).shape[0]
        
    #     # print everything in the batch, with shape info
    #     # for k, v in batch.items():
    #     #     print(f"batch[{k}]: shape={v.shape}, dtype={v.dtype}, device={v.device}")


    #     # Prepare the latent for input to the transformer encoder.

    #         # When not using the VAE encoder, we set the latent to be all zeros.
    #     mu = log_sigma_x2 = None
    #     # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
    #     latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
    #         batch["observation.state"].device
    #     )

    #     # Prepare transformer encoder inputs.
    #     encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    #     encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    #     # Robot state token.
    #     # if self.config.robot_state_feature:
    #     #     encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
    #     # Environment state token.
    #     if self.config.env_state_feature:
    #         encoder_in_tokens.append(
    #             self.encoder_env_state_input_proj(batch["observation.environment_state"])
    #         )

    #     # Camera observation features and positional embeddings.
    #     if self.config.image_features:
    #         all_cam_features = []
    #         all_cam_pos_embeds = []

    #         for cam_index in range(batch["observation.images"].shape[-4]):
    #             cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
    #             # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
    #             # buffer
    #             cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
    #             cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
    #             all_cam_features.append(cam_features)
    #             all_cam_pos_embeds.append(cam_pos_embed)
    #         # Concatenate camera observation feature maps and positional embeddings along the width dimension,
    #         # and move to (sequence, batch, dim).
    #         all_cam_features = torch.cat(all_cam_features, axis=-1)
    #         encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
    #         all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
    #         encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))

    #     # Stack all tokens along the sequence dimension.
    #     encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    #     encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        
    #     cache = transformers.DynamicCache()  # Initialize empty cache for KV caching

    #     # Forward pass through the transformer modules.
    #     encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed, past_key_value=cache)
    #     # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
    #     decoder_in = torch.zeros(
    #         (self.config.chunk_size, batch_size, self.config.dim_model),
    #         dtype=encoder_in_pos_embed.dtype,
    #         device=encoder_in_pos_embed.device,
    #     )
    #     decoder_in[:] = self.encoder_robot_state_input_proj(batch["observation.state"])
    #     decoder_out = self.decoder(
    #         decoder_in,
    #         encoder_out,
    #         encoder_pos_embed=encoder_in_pos_embed,
    #         decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
    #     )

    #     # Move back to (B, S, C).
    #     decoder_out = decoder_out.transpose(0, 1)

    #     actions = self.action_head(decoder_out)

    #     return actions, (mu, log_sigma_x2)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Training Forward:
        1. VLM Prefix -> Cache (Pos 0)
        2. States (History + Future) -> Decoder[hist:hist + future] -> Future Actions
        """
        
        # 1. Vision Encoder (Prefix)
        encoder_in_tokens = []
        if self.config.image_features:
            all_cam_features = []
            for cam_index in range(batch["observation.images"].shape[-4]):
                img = batch["observation.images"][:, cam_index]
                feat = self.backbone(img)["feature_map"]
                feat = self.encoder_img_feat_input_proj(feat)
                pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat = feat + pos
                all_cam_features.append(feat)
            
            all_cam_features = torch.cat(all_cam_features, dim=-1)
            encoder_in_tokens.append(einops.rearrange(all_cam_features, "b c h w -> b (h w) c"))

        encoder_in = torch.cat(encoder_in_tokens, dim=1)
        
        # Populate Cache (Prefix)
        cache = transformers.DynamicCache()
        _ = self.encoder(encoder_in, past_key_value=cache)
        prefix_len = cache[0][0].shape[-2]
        
        # print cache shapes, 2nd dim is 0,1 for k,v
        # for i, layer_cache in enumerate(cache):
        #     print(f"Layer {i}:")
        #     print(f"  Key shape: {layer_cache[0].shape}")
        #     print(f"  Value shape: {layer_cache[1].shape}")

        # 2. State Projection (History + Future combined)
        # We don't slice history/future separately; we treat it as one sequence.
        all_states = batch["observation.state"] # [B, Seq_Len, Dim]
        decoder_input = self.state_input_proj(all_states)
        
        batch_size, seq_len, _ = decoder_input.shape
        device = decoder_input.device
        hist_len = self.config.history_length
        
        assert seq_len == hist_len + self.config.chunk_size, ValueError(
            f"Sequence length {seq_len} does not match history {hist_len} + chunk size {self.config.chunk_size}"
        )

        # 3. Construct RoPE Positions
        # Sequence indices: [0, 1, 2, ... L]
        # Shift by history: [-H, -H+1, ... -1, 0, 1 ... F]
        position_ids = torch.arange(seq_len, device=device) - hist_len
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 4. Build Mask
        # Rules:
        # - Indices < 0 (History): Cannot see Prefix. Causal Self.
        # - Indices >= 0 (Future): See Prefix. Causal Self.
        mask = self._build_pizero_mask(batch_size, prefix_len, hist_len, seq_len, device)

        # 5. Run Decoder
        # DynamicCache automatically appends to the end, so we don't strictly need cache_position for training
        # unless we were doing something non-contiguous.
        decoder_out = self.decoder(
            decoder_input,
            attn_mask=mask,
            cache=cache,
            attn_kwargs={
                'query_position_indices': position_ids,
                'key_position_indices': position_ids,
                'cache_kwargs': {'cache_position': None} # Default: Append to end
            }
        )
        
        # for i, layer_cache in enumerate(cache):
        #     print(f"Layer {i}:")
        #     print(f"final  Key shape: {layer_cache[0].shape}")
        #     print(f"final  Value shape: {layer_cache[1].shape}")

        # 6. Extract Future (Indices corresponding to >= 0)
        future_out = decoder_out[:, hist_len:]
        
        return self.action_head(future_out), {}
    
    def _build_pizero_mask(self, batch_size, prefix_len, hist_len, seq_len, device):
        """
        Masking Logic:
        Q: [Seq_Len] (History + Future)
        K: [Prefix + Seq_Len]
        """
        k_len = prefix_len + seq_len
        mask = torch.zeros(batch_size, 1, seq_len, k_len, dtype=torch.bool, device=device)
        
        # Slices
        c_prefix = slice(0, prefix_len)
        c_seq = slice(prefix_len, k_len) # The AR part of Keys
        
        r_future = slice(hist_len, seq_len)
        
        # 1. Causal Self-Attention for the whole sequence
        # This handles History->History and Future->Future and Future->History
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask[:, :, :, c_seq] = causal
        
        # 2. Handle Vision Prefix Visibility
        # History (r_hist) -> Prefix: BLOCKED (False) -> Default is False, so do nothing.
        # Future (r_future) -> Prefix: VISIBLE (True)
        if prefix_len > 0:
            mask[:, :, r_future, c_prefix] = True
            
        return mask
    



class ARTEncoder(nn.Module):
    def __init__(self, config: ARTConfig, is_vae_encoder: bool = False):
        super().__init__()
        # Reusing ARTDecoderLayer logic but for Encoding
        num_layers = config.n_vae_encoder_layers if is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([
            ARTDecoderLayer(config, layer_idx=i) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, past_key_value: transformers.Cache | None = None) -> Tensor:
        for layer in self.layers:
            # Encoder call: No RoPE indices, No Mask (Full Attention)
            x = layer(
                x,
                attention_mask=None,
                cache=past_key_value,
                attn_kwargs=None # No RoPE indices -> No Rotation applied
            )
        x = self.norm(x)
        return x


class ARTDecoder(nn.Module):
    def __init__(self, config: ARTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ARTDecoderLayer(config, layer_idx=i) for i in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, attn_mask=None, cache=None, attn_kwargs=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask, cache=cache, attn_kwargs=attn_kwargs)
        x = self.norm(x)
        return x


class ARTDecoderLayer(nn.Module):
    def __init__(self, config: ARTConfig, layer_idx: int):
        super().__init__()
        # Shared Attention Class
        self.self_attn = ARTMaskAttention(config, layer_idx=layer_idx)
        
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[transformers.Cache] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        
        residual = x
        x = self.norm1(x)
        
        # Shared Attention Logic
        x = self.self_attn(
            x, 
            attn_mask=attention_mask, 
            cache=cache, 
            attn_kwargs=attn_kwargs
        )
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)
        
        return x


class ARTMaskAttention(nn.Module):
    """
    Unified Attention Block.
    - If attn_kwargs is None (Encoder): Standard Self-Attention, No RoPE.
    - If attn_kwargs is Set (Decoder): RoPE + Cache indexing.
    - Expects BATCH FIRST inputs (B, L, D).
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEncoding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[transformers.Cache] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        
        # 1. Input is [B, L, D] (Batch First)
        bsz, q_len, _ = hidden_states.size()

        # 2. Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 3. Reshape for SDPA: [B, H, L, D]
        # view: [B, L, H, D] -> transpose: [B, H, L, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. RoPE (Only if kwargs provided)
        if attn_kwargs is not None:
            q_pos_ids = attn_kwargs.get('query_position_indices')
            k_pos_ids = attn_kwargs.get('key_position_indices')
            
            if q_pos_ids is not None:
                cos_q, sin_q = self.rotary_emb(q_pos_ids, device=query_states.device, dtype=query_states.dtype)
                query_states = apply_rotary_pos_emb(query_states, cos_q, sin_q)
            
            if k_pos_ids is not None:
                if q_pos_ids is not None and torch.equal(q_pos_ids, k_pos_ids):
                    key_states = apply_rotary_pos_emb(key_states, cos_q, sin_q)
                else:
                    cos_k, sin_k = self.rotary_emb(k_pos_ids, device=key_states.device, dtype=key_states.dtype)
                    key_states = apply_rotary_pos_emb(key_states, cos_k, sin_k)

        # 5. Cache
        if cache is not None:
            cache_args = attn_kwargs.get('cache_kwargs', {}) if attn_kwargs else {}
            # Update cache and get FULL sequence
            key_states, value_states = cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_args
            )

        # 6. Attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=attn_mask, 
            dropout_p=self.config.dropout if self.training else 0.0
        )

        # 7. Reshape Back: [B, L, D]
        # transpose: [B, L, H, D] -> view: [B, L, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        
        return self.out_proj(attn_output)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids, device=None, dtype=None):
        if device is None: device = self.inv_freq.device
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        if dtype is not None: return cos.to(dtype=dtype), sin.to(dtype=dtype)
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    if cos.ndim == 3: cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

def get_activation_fn(activation: str) -> Callable:
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

# Keep existing embedding utils
class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed