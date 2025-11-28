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
        actions_hat = self.model(batch)

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
        for i, layer_cache in enumerate(cache):
            print(f"Layer {i}:")
            print(f"  Key shape: {layer_cache[0].shape}")
            print(f"  Value shape: {layer_cache[1].shape}")

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
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ARTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        
        # MODIFIED: Pass layer_idx to ACTEncoderLayer
        self.layers = nn.ModuleList([
            ARTEncoderLayer(config, layer_idx=i) for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, 
        x: Tensor, 
        pos_embed: Tensor | None = None, 
        key_padding_mask: Tensor | None = None,
        past_key_value: transformers.Cache | None = None # MODIFIED: Add cache arg
    ) -> Tensor:
        for layer in self.layers:
            # MODIFIED: Pass cache to layer
            x = layer(
                x, 
                pos_embed=pos_embed, 
                key_padding_mask=key_padding_mask,
                past_key_value=past_key_value 
            )
        x = self.norm(x)
        return x


class ARTEncoderLayer(nn.Module):
    def __init__(self, config: ARTConfig, layer_idx: int = 0):
        super().__init__()
        # MODIFIED: Replace nn.MultiheadAttention with ACTEncoderAttention
        self.self_attn = ARTEncoderAttention(config, layer_idx=layer_idx)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self, 
        x, 
        pos_embed: Tensor | None = None, 
        key_padding_mask: Tensor | None = None,
        past_key_value: transformers.Cache | None = None # MODIFIED: Add cache arg
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        
        # Logic remains: add pos_embed to q and k, but not v
        q = k = x if pos_embed is None else x + pos_embed
        
        # MODIFIED: Call custom attention with explicit q, k, v and cache
        x = self.self_attn(
            query=q, 
            key=k, 
            value=x, # Value does not get pos_embed
            key_padding_mask=key_padding_mask,
            past_key_value=past_key_value
        )
        
        # Rest remains the same
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class ARTDecoder(nn.Module):
    def __init__(self, config: ARTConfig):
        super().__init__()
        self.config = config
        self.dim_model = config.dim_model
        
        self.layers = nn.ModuleList([
            ARTDecoderLayer(config, layer_idx=i) for i in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(self.dim_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[transformers.Cache] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attn_mask,
                cache=cache,
                attn_kwargs=attn_kwargs
            )
            
        x = self.norm(x)
        return x


class ARTDecoderLayer(nn.Module):
    def __init__(self, config: ARTConfig, layer_idx: int):
        super().__init__()
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
        
        # Pass kwargs (including position indices and cache settings) to attention
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
    Causal Self-Attention with RoPE and explicit KV Cache control.
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
        
        bsz, q_len, _ = hidden_states.size()

        # 1. Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE
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

        # 4. Cache
        if cache is not None:
            # Extract specific cache_kwargs if provided (e.g. for StaticCache indexing)
            cache_args = attn_kwargs.get('cache_kwargs', {}) if attn_kwargs else {}
            
            # print k,v shapes before update
            print(f"Before Cache Update - Key shape: {key_states.shape}, Value shape: {value_states.shape}")
            
            key_states, value_states = cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_args
            )

        # 5. Attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=attn_mask, 
            dropout_p=self.config.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.out_proj(attn_output)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # Pre-compute theta (1/frequency)
        # We register this as a buffer so it saves with state_dict and moves to device automatically
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad() # CRITICAL: No gradients needed for position calc
    def forward(self, position_ids: torch.LongTensor, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Cos and Sin embeddings for the given position IDs.
        
        Args:
            position_ids: [Batch, Seq_Len] or [1, Seq_Len]
            device: Target device (optional, usually inferred from inputs in higher layers)
            dtype: Target dtype (optional)
        """
        # 1. Use the device of the buffer (which tracks the model device)
        if device is None:
            device = self.inv_freq.device
            
        # 2. Expand frequencies to match batch size
        # inv_freq: [Dim/2] -> [1, 1, Dim/2]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # 3. Expand position IDs
        # position_ids: [Batch, Seq] -> [Batch, Seq, 1]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # 4. Matrix Mult: Outer product of Positions and Frequencies
        # Result: [Batch, Seq, Dim/2]
        # We force float32 here for numerical stability during the rotation calc
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
        # 5. Cast to target dtype (e.g., bfloat16) only at the end
        if dtype is not None:
            return cos.to(dtype=dtype), sin.to(dtype=dtype)
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies the computed Cos/Sin to query or key `x`.
    x: [Batch, Heads, Seq_Len, Head_Dim]
    cos, sin: [Batch, 1, Seq_Len, Head_Dim]
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Ensure broadcasting dimensions match (Batch, 1, Seq, Dim)
    # This handles the case where cos/sin output is [B, S, D]
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    return (x * cos) + (rotate_half(x) * sin)



def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

class ARTEncoderAttention(nn.Module):
    """
    Decomposed MultiHeadAttention for the Encoder to support KV Caching.
    Does NOT use RoPE. Uses standard absolute position embeddings passed via Q/K inputs.
    """
    def __init__(self, config: ARTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.dim_model
        self.num_heads = config.n_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx 
        
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {self.embed_dim} and {self.num_heads})")

        # Projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        past_key_value: transformers.Cache | None = None, 
    ) -> Tensor:
        # Input shape: (Seq_Len, Batch, Dim) - adhering to original ACT convention
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        # 1. Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape to (Batch, Heads, Seq_Len, Head_Dim) for Cache & SDPA
        # Transpose (Seq, Batch, Dim) -> (Batch, Seq, Dim) first
        q = q.transpose(0, 1).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.transpose(0, 1).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(0, 1).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Update Cache (The main reason we are doing this)
        # Note: The encoder typically processes the whole prefix at once, so we store the full K/V.
        if past_key_value is not None:
            # We don't usually need cache_position for the encoder as it's not autoregressive,
            # but DynamicCache expects the call.
            k, v = past_key_value.update(k, v, self.layer_idx)

        # 4. Attention
        # Standard scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_padding_mask, # Note: PyTorch SDPA handles broadcasting for mask
            dropout_p=self.config.dropout if self.training else 0.0
        )

        # 5. Reshape back to (Seq_Len, Batch, Dim)
        # (Batch, Heads, Seq, Head_Dim) -> (Batch, Seq, Dim) -> (Seq, Batch, Dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = attn_output.transpose(0, 1)

        # 6. Output Projection
        return self.out_proj(attn_output)