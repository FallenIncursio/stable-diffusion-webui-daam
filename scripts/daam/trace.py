from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Type, Any, Dict, Optional
import math
from modules.devices import device

try:
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from ldm.modules.diffusionmodules.openaimodel import UNetModel
    from ldm.modules.attention import CrossAttention, default, exists
except ModuleNotFoundError:
    from backend.nn.unet import CrossAttention, default, exists
    LatentDiffusion = Any
    UNetModel = Any

import numba
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

from .labels import COCO80_LABELS
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator
from .utils import PromptAnalyzer


__all__ = ['trace', 'DiffusionHeatMapHooker', 'HeatMap', 'MmDetectHeatMap']

HeatMapByFactor = dict[int, list[torch.Tensor]]
HeatMapByBatch = dict[int, HeatMapByFactor]


def _resolve_diffusion_model(model):
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        return model.model.diffusion_model

    forge_objects = getattr(model, "forge_objects", None)
    if forge_objects is not None:
        unet = getattr(forge_objects, "unet", None)
        if unet is not None and hasattr(unet, "model") and hasattr(unet.model, "diffusion_model"):
            return unet.model.diffusion_model

    raise AttributeError(f"Unsupported model structure for DAAM tracing: {type(model)}")


class UNetForwardHooker(ObjectHooker[UNetModel]):
    def __init__(
        self,
        module: UNetModel,
        heat_maps_cond: HeatMapByBatch,
        heat_maps_uncond: HeatMapByBatch,
        runtime_state: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(module)
        self.all_heat_maps_cond: list[HeatMapByBatch] = []
        self.all_heat_maps_uncond: list[HeatMapByBatch] = []
        self.all_heat_maps = self.all_heat_maps_cond  # backward-compatible alias
        self.heat_maps_cond = heat_maps_cond
        self.heat_maps_uncond = heat_maps_uncond
        self.runtime_state = runtime_state if runtime_state is not None else {}

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    def _unhook_impl(self):
        pass

    def _forward(hk_self, self, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options", None)
        if transformer_options is None and len(args) >= 6 and isinstance(args[5], dict):
            transformer_options = args[5]

        if isinstance(transformer_options, dict):
            hk_self.runtime_state["cond_or_uncond"] = transformer_options.get("cond_or_uncond", None)
        else:
            hk_self.runtime_state["cond_or_uncond"] = None

        super_return = hk_self.monkey_super('forward', *args, **kwargs)
        hk_self.all_heat_maps_cond.append(deepcopy(hk_self.heat_maps_cond))
        hk_self.all_heat_maps_uncond.append(deepcopy(hk_self.heat_maps_uncond))
        hk_self.heat_maps_cond.clear()
        hk_self.heat_maps_uncond.clear()

        return super_return


class HeatMap:
    def __init__(self, prompt_analyzer: PromptAnalyzer, prompt: str, heat_maps: torch.Tensor):
        self.prompt_analyzer = prompt_analyzer.create(prompt)
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str, word_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        merge_idxs, _ = self.prompt_analyzer.calc_word_indecies(word)
        # print("merge_idxs", merge_idxs)
        if len(merge_idxs) == 0:
            return None
        
        return self.heat_maps[merge_idxs].mean(0)


class MmDetectHeatMap:
    def __init__(self, pred_file: str | Path, threshold: float = 0.95):
        @numba.njit
        def _compute_mask(masks: np.ndarray, bboxes: np.ndarray):
            x_any = np.any(masks, axis=1)
            y_any = np.any(masks, axis=2)
            num_masks = len(bboxes)

            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                bboxes[idx, :4] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

        pred_file = Path(pred_file)
        self.word_masks: Dict[str, torch.Tensor] = {}
        bbox_result, masks = torch.load(pred_file)
        label_parts = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels_np = np.concatenate(label_parts)
        bboxes = np.vstack(bbox_result)

        if masks is not None and bboxes[:, :4].sum() == 0:
            _compute_mask(masks, bboxes)
            scores = bboxes[:, -1]
            inds = scores > threshold
            labels_np = labels_np[inds]
            masks = masks[inds, ...]

            for lbl, mask in zip(labels_np, masks):
                key = COCO80_LABELS[lbl]
                mask_tensor = torch.from_numpy(mask).bool()
                if key in self.word_masks:
                    self.word_masks[key] = torch.logical_or(self.word_masks[key].bool(), mask_tensor)
                else:
                    self.word_masks[key] = mask_tensor

            self.word_masks = {k: v.float() for k, v in self.word_masks.items()}

    def compute_word_heat_map(self, word: str) -> torch.Tensor:
        return self.word_masks[word]


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
        self,
        model: LatentDiffusion,
        heigth: int,
        width: int,
        context_size: int = 77,
        weighted: bool = False,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
    ):
        diffusion_model = _resolve_diffusion_model(model)
        heat_maps_cond: HeatMapByBatch = defaultdict(lambda: defaultdict(list))  # batch index, factor, attention
        heat_maps_uncond: HeatMapByBatch = defaultdict(lambda: defaultdict(list))  # batch index, factor, attention
        runtime_state = {"cond_or_uncond": None}
        modules: list[Any] = [
            UNetCrossAttentionHooker(
                x,
                heigth,
                width,
                heat_maps_cond,
                heat_maps_uncond,
                context_size=context_size,
                weighted=weighted,
                head_idx=head_idx,
                runtime_state=runtime_state,
            )
            for x in UNetCrossAttentionLocator().locate(diffusion_model, layer_idx)
        ]
        self.forward_hook = UNetForwardHooker(diffusion_model, heat_maps_cond, heat_maps_uncond, runtime_state=runtime_state)
        modules.append(self.forward_hook)
        
        self.height = heigth
        self.width = width
        self.model = model
        self.last_prompt = ''
        
        super().__init__(modules)

        

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps_cond
    
    def reset(self):
        for module in self.module:
            if hasattr(module, "reset"):
                module.reset()
        self.forward_hook.all_heat_maps_cond.clear()
        self.forward_hook.all_heat_maps_uncond.clear()
        return

    def compute_global_heat_map(
        self,
        prompt_analyzer,
        prompt,
        batch_index,
        time_weights: Optional[list[float]] = None,
        time_idx: Optional[int] = None,
        last_n: Optional[int] = None,
        first_n: Optional[int] = None,
        factors: Optional[list[int] | set[int]] = None,
        guidance_mode: str = "cond",
    ) -> Optional[HeatMap]:
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for.
            time_weights: The weights to apply to each time step. If None, all time steps are weighted equally.
            time_idx: The time step to compute the heat map for. If None, the heat map is computed for all time steps.
                Mutually exclusive with `last_n` and `first_n`.
            last_n: The number of last n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            first_n: The number of first n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
        """
        if guidance_mode == "uncond":
            all_heat_maps = self.forward_hook.all_heat_maps_uncond
        else:
            all_heat_maps = self.forward_hook.all_heat_maps_cond

        if len(all_heat_maps) == 0:
            return None
        
        if time_weights is None:
            time_weights = [1.0] * len(all_heat_maps)
        if time_idx is not None:
            heat_maps = [all_heat_maps[time_idx]]
        else:
            heat_maps = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps
            heat_maps = heat_maps[:first_n] if first_n is not None else heat_maps
            

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []
        
        for batch_to_heat_maps in heat_maps:
            
            if not (batch_index in batch_to_heat_maps):
                continue    
            
            merge_list = []
                 
            factors_to_heat_maps = batch_to_heat_maps[batch_index]

            for k, heat_map in factors_to_heat_maps.items():
                # heat_map shape: (tokens, 1, height, width)
                # each v is a heat map tensor for a layer of factor size k across the tokens
                if k in factors:
                    merge_list.append(torch.stack(heat_map, 0).mean(0))

            if  len(merge_list) > 0:
               all_merges.append(merge_list)

        if len(all_merges) == 0:
            return None

        maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        maps = maps.sum(0).to(device).sum(2).sum(0)

        return HeatMap(prompt_analyzer, prompt, maps)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(
        self,
        module: CrossAttention,
        img_height: int,
        img_width: int,
        heat_maps_cond: HeatMapByBatch,
        heat_maps_uncond: HeatMapByBatch,
        context_size: int = 77,
        weighted: bool = False,
        head_idx: Optional[int] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(module)
        self.heat_maps = heat_maps_cond
        self.heat_maps_uncond = heat_maps_uncond
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.img_height = img_height
        self.img_width =  img_width
        self.calledCount = 0
        self.cond_or_uncond = None
        self.current_batch_size = None
        self.runtime_state = runtime_state if runtime_state is not None else {}
        
    def reset(self):
        self.heat_maps.clear()
        self.heat_maps_uncond.clear()
        self.calledCount = 0
        self.cond_or_uncond = None
        self.current_batch_size = None
        self.runtime_state["cond_or_uncond"] = None

    def _infer_spatial_hw(self, token_count: int):
        if token_count <= 0:
            return 1, 1

        target_ratio = self.img_width / max(self.img_height, 1)
        best_h, best_w = token_count, 1
        best_err = float("inf")
        limit = int(math.sqrt(token_count)) + 1

        for h in range(1, limit):
            if token_count % h != 0:
                continue

            w = token_count // h
            for hh, ww in ((h, w), (w, h)):
                err = abs((ww / max(hh, 1)) - target_ratio)
                if err < best_err:
                    best_err = err
                    best_h, best_w = hh, ww

        return best_h, best_w
        
    @torch.no_grad()
    def _up_sample_attn(self, x: torch.Tensor, value: torch.Tensor, factor: int, method: str = 'bicubic') -> torch.Tensor:
        # x shape: (heads, height * width, tokens)
        """
        Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
        Diffusion model.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.
            method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

        Returns:
            `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
        """
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)
        
        h, w = self._infer_spatial_hw(x.size(1))
        
        h_fix = w_fix = 64
        if h >= w:
            w_fix = max(1, int((w * h_fix) / h))
        else:
            h_fix = max(1, int((h * w_fix) / w))
                
        maps: list[torch.Tensor] = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = 1

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(h_fix, w_fix), mode='bicubic')
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1))

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps_tensor = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        
        if self.head_idx:
            maps_tensor = maps_tensor[:, self.head_idx:self.head_idx+1, :, :]

        return (weights * maps_tensor).sum(1, keepdim=True).cpu()
    
    def _forward(hk_self, self, x, context=None, value=None, mask=None, additional_tokens=None, transformer_options=None, **kwargs):
        n_tokens_to_mask = 0
        
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        
        hk_self.calledCount += 1
        if isinstance(transformer_options, dict):
            hk_self.cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        else:
            hk_self.cond_or_uncond = hk_self.runtime_state.get("cond_or_uncond", None)

        batch_size, sequence_length, _ = x.shape
        hk_self.current_batch_size = batch_size
        h = self.heads
        scale = getattr(self, "scale", (self.dim_head ** -0.5))

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
        else:
            v = self.to_v(context)
        
        dim = q.shape[-1]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        
        out = hk_self._hooked_attention(self, q, k, v, batch_size, sequence_length, dim, n_tokens_to_mask=n_tokens_to_mask)
        
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        
        return self.to_out(out)

    def _resolve_guidance_batch_index(self, batch_index: int):
        cond_or_uncond = self.cond_or_uncond
        if isinstance(cond_or_uncond, (list, tuple)):
            markers = list(cond_or_uncond)
            batch_size = self.current_batch_size if isinstance(self.current_batch_size, int) and self.current_batch_size > 0 else None

            # Some Forge paths pass compact markers, e.g. [UNCOND, COND], while attention
            # runs on expanded batch slots. Expand markers to per-slot resolution.
            if batch_size is not None and len(markers) > 0 and len(markers) < batch_size and batch_size % len(markers) == 0:
                repeat = batch_size // len(markers)
                expanded = []
                for marker in markers:
                    expanded.extend([marker] * repeat)
                markers = expanded

            if batch_index >= len(markers):
                return None, None
            marker_at_index = markers[batch_index]
            if marker_at_index not in (0, 1):
                return None, None

            # Forge marks COND=0, UNCOND=1.
            match_marker = marker_at_index
            guidance = "cond" if match_marker == 0 else "uncond"
            guidance_position = 0
            for i, marker in enumerate(markers):
                if marker == match_marker:
                    if i == batch_index:
                        return guidance, guidance_position
                    guidance_position += 1
            return None, None

        # Legacy fallback: old A1111-style invocation commonly packs [uncond, cond].
        batch_size = self.current_batch_size if isinstance(self.current_batch_size, int) and self.current_batch_size > 0 else None
        if batch_size is not None and batch_size % 2 == 0:
            cond_start = batch_size // 2
            if batch_index >= cond_start:
                return "cond", batch_index - cond_start
            return "uncond", batch_index

        # Last resort fallback: odd call=cond, even call=uncond.
        if self.calledCount % 2 == 1:
            return "cond", batch_index
        return "uncond", batch_index
    
    ### forward implemetation of diffuser CrossAttention
    # def forward(self, hidden_states, context=None, mask=None):
    #     batch_size, sequence_length, _ = hidden_states.shape

    #     query = self.to_q(hidden_states)
    #     context = context if context is not None else hidden_states
    #     key = self.to_k(context)
    #     value = self.to_v(context)

    #     dim = query.shape[-1]

    #     query = self.reshape_heads_to_batch_dim(query)
    #     key = self.reshape_heads_to_batch_dim(key)
    #     value = self.reshape_heads_to_batch_dim(value)

    #     # TODO(PVP) - mask is currently never used. Remember to re-implement when used

    #     # attention, what we cannot get enough of
    #     if self._use_memory_efficient_attention_xformers:
    #         hidden_states = self._memory_efficient_attention_xformers(query, key, value)
    #         # Some versions of xformers return output in fp32, cast it back to the dtype of the input
    #         hidden_states = hidden_states.to(query.dtype)
    #     else:
    #         if self._slice_size is None or query.shape[0] // self._slice_size == 1:
    #             hidden_states = self._attention(query, key, value)
    #         else:
    #             hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

    #     # linear proj
    #     hidden_states = self.to_out[0](hidden_states)
    #     # dropout
    #     hidden_states = self.to_out[1](hidden_states)
    #     return hidden_states

    def _hooked_attention(hk_self, self, query, key, value, batch_size, sequence_length, dim, use_context: bool = True, n_tokens_to_mask: int = 0):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
            batch_size (`int`): the batch size
            use_context (`bool`): whether to check if the resulting attention slices are between the words and the image
        """
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = hidden_states.shape[0] // batch_size # self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        
        def calc_factor_base(w, h):
            z = max(w/64, h/64)
            factor_b = min(w, h) * z
            return factor_b
        
        factor_base = calc_factor_base(hk_self.img_width, hk_self.img_height)
        scale = getattr(self, "scale", (self.dim_head ** -0.5))
        for batch_index in range(hidden_states.shape[0] // slice_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size
            attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * scale
            )
            attn_slice = attn_slice.softmax(-1)
            map_attn_slice = attn_slice[:, n_tokens_to_mask:, :] if n_tokens_to_mask > 0 else attn_slice
            factor = int(math.sqrt(factor_base // max(map_attn_slice.shape[1], 1)))
            hid_states = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])            

            guidance_mode, target_batch_index = hk_self._resolve_guidance_batch_index(batch_index)

            aligned_map_attn_slice = (
                hk_self._align_context_tokens(map_attn_slice)
                if use_context and target_batch_index is not None and guidance_mode in ("cond", "uncond")
                else None
            )

            if use_context and target_batch_index is not None and aligned_map_attn_slice is not None and aligned_map_attn_slice.shape[1] > 0:
                if factor >= 1:
                    maps = hk_self._up_sample_attn(aligned_map_attn_slice, value[start_idx:end_idx], factor)
                    if guidance_mode == "uncond":
                        hk_self.heat_maps_uncond[target_batch_index][factor].append(maps)
                    else:
                        hk_self.heat_maps[target_batch_index][factor].append(maps)

            hidden_states[start_idx:end_idx] = hid_states

        # reshape hidden_states
        hidden_states = hk_self.reshape_batch_dim_to_heads(self, hidden_states)
        return hidden_states

    def _align_context_tokens(self, map_attn_slice: torch.Tensor) -> Optional[torch.Tensor]:
        target = self.context_size if isinstance(self.context_size, int) else 0
        if target <= 0:
            return map_attn_slice

        context_tokens = map_attn_slice.shape[-1]
        if context_tokens == target:
            return map_attn_slice
        if context_tokens < target:
            return None

        # ReForge can repeat cross-attn context to an LCM length before concat.
        # Sum repeated groups back into the target token count.
        if context_tokens % target == 0:
            repeat = context_tokens // target
            if repeat > 1:
                new_shape = map_attn_slice.shape[:-1] + (repeat, target)
                return map_attn_slice.reshape(new_shape).sum(-2)

        return map_attn_slice[..., :target]
    
    def reshape_batch_dim_to_heads(hk_self, self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
