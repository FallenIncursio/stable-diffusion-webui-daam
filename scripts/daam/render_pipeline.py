from __future__ import annotations

import os

import torch
from PIL import Image

from scripts.daam import utils


class RenderPipelineMixin:
    def _get_prompt_analyzer_for_batch(self, batch_pos: int, prompt_text: str):
        analyzers = getattr(self, "prompt_analyzers", None)
        if isinstance(analyzers, list) and len(analyzers) > 0:
            if 0 <= batch_pos < len(analyzers):
                return analyzers[batch_pos]
            return analyzers[0]

        analyzer = getattr(self, "prompt_analyzer", None)
        if analyzer is not None:
            return analyzer

        text_engine = getattr(self, "text_engine", None)
        if text_engine is None or not prompt_text:
            return None
        return utils.PromptAnalyzer(text_engine, prompt_text)

    def _get_negative_prompt_analyzer_for_batch(self, batch_pos: int, prompt_text: str):
        analyzers = getattr(self, "negative_prompt_analyzers", None)
        if isinstance(analyzers, list) and len(analyzers) > 0:
            if 0 <= batch_pos < len(analyzers):
                return analyzers[batch_pos]
            return analyzers[0]

        if not prompt_text:
            return None

        cache = getattr(self, "negative_prompt_analyzer_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self.negative_prompt_analyzer_cache = cache

        if prompt_text in cache:
            return cache[prompt_text]

        text_engine = getattr(self, "text_engine", None)
        if text_engine is None:
            return None

        analyzer = utils.PromptAnalyzer(text_engine, prompt_text)
        cache[prompt_text] = analyzer
        return analyzer

    @staticmethod
    def _compute_word_heat_map_from_candidates(global_heat_map, candidates):
        if global_heat_map is None:
            return None, None
        for candidate in candidates:
            heat_map = global_heat_map.compute_word_heat_map(candidate)
            if heat_map is not None:
                return heat_map, candidate
        return None, None

    @staticmethod
    def _prepare_delta_maps(positive_map, negative_map):
        if positive_map is None and negative_map is None:
            return None, None

        reference = positive_map if positive_map is not None else negative_map
        if reference is None:
            return None, None

        if positive_map is None:
            positive_map = torch.zeros_like(reference)
        if negative_map is None:
            negative_map = torch.zeros_like(reference)
        return positive_map, negative_map

    @classmethod
    def _compute_delta_heat_map(cls, positive_map, negative_map):
        positive_map, negative_map = cls._prepare_delta_maps(positive_map, negative_map)
        if positive_map is None:
            return None

        delta = torch.clamp(positive_map - negative_map, min=0.0)
        max_value = float(delta.max().item()) if delta.numel() > 0 else 0.0
        if max_value > 0:
            delta = delta / max_value
        return delta

    @classmethod
    def _compute_abs_delta_heat_map(cls, positive_map, negative_map):
        positive_map, negative_map = cls._prepare_delta_maps(positive_map, negative_map)
        if positive_map is None:
            return None

        delta = torch.abs(positive_map - negative_map)
        max_value = float(delta.max().item()) if delta.numel() > 0 else 0.0
        if max_value <= 0:
            return None
        return delta / max_value

    @classmethod
    def _compute_signed_delta_heat_map(cls, positive_map, negative_map):
        positive_map, negative_map = cls._prepare_delta_maps(positive_map, negative_map)
        if positive_map is None:
            return None

        signed = positive_map - negative_map
        max_abs = float(signed.abs().max().item()) if signed.numel() > 0 else 0.0
        if max_abs <= 0:
            return None
        signed = signed / max_abs
        # Map [-1, 1] -> [0, 1] so the existing heatmap renderer can colorize it.
        signed = (signed + 1.0) * 0.5
        return torch.clamp(signed, min=0.0, max=1.0)

    def _resolve_focus_kwargs(self, tracer, focus_override=None):
        time_focus = self._normalize_time_focus(
            focus_override if focus_override is not None else getattr(self, "time_focus", self.TIME_FOCUS_ALL)
        )
        if time_focus == self.TIME_FOCUS_ALL:
            return {}

        all_heat_maps = getattr(tracer, "all_heat_maps", None) or []
        total_steps = len(all_heat_maps)
        if total_steps <= 0:
            return {}

        third = max(1, total_steps // 3)
        if time_focus == self.TIME_FOCUS_EARLY:
            return {"first_n": third}
        if time_focus == self.TIME_FOCUS_MID:
            return {"time_idx": max(0, min(total_steps - 1, total_steps // 2))}
        if time_focus == self.TIME_FOCUS_LATE:
            return {"last_n": third}
        return {}

    def _resolve_focus_targets(self, tracer):
        enable_time_focus = bool(getattr(self, "enable_time_focus", False))
        if not enable_time_focus:
            return [(self.TIME_FOCUS_ALL, {})]

        focus_mode = self._normalize_time_focus(getattr(self, "time_focus", self.TIME_FOCUS_ALL))
        if focus_mode == self.TIME_FOCUS_TRIPLET:
            return [
                (self.TIME_FOCUS_EARLY, self._resolve_focus_kwargs(tracer, self.TIME_FOCUS_EARLY)),
                (self.TIME_FOCUS_MID, self._resolve_focus_kwargs(tracer, self.TIME_FOCUS_MID)),
                (self.TIME_FOCUS_LATE, self._resolve_focus_kwargs(tracer, self.TIME_FOCUS_LATE)),
            ]

        return [(focus_mode, self._resolve_focus_kwargs(tracer, focus_mode))]

    def before_image_saved(self, params):
        if self._is_grid_save(params):
            return

        batch_pos = self._resolve_batch_pos(params)
        if batch_pos < 0:
            return

        if self.tracers is not None and len(self.attentions) > 0:
            diagnostics_entries = []
            for i, tracer in enumerate(self.tracers):
                with torch.no_grad():
                    effective_prompt = self._resolve_effective_prompt(params, batch_pos)
                    styled_prompt = self._canonicalize_prompt_for_daam(effective_prompt)
                    prompt_analyzer = self._get_prompt_analyzer_for_batch(batch_pos, styled_prompt)
                    if prompt_analyzer is None:
                        continue
                    effective_negative_prompt = self._resolve_effective_negative_prompt(params, batch_pos)
                    styled_negative_prompt = self._canonicalize_prompt_for_daam(effective_negative_prompt)
                    negative_prompt_analyzer = self._get_negative_prompt_analyzer_for_batch(batch_pos, styled_negative_prompt)
                    influence_mode = self._normalize_influence_mode(
                        getattr(self, "influence_mode", self.INFLUENCE_POSITIVE)
                    )

                    if i not in self.heatmap_images:
                        self.heatmap_images[i] = []

                    heatmap_images = []
                    for focus_label, focus_kwargs in self._resolve_focus_targets(tracer):
                        positive_global_heat_map = None
                        negative_global_heat_map = None
                        try:
                            if influence_mode == self.INFLUENCE_NEGATIVE:
                                if negative_prompt_analyzer is not None:
                                    negative_global_heat_map = tracer.compute_global_heat_map(
                                        negative_prompt_analyzer,
                                        styled_negative_prompt,
                                        batch_pos,
                                        guidance_mode="uncond",
                                        **focus_kwargs,
                                    )
                            elif influence_mode in (
                                self.INFLUENCE_DELTA,
                                self.INFLUENCE_DELTA_SIGNED,
                                self.INFLUENCE_DELTA_ABS,
                            ):
                                positive_global_heat_map = tracer.compute_global_heat_map(
                                    prompt_analyzer, styled_prompt, batch_pos, guidance_mode="cond", **focus_kwargs
                                )
                                if negative_prompt_analyzer is not None:
                                    negative_global_heat_map = tracer.compute_global_heat_map(
                                        negative_prompt_analyzer,
                                        styled_negative_prompt,
                                        batch_pos,
                                        guidance_mode="uncond",
                                        **focus_kwargs,
                                    )
                            else:
                                positive_global_heat_map = tracer.compute_global_heat_map(
                                    prompt_analyzer, styled_prompt, batch_pos, guidance_mode="cond", **focus_kwargs
                                )
                        except Exception:
                            continue

                        if (
                            influence_mode == self.INFLUENCE_POSITIVE
                            and positive_global_heat_map is None
                        ):
                            continue
                        if (
                            influence_mode == self.INFLUENCE_NEGATIVE
                            and negative_global_heat_map is None
                        ):
                            continue
                        if (
                            influence_mode
                            in (
                                self.INFLUENCE_DELTA,
                                self.INFLUENCE_DELTA_SIGNED,
                                self.INFLUENCE_DELTA_ABS,
                            )
                            and positive_global_heat_map is None
                            and negative_global_heat_map is None
                        ):
                            continue

                        for raw_attention in self.attentions:
                            resolve_base_prompt = (
                                styled_negative_prompt
                                if influence_mode == self.INFLUENCE_NEGATIVE and styled_negative_prompt
                                else styled_prompt
                            )
                            attention, attn_candidates = self._resolve_attention_and_candidates(
                                raw_attention, resolve_base_prompt
                            )

                            img_size = params.image.size
                            attn_caption = self.attn_captions[i] if i < len(self.attn_captions) else ""
                            caption_tags = []
                            focus_caption = focus_label if getattr(self, "enable_time_focus", False) else ""
                            if focus_caption:
                                caption_tags.append(focus_caption)
                            if influence_mode != self.INFLUENCE_POSITIVE:
                                caption_tags.append(influence_mode.split(" ")[0])
                            mode_caption = "|".join(caption_tags)
                            caption = (
                                attention
                                + (" " + attn_caption if attn_caption else "")
                                + (f" [{mode_caption}]" if mode_caption else "")
                                if not self.hide_caption
                                else None
                            )

                            heat_map = None
                            matched_candidate = None
                            reason = "ok"
                            delta_detail = None

                            if influence_mode == self.INFLUENCE_NEGATIVE:
                                heat_map, matched_candidate = self._compute_word_heat_map_from_candidates(
                                    negative_global_heat_map, attn_candidates
                                )
                                if heat_map is None:
                                    reason = self._diagnose_missing_heatmap(
                                        raw_attention,
                                        attention,
                                        attn_candidates,
                                        styled_negative_prompt,
                                        negative_prompt_analyzer,
                                    )
                            elif influence_mode in (
                                self.INFLUENCE_DELTA,
                                self.INFLUENCE_DELTA_SIGNED,
                                self.INFLUENCE_DELTA_ABS,
                            ):
                                pos_map, pos_candidate = self._compute_word_heat_map_from_candidates(
                                    positive_global_heat_map, attn_candidates
                                )
                                neg_map, neg_candidate = self._compute_word_heat_map_from_candidates(
                                    negative_global_heat_map, attn_candidates
                                )
                                if influence_mode == self.INFLUENCE_DELTA_SIGNED:
                                    heat_map = self._compute_signed_delta_heat_map(pos_map, neg_map)
                                elif influence_mode == self.INFLUENCE_DELTA_ABS:
                                    heat_map = self._compute_abs_delta_heat_map(pos_map, neg_map)
                                else:
                                    heat_map = self._compute_delta_heat_map(pos_map, neg_map)
                                matched_candidate = pos_candidate if pos_candidate is not None else neg_candidate
                                if heat_map is None:
                                    pos_reason = self._diagnose_missing_heatmap(
                                        raw_attention,
                                        attention,
                                        attn_candidates,
                                        styled_prompt,
                                        prompt_analyzer,
                                    )
                                    neg_reason = self._diagnose_missing_heatmap(
                                        raw_attention,
                                        attention,
                                        attn_candidates,
                                        styled_negative_prompt,
                                        negative_prompt_analyzer,
                                    )
                                    reason = "delta_missing_both"
                                else:
                                    pos_reason = "ok" if pos_map is not None else "positive_missing"
                                    neg_reason = "ok" if neg_map is not None else "negative_missing"
                                delta_detail = {
                                    "positive_matched": pos_map is not None,
                                    "negative_matched": neg_map is not None,
                                    "positive_candidate": pos_candidate,
                                    "negative_candidate": neg_candidate,
                                    "positive_reason": pos_reason,
                                    "negative_reason": neg_reason,
                                }
                            else:
                                heat_map, matched_candidate = self._compute_word_heat_map_from_candidates(
                                    positive_global_heat_map, attn_candidates
                                )
                                if heat_map is None:
                                    reason = self._diagnose_missing_heatmap(
                                        raw_attention,
                                        attention,
                                        attn_candidates,
                                        styled_prompt,
                                        prompt_analyzer,
                                    )

                            if heat_map is None:
                                print(
                                    f"No heatmaps for '{raw_attention}' "
                                    f"(resolved='{attention}', focus={focus_label}, mode={influence_mode}, reason={reason})"
                                )

                            if self.enable_diagnostics:
                                entry = {
                                    "layer": attn_caption or "ALL",
                                    "focus": focus_label,
                                    "influence_mode": influence_mode,
                                    "raw_attention": raw_attention,
                                    "resolved_attention": attention,
                                    "candidates": attn_candidates,
                                    "matched": heat_map is not None,
                                    "matched_candidate": matched_candidate,
                                    "reason": reason,
                                }
                                if delta_detail is not None:
                                    entry.update(delta_detail)
                                diagnostics_entries.append(entry)

                            heat_map_img = (
                                utils.expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                            )
                            img: Image.Image = utils.image_overlay_heat_map(
                                params.image,
                                heat_map_img,
                                alpha=self.alpha,
                                caption=caption,
                                image_scale=self.heatmap_image_scale,
                            )

                            fullfn_without_extension, extension = os.path.splitext(params.filename)
                            filename_attention = self._sanitize_filename_fragment(attention)
                            filename_caption = self._sanitize_filename_fragment(attn_caption) if attn_caption else ""
                            filename_focus = self._sanitize_filename_fragment(focus_caption) if focus_caption else ""
                            filename_influence = (
                                self._sanitize_filename_fragment(influence_mode)
                                if influence_mode != self.INFLUENCE_POSITIVE
                                else ""
                            )
                            full_filename = (
                                fullfn_without_extension
                                + "_"
                                + filename_attention
                                + ("_" + filename_focus if filename_focus else "")
                                + ("_" + filename_influence if filename_influence else "")
                                + ("_" + filename_caption if filename_caption else "")
                                + extension
                            )

                            if self.use_grid:
                                heatmap_images.append(img)
                            else:
                                heatmap_images.append(img)
                                if not self.dont_save_images:
                                    img.save(full_filename)

                    if len(heatmap_images) > 0:
                        self.heatmap_images[i] += heatmap_images

            if self.enable_diagnostics and len(diagnostics_entries) > 0:
                diagnostics_payload = {
                    "filename": params.filename,
                    "seed": self._extract_seed_from_filename(getattr(params, "filename", "")),
                    "batch_pos": batch_pos,
                    "enable_time_focus": self.enable_time_focus,
                    "time_focus": self.time_focus,
                    "influence_mode": self.influence_mode,
                    "prompt": styled_prompt,
                    "negative_prompt": styled_negative_prompt,
                    "entries": diagnostics_entries,
                }
                self._save_diagnostics(params, diagnostics_payload)

        self.heatmap_images = {j: self.heatmap_images[j] for j in self.heatmap_images.keys() if self.heatmap_images[j]}

        # if it is last batch pos, clear heatmaps
        if self.tracers is not None and batch_pos == params.p.batch_size - 1:
            for tracer in self.tracers:
                tracer.reset()

        self.saved_sample_count += 1

        return
