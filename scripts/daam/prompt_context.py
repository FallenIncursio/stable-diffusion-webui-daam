from __future__ import annotations

import os
import re
from typing import ClassVar

from scripts.daam.types import ParsedDaamFlags


class PromptContextMixin:
    saved_sample_count: int = 0
    _warned_output_overlap: ClassVar[bool] = False

    @classmethod
    def _normalize_time_focus(cls, value):
        if not isinstance(value, str):
            return cls.TIME_FOCUS_ALL
        normalized = value.strip().lower()
        if normalized == "disabled":
            return cls.TIME_FOCUS_ALL
        if normalized == "early":
            return cls.TIME_FOCUS_EARLY
        if normalized == "mid":
            return cls.TIME_FOCUS_MID
        if normalized == "late":
            return cls.TIME_FOCUS_LATE
        if normalized == "triplet":
            return cls.TIME_FOCUS_TRIPLET
        return cls.TIME_FOCUS_ALL

    @classmethod
    def _normalize_influence_mode(cls, value):
        if not isinstance(value, str):
            return cls.INFLUENCE_POSITIVE
        normalized = value.strip().lower()
        if normalized.startswith("negative"):
            return cls.INFLUENCE_NEGATIVE
        if normalized.startswith("signed delta") or ("signed" in normalized and "delta" in normalized):
            return cls.INFLUENCE_DELTA_SIGNED
        if normalized.startswith("abs delta") or normalized.startswith("absolute delta"):
            return cls.INFLUENCE_DELTA_ABS
        if normalized.startswith("delta"):
            return cls.INFLUENCE_DELTA
        return cls.INFLUENCE_POSITIVE

    def _parse_optional_daam_flags(self, enable_daam, extra_args, **kwargs):
        """
        Parse optional DAAM flags while keeping old API payloads valid.

        Supported positional forms:
        - [enable_daam]
        - [enable_daam, time_focus, enable_diagnostics, influence_mode]
        - [time_focus, enable_diagnostics, enable_daam] (legacy sprint test payload)
        """
        parsed_enable = bool(enable_daam)
        parsed_focus = self.TIME_FOCUS_ALL
        parsed_diagnostics = False
        parsed_enable_time_focus = None
        parsed_influence_mode = self.INFLUENCE_POSITIVE
        focus_explicit = False

        args = list(extra_args or [])

        # Compatibility for payload order where focus was inserted before enable.
        if isinstance(enable_daam, str):
            parsed_focus = enable_daam
            focus_explicit = True
            if len(args) >= 1 and isinstance(args[0], bool):
                parsed_diagnostics = args[0]
                args = args[1:]
            if len(args) >= 1 and isinstance(args[0], bool):
                parsed_enable = args[0]
                args = args[1:]

        # Preferred new order: [enable_time_focus, time_focus, enable_diagnostics, influence_mode]
        if len(args) >= 3 and isinstance(args[0], bool) and isinstance(args[1], str) and isinstance(args[2], bool):
            parsed_enable_time_focus = args[0]
            parsed_focus = args[1]
            focus_explicit = True
            parsed_diagnostics = args[2]
            args = args[3:]
            if len(args) >= 1 and isinstance(args[0], str):
                parsed_influence_mode = args[0]
                args = args[1:]

        # Legacy order from earlier tests: [time_focus, diagnostics, enable]
        elif len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], bool) and isinstance(args[2], bool):
            parsed_focus = args[0]
            focus_explicit = True
            parsed_diagnostics = args[1]
            parsed_enable = args[2]
            args = args[3:]
            if len(args) >= 1 and isinstance(args[0], str):
                parsed_influence_mode = args[0]
                args = args[1:]
        else:
            # Legacy order: [time_focus?, diagnostics?] after explicit enable_daam arg.
            if len(args) >= 1 and isinstance(args[0], str):
                parsed_focus = args[0]
                focus_explicit = True
                args = args[1:]
            if len(args) >= 1 and isinstance(args[0], bool):
                parsed_diagnostics = args[0]
                args = args[1:]
            if len(args) >= 1 and isinstance(args[0], str):
                parsed_influence_mode = args[0]
                args = args[1:]

        if "enable_daam" in kwargs:
            parsed_enable = bool(kwargs.get("enable_daam"))
        if "time_focus" in kwargs:
            parsed_focus = kwargs.get("time_focus")
            focus_explicit = True
        if "enable_time_focus" in kwargs:
            parsed_enable_time_focus = bool(kwargs.get("enable_time_focus"))
        if "enable_diagnostics" in kwargs:
            parsed_diagnostics = bool(kwargs.get("enable_diagnostics"))
        if "prompt_influence_mode" in kwargs:
            parsed_influence_mode = kwargs.get("prompt_influence_mode")
        if "influence_mode" in kwargs:
            parsed_influence_mode = kwargs.get("influence_mode")

        if parsed_enable_time_focus is None:
            # Backward compatibility:
            # if a focus was explicitly provided in older payloads, keep it active.
            parsed_enable_time_focus = focus_explicit
        if not parsed_enable_time_focus:
            parsed_focus = self.TIME_FOCUS_ALL

        parsed = ParsedDaamFlags(
            enable_daam=bool(parsed_enable),
            enable_time_focus=bool(parsed_enable_time_focus),
            time_focus=self._normalize_time_focus(parsed_focus),
            enable_diagnostics=bool(parsed_diagnostics),
            influence_mode=self._normalize_influence_mode(parsed_influence_mode),
            remaining_args=args,
        )
        return parsed.to_tuple()

    def _resolve_batch_prompts(self, prompts, extra_args, kwargs, p=None):
        if prompts is None:
            prompts = kwargs.get("prompts", None)

        if prompts is not None:
            return prompts

        # Some call paths pass prompts positionally after script args.
        for candidate in extra_args:
            if isinstance(candidate, (list, tuple)):
                if len(candidate) == 0:
                    prompts = candidate
                    break
                if isinstance(candidate[0], str):
                    prompts = candidate
                    break

        # Fallback: first list/tuple candidate even if not string-typed (legacy edge case).
        if prompts is None:
            for candidate in extra_args:
                if isinstance(candidate, (list, tuple)):
                    prompts = candidate
                    break

        if prompts is None:
            for candidate in extra_args:
                if isinstance(candidate, str):
                    prompts = [candidate]
                    break

        if prompts is None and p is not None:
            processing_prompts = getattr(p, "prompts", None)
            if isinstance(processing_prompts, (list, tuple)) and len(processing_prompts) > 0:
                prompts = processing_prompts
            elif isinstance(processing_prompts, str) and processing_prompts.strip():
                prompts = [processing_prompts]

        if prompts is None and p is not None:
            all_processing_prompts = getattr(p, "all_prompts", None)
            if isinstance(all_processing_prompts, (list, tuple)) and len(all_processing_prompts) > 0:
                prompts = all_processing_prompts

        if prompts is None and p is not None:
            single_prompt = getattr(p, "prompt", None)
            if isinstance(single_prompt, (list, tuple)) and len(single_prompt) > 0:
                prompts = single_prompt
            elif isinstance(single_prompt, str) and single_prompt.strip():
                prompts = [single_prompt]

        return prompts

    @staticmethod
    def _is_dummy_postprocess_call(processed):
        images_list = getattr(processed, "images", None)
        return isinstance(images_list, list) and len(images_list) == 0

    @staticmethod
    def _extract_seed_from_filename(filename):
        if not filename:
            return None
        basename = os.path.basename(filename)
        match = re.search(r"^\d+-(\d+)", basename)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_batch_pos_from_pnginfo(pnginfo):
        if not isinstance(pnginfo, dict):
            return None
        params_text = pnginfo.get("parameters", "")
        for pattern in (r"Batch pos: (\d+)", r"Batch index: (\d+)", r"Batch position: (\d+)"):
            match = re.search(pattern, params_text)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _is_grid_save(params):
        filename = os.path.abspath(getattr(params, "filename", "") or "")
        basename = os.path.basename(filename).lower()
        if basename.startswith("grid-") or basename.startswith("grid_") or basename.startswith("griddaam") or basename.startswith("grid_daam"):
            return True
        return False

    @classmethod
    def _warn_if_output_dirs_overlap(cls, p):
        if cls._warned_output_overlap:
            return

        samples_dir = os.path.abspath(getattr(p, "outpath_samples", "") or "")
        grids_dir = os.path.abspath(getattr(p, "outpath_grids", "") or "")
        if not samples_dir or not grids_dir:
            return

        try:
            common = os.path.commonpath([samples_dir, grids_dir])
        except ValueError:
            return

        # If grid output is a parent directory of sample output, directory-based
        # grid heuristics can accidentally classify normal samples as grids.
        if common == grids_dir and os.path.normcase(samples_dir) != os.path.normcase(grids_dir):
            print(
                "[DAAM] Warning: outpath_grids is a parent directory of outpath_samples. "
                "Use separate grid folders (e.g. outputs/txt2img-grids, outputs/img2img-grids) "
                "to avoid missing heatmaps in legacy configs."
            )
            cls._warned_output_overlap = True

    def _index_in_seed_list(self, seed_list, seed):
        if not isinstance(seed_list, (list, tuple)):
            return None
        for idx, value in enumerate(seed_list):
            try:
                if int(value) == seed:
                    return idx
            except Exception:
                continue
        return None

    def _resolve_batch_pos(self, params):
        p = getattr(params, "p", None)
        batch_size = max(int(getattr(p, "batch_size", 1) or 1), 1)

        # Preferred path on modern Forge: processing keeps a batch_index on each sample save.
        batch_index = getattr(p, "batch_index", None)
        if isinstance(batch_index, int) and batch_index >= 0:
            return batch_index % batch_size

        # Legacy metadata-based fallback used by older A1111 variants.
        batch_pos = self._extract_batch_pos_from_pnginfo(getattr(params, "pnginfo", None))
        if isinstance(batch_pos, int) and batch_pos >= 0:
            return batch_pos % batch_size

        # Resolve from seed filename against processing seed lists.
        seed = self._extract_seed_from_filename(getattr(params, "filename", ""))
        if seed is not None:
            all_seeds = getattr(p, "all_seeds", None)
            idx = self._index_in_seed_list(all_seeds, seed)
            if isinstance(idx, int):
                return idx % batch_size

            seeds = getattr(p, "seeds", None)
            idx = self._index_in_seed_list(seeds, seed)
            if isinstance(idx, int):
                return idx % batch_size

        # Last-resort fallback: monotonically assign save callbacks into batch slots.
        return self.saved_sample_count % batch_size

    def _resolve_effective_prompt(self, params, batch_pos: int):
        p = getattr(params, "p", None)
        prompts = getattr(p, "prompts", None)
        if isinstance(prompts, (list, tuple)) and len(prompts) > 0:
            if 0 <= batch_pos < len(prompts):
                return prompts[batch_pos]
            return prompts[0]
        if isinstance(prompts, str) and prompts:
            return prompts

        all_prompts = getattr(p, "all_prompts", None)
        if isinstance(all_prompts, (list, tuple)) and len(all_prompts) > 0:
            seed = self._extract_seed_from_filename(getattr(params, "filename", ""))
            if seed is not None:
                idx = self._index_in_seed_list(getattr(p, "all_seeds", None), seed)
                if isinstance(idx, int) and 0 <= idx < len(all_prompts):
                    return all_prompts[idx]
            if 0 <= self.saved_sample_count < len(all_prompts):
                return all_prompts[self.saved_sample_count]
            if 0 <= batch_pos < len(all_prompts):
                return all_prompts[batch_pos]
            return all_prompts[0]

        prompt_text = getattr(p, "prompt", "")
        if isinstance(prompt_text, (list, tuple)):
            return prompt_text[0] if len(prompt_text) > 0 else ""
        return prompt_text or ""

    def _resolve_effective_negative_prompt(self, params, batch_pos: int):
        p = getattr(params, "p", None)
        negative_prompts = getattr(p, "negative_prompts", None)
        if isinstance(negative_prompts, (list, tuple)) and len(negative_prompts) > 0:
            if 0 <= batch_pos < len(negative_prompts):
                return negative_prompts[batch_pos]
            return negative_prompts[0]
        if isinstance(negative_prompts, str) and negative_prompts:
            return negative_prompts

        all_negative_prompts = getattr(p, "all_negative_prompts", None)
        if isinstance(all_negative_prompts, (list, tuple)) and len(all_negative_prompts) > 0:
            seed = self._extract_seed_from_filename(getattr(params, "filename", ""))
            if seed is not None:
                idx = self._index_in_seed_list(getattr(p, "all_seeds", None), seed)
                if isinstance(idx, int) and 0 <= idx < len(all_negative_prompts):
                    return all_negative_prompts[idx]
            if 0 <= self.saved_sample_count < len(all_negative_prompts):
                return all_negative_prompts[self.saved_sample_count]
            if 0 <= batch_pos < len(all_negative_prompts):
                return all_negative_prompts[batch_pos]
            return all_negative_prompts[0]

        negative_prompt_text = getattr(p, "negative_prompt", "")
        if isinstance(negative_prompt_text, (list, tuple)):
            return negative_prompt_text[0] if len(negative_prompt_text) > 0 else ""
        return negative_prompt_text or ""
