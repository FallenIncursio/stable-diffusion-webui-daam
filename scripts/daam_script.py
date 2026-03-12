from __future__ import annotations
import glob
import json
import os
import re

import gradio as gr
import modules.images as images
import modules.scripts as scripts
import torch
from modules import script_callbacks
from modules.processing import StableDiffusionProcessing, fix_seed
from modules.shared import opts
from PIL import Image

from scripts.daam import trace, utils

before_image_saved_handler = None


def _resolve_text_engine(sd_model):
    for attr in ("text_processing_engine", "text_processing_engine_l", "text_processing_engine_g"):
        engine = getattr(sd_model, attr, None)
        if engine is not None:
            return engine

    cond_stage_model = getattr(sd_model, "cond_stage_model", None)
    if cond_stage_model is None:
        return None

    if hasattr(cond_stage_model, "tokenize_line") or hasattr(cond_stage_model, "tokenize"):
        return cond_stage_model

    embedders = getattr(cond_stage_model, "embedders", None)
    if isinstance(embedders, (list, tuple)) and len(embedders) > 0:
        return embedders[0]

    return None


def _resolve_diffusion_model(sd_model):
    if hasattr(sd_model, "model") and hasattr(sd_model.model, "diffusion_model"):
        return sd_model.model.diffusion_model

    forge_objects = getattr(sd_model, "forge_objects", None)
    if forge_objects is not None:
        unet = getattr(forge_objects, "unet", None)
        if unet is not None and hasattr(unet, "model") and hasattr(unet.model, "diffusion_model"):
            return unet.model.diffusion_model

    raise AttributeError(f"Unsupported model structure for DAAM tracing: {type(sd_model)}")


class Script(scripts.Script):
    
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"
    TIME_FOCUS_DISABLED = "Disabled"
    TIME_FOCUS_ALL = "All"
    TIME_FOCUS_EARLY = "Early"
    TIME_FOCUS_MID = "Mid"
    TIME_FOCUS_LATE = "Late"
    TIME_FOCUS_TRIPLET = "Triplet"
    _warned_output_overlap = False
    _invalid_filename_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
    _break_regex = re.compile(r"\bBREAK\b", flags=re.IGNORECASE)
    _variant_block_regex = re.compile(r"\{[^{}]*\|[^{}]*\}|\[[^\[\]]*\|[^\[\]]*\]")
    _wildcard_token_regex = re.compile(r"__([A-Za-z0-9_\-./\\*\[\]?]+)__")
    _extra_network_tag_regex = re.compile(r"<[^<>:]+:[^<>]+>")
    

    def title(self):
        return "Daam script"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def _safe_unhook_tracers(self):
        if self.tracers is None:
            return
        for tracer in self.tracers:
            if getattr(tracer, "hooked", False):
                tracer.unhook()

    @classmethod
    def _split_attention_texts(cls, attention_texts: str):
        normalized = cls._break_regex.sub(",", attention_texts or "")
        return [s.strip() for s in normalized.split(",") if s.strip()]

    @classmethod
    def _normalize_for_match(cls, text: str):
        if not text:
            return ""
        normalized = text.lower()
        normalized = cls._break_regex.sub(" ", normalized)
        normalized = re.sub(r"<lora:[^>]+>", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _contains_phrase(cls, prompt: str, phrase: str):
        prompt_n = cls._normalize_for_match(prompt)
        phrase_n = cls._normalize_for_match(phrase)
        if not prompt_n or not phrase_n:
            return False
        return phrase_n in prompt_n

    @classmethod
    def _best_prompt_match(cls, prompt: str, candidates):
        best = None
        best_len = -1
        for candidate in candidates:
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            if cls._contains_phrase(prompt, candidate):
                candidate_len = len(candidate)
                if candidate_len > best_len:
                    best = candidate
                    best_len = candidate_len
        return best

    @classmethod
    def _extract_variant_options(cls, token: str):
        token = (token or "").strip()
        if len(token) < 3:
            return []
        if not ((token.startswith("{") and token.endswith("}")) or (token.startswith("[") and token.endswith("]"))):
            return []
        inner = token[1:-1].strip()
        if "|" not in inner:
            return []
        # Dynamic prompts can prefix range/joiner blocks with '$$'.
        if "$$" in inner:
            inner = inner.rsplit("$$", 1)[-1]
        options = []
        for opt in inner.split("|"):
            cleaned = re.sub(r"^\s*[-+]?\d+(?:\.\d+)?::", "", opt).strip()
            if cleaned:
                options.append(cleaned)
        return options

    @staticmethod
    def _get_webui_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    @classmethod
    def _get_wildcard_dirs(cls):
        wildcard_dirs = []
        configured_wildcard_dir = getattr(opts, "wildcard_dir", None)
        if configured_wildcard_dir:
            wildcard_dirs.append(os.path.abspath(configured_wildcard_dir))
        wildcard_dirs.append(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "sd-dynamic-prompts", "wildcards")
            )
        )
        wildcard_dirs.append(os.path.join(cls._get_webui_root(), "wildcards"))
        deduped = []
        for wildcard_dir in wildcard_dirs:
            if wildcard_dir and wildcard_dir not in deduped and os.path.isdir(wildcard_dir):
                deduped.append(wildcard_dir)
        return deduped

    @classmethod
    def _load_wildcard_values(cls, wildcard_name: str):
        wildcard_name = (wildcard_name or "").strip().replace("\\", "/")
        if not wildcard_name:
            return []
        has_glob = any(ch in wildcard_name for ch in ("*", "?", "["))
        values = []
        seen = set()
        for wildcard_dir in cls._get_wildcard_dirs():
            patterns = []
            if wildcard_name.endswith(".txt"):
                patterns.append(os.path.join(wildcard_dir, wildcard_name))
            else:
                patterns.append(os.path.join(wildcard_dir, wildcard_name + ".txt"))
            if has_glob:
                wildcard_pattern = wildcard_name if wildcard_name.endswith(".txt") else wildcard_name + ".txt"
                patterns.append(os.path.join(wildcard_dir, "**", wildcard_pattern))
            for pattern in patterns:
                for wildcard_file in glob.glob(pattern, recursive=has_glob):
                    if not os.path.isfile(wildcard_file):
                        continue
                    try:
                        with open(wildcard_file, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                if line not in seen:
                                    seen.add(line)
                                    values.append(line)
                    except Exception:
                        continue
        return values

    @classmethod
    def _resolve_variant_blocks(cls, text: str, prompt: str):
        def _replace(match: re.Match):
            token = match.group(0)
            options = cls._extract_variant_options(token)
            if not options:
                return token
            best = cls._best_prompt_match(prompt, options)
            return best if best is not None else options[0]

        return cls._variant_block_regex.sub(_replace, text)

    @classmethod
    def _resolve_wildcard_tokens(cls, text: str, prompt: str):
        def _replace(match: re.Match):
            wildcard_name = match.group(1)
            candidates = cls._load_wildcard_values(wildcard_name)
            if not candidates:
                return match.group(0)
            best = cls._best_prompt_match(prompt, candidates)
            return best if best is not None else match.group(0)

        return cls._wildcard_token_regex.sub(_replace, text)

    @classmethod
    def _resolve_attention_term(cls, raw_attention: str, prompt: str):
        attention = (raw_attention or "").strip()
        if not attention:
            return ""
        if not prompt:
            return cls._break_regex.sub(" ", attention).strip()
        resolved = cls._resolve_variant_blocks(attention, prompt)
        resolved = cls._resolve_wildcard_tokens(resolved, prompt)
        resolved = cls._break_regex.sub(" ", resolved)
        resolved = re.sub(r"\s+", " ", resolved).strip(" ,")
        return resolved if resolved else attention

    @classmethod
    def _attention_candidates(cls, raw_attention: str, resolved_attention: str):
        candidates = []
        for term in (resolved_attention, raw_attention):
            term = (term or "").strip()
            if not term:
                continue
            candidates.append(cls._break_regex.sub(" ", term).strip(" ,"))
            for piece in re.split(r",", cls._break_regex.sub(",", term)):
                piece = piece.strip()
                if piece:
                    candidates.append(piece)
        deduped = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    @classmethod
    def _sanitize_filename_fragment(cls, text: str):
        text = cls._invalid_filename_chars.sub("_", (text or "").strip())
        text = re.sub(r"\s+", " ", text).strip().strip(".")
        return text or "attention"

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

    def _parse_optional_daam_flags(self, enable_daam, extra_args, **kwargs):
        """
        Parse optional Sprint-A flags while keeping old API payloads valid.

        Supported positional forms:
        - [enable_daam]
        - [enable_daam, time_focus, enable_diagnostics]
        - [time_focus, enable_diagnostics, enable_daam] (legacy sprint test payload)
        """
        parsed_enable = bool(enable_daam)
        parsed_focus = self.TIME_FOCUS_ALL
        parsed_diagnostics = False
        parsed_enable_time_focus = None
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

        # Preferred new order: [enable_time_focus, time_focus, enable_diagnostics]
        if len(args) >= 3 and isinstance(args[0], bool) and isinstance(args[1], str) and isinstance(args[2], bool):
            parsed_enable_time_focus = args[0]
            parsed_focus = args[1]
            focus_explicit = True
            parsed_diagnostics = args[2]
            args = args[3:]

        # Legacy order from earlier tests: [time_focus, diagnostics, enable]
        elif len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], bool) and isinstance(args[2], bool):
            parsed_focus = args[0]
            focus_explicit = True
            parsed_diagnostics = args[1]
            parsed_enable = args[2]
            args = args[3:]
        else:
            # Legacy order: [time_focus?, diagnostics?] after explicit enable_daam arg.
            if len(args) >= 1 and isinstance(args[0], str):
                parsed_focus = args[0]
                focus_explicit = True
                args = args[1:]
            if len(args) >= 1 and isinstance(args[0], bool):
                parsed_diagnostics = args[0]
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

        if parsed_enable_time_focus is None:
            # Backward compatibility:
            # if a focus was explicitly provided in older payloads, keep it active.
            parsed_enable_time_focus = focus_explicit
        if not parsed_enable_time_focus:
            parsed_focus = self.TIME_FOCUS_ALL

        return (
            bool(parsed_enable),
            bool(parsed_enable_time_focus),
            self._normalize_time_focus(parsed_focus),
            bool(parsed_diagnostics),
            args,
        )

    def _resolve_batch_prompts(self, prompts, extra_args, kwargs):
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

        return prompts

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

    def _diagnose_missing_heatmap(self, raw_attention: str, resolved_attention: str, candidates, prompt_text: str, prompt_analyzer):
        if len(candidates) == 0:
            return "no_candidates"

        if not any(self._contains_phrase(prompt_text, candidate) for candidate in candidates):
            return "term_not_in_resolved_prompt"

        used_custom_terms = getattr(prompt_analyzer, "used_custom_terms", []) if prompt_analyzer is not None else []
        attention_keys = {self._normalize_for_match(raw_attention), self._normalize_for_match(resolved_attention)}
        for item in used_custom_terms:
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                continue
            key = self._normalize_for_match(str(item[0]))
            if key in attention_keys:
                return "embedding_or_custom_term"

        if prompt_analyzer is not None:
            token_count = int(getattr(prompt_analyzer, "token_count", 0) or 0)
            context_size = int(getattr(prompt_analyzer, "context_size", 0) or 0)
            if context_size > 0 and token_count >= max(1, context_size - 2):
                return "possible_context_truncation"

        return "token_not_found_in_attention_map"

    def _save_diagnostics(self, params, payload):
        if not payload:
            return
        fullfn_without_extension, _ = os.path.splitext(params.filename)
        diagnostics_path = fullfn_without_extension + "_daam_diag.json"
        try:
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DAAM] Failed to save diagnostics at {diagnostics_path}: {e}")

    @classmethod
    def _canonicalize_prompt_for_daam(cls, prompt: str):
        """
        Normalize extra-network tag placement for token alignment.
        Forge can expose slightly different chunking when tags (for example LoRA)
        are placed at the start of the prompt, which can destabilize DAAM mapping.
        """
        if prompt is None:
            return ""
        if not isinstance(prompt, str):
            return ""
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt

        tags = cls._extra_network_tag_regex.findall(prompt)
        text_wo_tags = cls._extra_network_tag_regex.sub(" ", prompt)
        text_wo_tags = re.sub(r"\s*,\s*", ", ", text_wo_tags)
        text_wo_tags = re.sub(r",\s*,+", ", ", text_wo_tags)
        text_wo_tags = re.sub(r"\s+", " ", text_wo_tags).strip(" ,")

        if not tags:
            return text_wo_tags

        tag_tail = ", ".join(tag.strip() for tag in tags if tag.strip())
        if not text_wo_tags:
            return tag_tail
        return f"{text_wo_tags}, {tag_tail}"

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

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Attention Heatmap", open=False):
                enable_daam = gr.Checkbox(label='Enable DAAM', value=False)
                attention_texts = gr.Text(label='Attention texts for visualization. (comma separated)', value='')

                with gr.Row():
                    hide_images = gr.Checkbox(label='Hide heatmap images', value=False)
                    
                    dont_save_images = gr.Checkbox(label='Do not save heatmap images', value=False)
                    
                    hide_caption = gr.Checkbox(label='Hide caption', value=False)
                    
                with gr.Row():
                    use_grid = gr.Checkbox(label='Use grid (output to grid dir)', value=False)
                        
                    grid_layouyt = gr.Dropdown(
                            [Script.GRID_LAYOUT_AUTO, Script.GRID_LAYOUT_PREVENT_EMPTY, Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW], label="Grid layout",
                            value=Script.GRID_LAYOUT_AUTO
                        )
                        
                with gr.Row():
                    alpha = gr.Slider(label='Heatmap blend alpha', value=0.5, minimum=0, maximum=1, step=0.01)
                
                    heatmap_image_scale = gr.Slider(label='Heatmap image scale', value=1.0, minimum=0.1, maximum=1, step=0.025)

                with gr.Row():
                    trace_each_layers = gr.Checkbox(label = 'Trace each layers', value=False)

                    layers_as_row = gr.Checkbox(label = 'Use layers as row instead of Batch Length', value=False)

                with gr.Row():
                    enable_time_focus = gr.Checkbox(label='Enable time focus', value=False)

                    time_focus = gr.Dropdown(
                        [
                            Script.TIME_FOCUS_DISABLED,
                            Script.TIME_FOCUS_ALL,
                            Script.TIME_FOCUS_EARLY,
                            Script.TIME_FOCUS_MID,
                            Script.TIME_FOCUS_LATE,
                            Script.TIME_FOCUS_TRIPLET,
                        ],
                        label="Time focus",
                        value=Script.TIME_FOCUS_DISABLED,
                    )

                with gr.Row():
                    enable_diagnostics = gr.Checkbox(label='Enable diagnostics', value=False)
        
        
        self.tracers = None
        self.run_active = False
        
        # Keep enable toggle at index 10 for API backward compatibility.
        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale, trace_each_layers, layers_as_row, enable_daam, enable_time_focus, time_focus, enable_diagnostics] 
    
    def process(self, 
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool,
            enable_daam: bool = True,
            *extra_args,
            **kwargs):
        enable_daam, enable_time_focus, time_focus, enable_diagnostics, _ = self._parse_optional_daam_flags(
            enable_daam, extra_args, **kwargs
        )
        attentions = self._split_attention_texts(attention_texts)

        # ADetailer can trigger nested script.process() calls during a running txt2img pass.
        # Do not reset DAAM state in that case, otherwise accumulated batch heatmaps are lost.
        if enable_daam and attentions and getattr(self, "run_active", False) and self.tracers is not None:
            return

        self.enabled = False # in case the assert fails
        assert opts.samples_save, "Cannot run Daam script. Enable 'Always save all generated images' setting."

        self.images = []
        self.hide_images = hide_images
        self.dont_save_images = dont_save_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = dict()
        self.deferred_cleanup = False
        self.saved_sample_count = 0
        self.enable_time_focus = enable_time_focus
        self.time_focus = time_focus
        self.enable_diagnostics = enable_diagnostics
        self.prompt_analyzers = []
        self.text_engine = None

        self.attentions = attentions
        self.enabled = bool(enable_daam) and len(self.attentions) > 0
        self.run_active = self.enabled

        if not self.enabled:
            self._safe_unhook_tracers()
            self.tracers = None
            global before_image_saved_handler
            before_image_saved_handler = None
            return

        Script._warn_if_output_dirs_overlap(p)

        fix_seed(p)
        
    def process_batch(self,
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool,
            enable_daam: bool = True,
            *extra_args,
            prompts=None,
            **kwargs):
        enable_daam, enable_time_focus, time_focus, enable_diagnostics, remaining_args = self._parse_optional_daam_flags(
            enable_daam, extra_args, **kwargs
        )
        prompts = self._resolve_batch_prompts(prompts, remaining_args, kwargs)

        self.enable_time_focus = enable_time_focus
        self.time_focus = time_focus
        self.enable_diagnostics = enable_diagnostics

        if not enable_daam:
            return
                 
        if not self.enabled:
            return
        
        if not prompts:
            return

        # Normalize batch prompts in-place so LoRA/extra-network tags are treated
        # consistently regardless of their original position in the user prompt.
        if isinstance(prompts, list):
            for idx, prompt in enumerate(prompts):
                prompts[idx] = self._canonicalize_prompt_for_daam(prompt)
        elif isinstance(prompts, tuple):
            prompts = tuple(self._canonicalize_prompt_for_daam(prompt) for prompt in prompts)

        prompts_list = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
        if len(prompts_list) == 0:
            return

        text_engine = _resolve_text_engine(p.sd_model)
        assert text_engine is not None, f"DAAM does not support this text encoder: {type(getattr(p.sd_model, 'cond_stage_model', None))}"
        self.text_engine = text_engine

        self.prompt_analyzers = []
        context_size = 77
        for prompt_text in prompts_list:
            if not isinstance(prompt_text, str):
                continue
            if not prompt_text.strip():
                continue
            prompt_analyzer = utils.PromptAnalyzer(text_engine, prompt_text)
            self.prompt_analyzers.append(prompt_analyzer)
            context_size = max(context_size, int(getattr(prompt_analyzer, "context_size", 77) or 77))
        if len(self.prompt_analyzers) == 0:
            return

        self.prompt_analyzer = self.prompt_analyzers[0]

        diffusion_model = _resolve_diffusion_model(p.sd_model)
                
        first_analyzer = self.prompt_analyzer
        print(
            f"daam run with context_size={context_size}, token_count={first_analyzer.token_count}, "
            f"time_focus_enabled={self.enable_time_focus}, time_focus={self.time_focus}, "
            f"diagnostics={self.enable_diagnostics}, batch_prompts={len(prompts_list)}"
        )
        print(f"remade_tokens={first_analyzer.tokens}, multipliers={first_analyzer.multipliers}")
        print(f"hijack_comments={first_analyzer.hijack_comments}, used_custom_terms={first_analyzer.used_custom_terms}")
        print(f"fixes={first_analyzer.fixes}")
        
        if any(isinstance(item, (list, tuple)) and len(item) > 0 and item[0] in self.attentions for item in self.prompt_analyzer.used_custom_terms):
            print("Embedding heatmap cannot be shown.")
            
        global before_image_saved_handler
        before_image_saved_handler = lambda params : self.before_image_saved(params)
                
        with torch.no_grad():
            # cannot trace the same block from two tracers
            num_input = len(getattr(diffusion_model, "input_blocks", []))
            num_output = len(getattr(diffusion_model, "output_blocks", []))
            has_middle = 1 if hasattr(diffusion_model, "middle_block") else 0

            if trace_each_layers and (num_input + num_output + has_middle) > 0:
                layer_count = num_input + num_output + has_middle
                self.tracers = [trace(p.sd_model, p.height, p.width, context_size, layer_idx=i) for i in range(layer_count)]
                self.attn_captions = [f"IN{i:02d}" for i in range(num_input)] + (["MID"] if has_middle else []) + [f"OUT{i:02d}" for i in range(num_output)]
            else:
                self.tracers = [trace(p.sd_model, p.height, p.width, context_size)]
                self.attn_captions = [""]
        
            for tracer in self.tracers:
                tracer.hook()

    def postprocess(self, p, processed,
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool,
            enable_daam: bool = True,
            *extra_args,
            **kwargs):
        _, self.enable_time_focus, self.time_focus, self.enable_diagnostics, _ = self._parse_optional_daam_flags(
            enable_daam, extra_args, **kwargs
        )
        if self.enabled == False:
            return

        # ADetailer calls `scripts.postprocess(copy(p), Processed(..., images=[]))`
        # inside its postprocess_image hook before the final save.
        # If we fully finalize here, DAAM is disabled for the real output image.
        if self._is_dummy_postprocess_call(processed):
            self._safe_unhook_tracers()
            self.deferred_cleanup = True
            return processed
        
        self._safe_unhook_tracers()
        self.tracers = None
        self.deferred_cleanup = False
        self.run_active = False
        
        initial_info = None

        if initial_info is None:
            initial_info = processed.info
            
        self.images += processed.images

        global before_image_saved_handler
        before_image_saved_handler = None

        if layers_as_row:
            images_list = []
            for i in range(p.batch_size * p.n_iter):
                imgs = []
                for k in sorted(self.heatmap_images.keys()):
                    imgs += [self.heatmap_images[k][len(self.attentions)*i + j] for j in range(len(self.attentions))]
                images_list.append(imgs)
        else:
            images_list = [self.heatmap_images[k] for k in sorted(self.heatmap_images.keys())]

        for img_list in images_list:

            if img_list and self.use_grid:

                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW
                        
                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = images.image_grid(img_list)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    if layers_as_row:
                        batch_size = len(self.attentions)
                        rows = len(self.heatmap_images)
                    else:
                        batch_size = p.batch_size
                        rows = p.batch_size * p.n_iter
                    grid_img = images.image_grid(img_list, batch_size=batch_size, rows=rows)
                else:
                    continue
                
                if not self.dont_save_images:
                    images.save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)
                
                if not self.hide_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])
            
            else:
                if not self.hide_images:
                    processed.images[:0] = img_list
                    processed.index_of_first_image += len(img_list)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(img_list)

        return processed
    
    def before_image_saved(self, params : script_callbacks.ImageSaveParams):               
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
                    if i not in self.heatmap_images:
                        self.heatmap_images[i] = []

                    heatmap_images = []
                    for focus_label, focus_kwargs in self._resolve_focus_targets(tracer):
                        try:
                            global_heat_map = tracer.compute_global_heat_map(
                                prompt_analyzer, styled_prompt, batch_pos, **focus_kwargs
                            )
                        except Exception:
                            continue

                        if global_heat_map is None:
                            continue

                        for raw_attention in self.attentions:
                            attention = self._resolve_attention_term(raw_attention, styled_prompt)
                            attn_candidates = self._attention_candidates(raw_attention, attention)

                            img_size = params.image.size
                            attn_caption = self.attn_captions[i] if i < len(self.attn_captions) else ""
                            focus_caption = focus_label if getattr(self, "enable_time_focus", False) else ""
                            caption = (
                                attention
                                + (" " + attn_caption if attn_caption else "")
                                + (f" [{focus_caption}]" if focus_caption else "")
                                if not self.hide_caption
                                else None
                            )

                            heat_map = None
                            matched_candidate = None
                            for candidate in attn_candidates:
                                heat_map = global_heat_map.compute_word_heat_map(candidate)
                                if heat_map is not None:
                                    matched_candidate = candidate
                                    break
                            reason = "ok"
                            if heat_map is None:
                                reason = self._diagnose_missing_heatmap(
                                    raw_attention, attention, attn_candidates, styled_prompt, prompt_analyzer
                                )
                                print(
                                    f"No heatmaps for '{raw_attention}' "
                                    f"(resolved='{attention}', focus={focus_label}, reason={reason})"
                                )
                            if self.enable_diagnostics:
                                diagnostics_entries.append(
                                    {
                                        "layer": attn_caption or "ALL",
                                        "focus": focus_label,
                                        "raw_attention": raw_attention,
                                        "resolved_attention": attention,
                                        "candidates": attn_candidates,
                                        "matched": heat_map is not None,
                                        "matched_candidate": matched_candidate,
                                        "reason": reason,
                                    }
                                )

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
                            full_filename = (
                                fullfn_without_extension
                                + "_"
                                + filename_attention
                                + ("_" + filename_focus if filename_focus else "")
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
                    "prompt": styled_prompt,
                    "entries": diagnostics_entries,
                }
                self._save_diagnostics(params, diagnostics_payload)
        
        self.heatmap_images = {j:self.heatmap_images[j] for j in self.heatmap_images.keys() if self.heatmap_images[j]}

        # if it is last batch pos, clear heatmaps
        if self.tracers is not None and batch_pos == params.p.batch_size - 1:
            for tracer in self.tracers:
                tracer.reset()

        self.saved_sample_count += 1
            
        return


def handle_before_image_saved(params : script_callbacks.ImageSaveParams):
    
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)
   
    return
 
script_callbacks.on_before_image_saved(handle_before_image_saved)   
