from __future__ import annotations
import glob
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
    def _canonicalize_prompt_for_daam(cls, prompt: str):
        """
        Normalize extra-network tag placement for token alignment.
        Forge can expose slightly different chunking when tags (for example LoRA)
        are placed at the start of the prompt, which can destabilize DAAM mapping.
        """
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
        
        
        self.tracers = None
        self.run_active = False
        
        # Keep toggle as the last arg for API backward compatibility.
        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale, trace_each_layers, layers_as_row, enable_daam] 
    
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
            enable_daam: bool = True):
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
            *extra_args,
            prompts=None,
            **kwargs):
        enable_daam = True
        if len(extra_args) > 0 and isinstance(extra_args[0], bool):
            enable_daam = extra_args[0]

        if prompts is None:
            # Some call paths pass prompts positionally after script args.
            if len(extra_args) > 0 and not isinstance(extra_args[0], bool):
                prompts = extra_args[0]
            elif len(extra_args) > 1 and isinstance(extra_args[0], bool):
                prompts = extra_args[1]
            else:
                prompts = kwargs.get("prompts", None)

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

        styled_prompt = prompts[0]

        text_engine = _resolve_text_engine(p.sd_model)
        assert text_engine is not None, f"DAAM does not support this text encoder: {type(getattr(p.sd_model, 'cond_stage_model', None))}"

        prompt_analyzer = utils.PromptAnalyzer(text_engine, styled_prompt)
        self.prompt_analyzer = prompt_analyzer
        context_size = prompt_analyzer.context_size

        diffusion_model = _resolve_diffusion_model(p.sd_model)
                
        print(f"daam run with context_size={prompt_analyzer.context_size}, token_count={prompt_analyzer.token_count}")
        print(f"remade_tokens={prompt_analyzer.tokens}, multipliers={prompt_analyzer.multipliers}")
        print(f"hijack_comments={prompt_analyzer.hijack_comments}, used_custom_terms={prompt_analyzer.used_custom_terms}")
        print(f"fixes={prompt_analyzer.fixes}")
        
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
            **kwargs):
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
            for i, tracer in enumerate(self.tracers):
                with torch.no_grad():
                    effective_prompt = self._resolve_effective_prompt(params, batch_pos)
                    styled_prompt = self._canonicalize_prompt_for_daam(effective_prompt)
                    try:
                        global_heat_map = tracer.compute_global_heat_map(self.prompt_analyzer, styled_prompt, batch_pos)              
                    except Exception:
                        continue
                    
                    if i not in self.heatmap_images:
                        self.heatmap_images[i] = []
                    
                    if global_heat_map is not None:
                        heatmap_images = []
                        for raw_attention in self.attentions:
                            attention = self._resolve_attention_term(raw_attention, styled_prompt)
                            attn_candidates = self._attention_candidates(raw_attention, attention)
                                     
                            img_size = params.image.size
                            attn_caption = self.attn_captions[i] if i < len(self.attn_captions) else ""
                            caption = attention + (" " + attn_caption if attn_caption else "") if not self.hide_caption else None
                            
                            heat_map = None
                            for candidate in attn_candidates:
                                heat_map = global_heat_map.compute_word_heat_map(candidate)
                                if heat_map is not None:
                                    break
                            if heat_map is None : print(f"No heatmaps for '{raw_attention}' (resolved='{attention}')")
                             
                            heat_map_img = utils.expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                            img : Image.Image = utils.image_overlay_heat_map(params.image, heat_map_img, alpha=self.alpha, caption=caption, image_scale=self.heatmap_image_scale)

                            fullfn_without_extension, extension = os.path.splitext(params.filename) 
                            filename_attention = self._sanitize_filename_fragment(attention)
                            filename_caption = self._sanitize_filename_fragment(attn_caption) if attn_caption else ""
                            full_filename = fullfn_without_extension + "_" + filename_attention +  ("_" + filename_caption if filename_caption else "") + extension
                            
                            if self.use_grid:
                                heatmap_images.append(img)
                            else:
                                heatmap_images.append(img)
                                if not self.dont_save_images:               
                                    img.save(full_filename)                            
                        
                        self.heatmap_images[i] += heatmap_images
        
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
