from __future__ import annotations
import re

import gradio as gr
import modules.images as images
import modules.scripts as scripts
import torch
from modules import script_callbacks
from modules.processing import StableDiffusionProcessing, fix_seed
from modules.shared import opts

from scripts.daam import utils
from scripts.daam.trace import trace
from scripts.daam.attention_resolver import AttentionResolverMixin
from scripts.daam.diagnostics import DiagnosticsMixin
from scripts.daam.prompt_context import PromptContextMixin
from scripts.daam.render_pipeline import RenderPipelineMixin

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


class Script(
    AttentionResolverMixin,
    PromptContextMixin,
    DiagnosticsMixin,
    RenderPipelineMixin,
    scripts.Script,
):
    
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"
    INFLUENCE_POSITIVE = "Positive"
    INFLUENCE_NEGATIVE = "Negative"
    INFLUENCE_DELTA = "Delta (Pos-Neg)"
    INFLUENCE_DELTA_SIGNED = "Signed Delta (Pos-Neg)"
    INFLUENCE_DELTA_ABS = "Abs Delta (|Pos-Neg|)"
    TIME_FOCUS_DISABLED = "Disabled"
    TIME_FOCUS_ALL = "All"
    TIME_FOCUS_EARLY = "Early"
    TIME_FOCUS_MID = "Mid"
    TIME_FOCUS_LATE = "Late"
    TIME_FOCUS_TRIPLET = "Triplet"
    DYNAMIC_RESOLVE_MAX_CANDIDATES = 512
    DYNAMIC_RESOLVE_MAX_WILDCARD_VALUES = 4096
    DYNAMIC_RESOLVE_CACHE_MAX_ENTRIES = 4096
    _warned_output_overlap = False
    _invalid_filename_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
    _break_regex = re.compile(r"\bBREAK\b", flags=re.IGNORECASE)
    _variant_block_regex = re.compile(r"\{[^{}]*\|[^{}]*\}|\[[^\[\]]*\|[^\[\]]*\]")
    _wildcard_token_regex = re.compile(r"__([A-Za-z0-9_\-./\\*\[\]?]+)__")
    _extra_network_tag_regex = re.compile(r"<[^<>:]+:[^<>]+>")
    _dp_generator_cache_key = None
    _dp_generator = None
    _dp_wildcard_manager = None
    _dp_resolve_cache = {}
    

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
                    prompt_influence_mode = gr.Dropdown(
                        [
                            Script.INFLUENCE_POSITIVE,
                            Script.INFLUENCE_NEGATIVE,
                            Script.INFLUENCE_DELTA,
                            Script.INFLUENCE_DELTA_SIGNED,
                            Script.INFLUENCE_DELTA_ABS,
                        ],
                        label="Prompt influence mode",
                        value=Script.INFLUENCE_POSITIVE,
                    )

                with gr.Row():
                    enable_diagnostics = gr.Checkbox(label='Enable diagnostics', value=False)
        
        
        self.tracers = None
        self.run_active = False
        
        # Keep enable toggle at index 10 for API backward compatibility.
        return [
            attention_texts,
            hide_images,
            dont_save_images,
            hide_caption,
            use_grid,
            grid_layouyt,
            alpha,
            heatmap_image_scale,
            trace_each_layers,
            layers_as_row,
            enable_daam,
            enable_time_focus,
            time_focus,
            enable_diagnostics,
            prompt_influence_mode,
        ]
    
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
        enable_daam, enable_time_focus, time_focus, enable_diagnostics, influence_mode, _ = self._parse_optional_daam_flags(
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
        self.influence_mode = influence_mode
        self.prompt_analyzers = []
        self.negative_prompt_analyzers = []
        self.negative_prompt_analyzer_cache = {}
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
        enable_daam, enable_time_focus, time_focus, enable_diagnostics, influence_mode, remaining_args = self._parse_optional_daam_flags(
            enable_daam, extra_args, **kwargs
        )
        prompts = self._resolve_batch_prompts(prompts, remaining_args, kwargs, p=p)

        self.enable_time_focus = enable_time_focus
        self.time_focus = time_focus
        self.enable_diagnostics = enable_diagnostics
        self.influence_mode = influence_mode

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
        self.negative_prompt_analyzers = []
        self.negative_prompt_analyzer_cache = {}
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

        negative_prompts = getattr(p, "negative_prompts", None)
        if isinstance(negative_prompts, (list, tuple)):
            negative_prompt_list = [self._canonicalize_prompt_for_daam(x) for x in list(negative_prompts)]
        elif isinstance(negative_prompts, str):
            negative_prompt_list = [self._canonicalize_prompt_for_daam(negative_prompts)]
        else:
            negative_prompt_text = getattr(p, "negative_prompt", None)
            if isinstance(negative_prompt_text, str):
                negative_prompt_list = [self._canonicalize_prompt_for_daam(negative_prompt_text)]
            else:
                negative_prompt_list = []

        for negative_prompt in negative_prompt_list:
            if isinstance(negative_prompt, str) and negative_prompt.strip():
                self.negative_prompt_analyzers.append(utils.PromptAnalyzer(text_engine, negative_prompt))

        diffusion_model = _resolve_diffusion_model(p.sd_model)
                
        first_analyzer = self.prompt_analyzer
        print(
            f"daam run with context_size={context_size}, token_count={first_analyzer.token_count}, "
            f"time_focus_enabled={self.enable_time_focus}, time_focus={self.time_focus}, "
            f"influence_mode={self.influence_mode}, diagnostics={self.enable_diagnostics}, "
            f"batch_prompts={len(prompts_list)}"
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
        _, self.enable_time_focus, self.time_focus, self.enable_diagnostics, self.influence_mode, _ = self._parse_optional_daam_flags(
            enable_daam, extra_args, **kwargs
        )
        if not self.enabled:
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
    
def handle_before_image_saved(params : script_callbacks.ImageSaveParams):
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)
   
    return
 
script_callbacks.on_before_image_saved(handle_before_image_saved)   

