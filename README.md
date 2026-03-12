# DAAM Extension for Stable Diffusion Web UI / Forge

This extension is a WebUI script port of [DAAM](https://github.com/castorini/daam) with compatibility fixes for recent WebUI and Forge backends.

## Features

- Attention heatmaps for comma-separated target phrases.
- `BREAK` is also treated as a separator in attention text input.
- Dynamic Prompts term resolution for attention targets:
  - Variant blocks like `{red eyes|blue eyes}` / `[red eyes|blue eyes]`
  - Wildcard tokens like `__eye_color__` (resolved against wildcard files)
- Explicit `Enable DAAM` toggle (on/off like other always-on scripts).
- Time-focus controls with explicit on/off:
  - `Enable time focus`
  - modes: `Disabled` (default), `All`, `Early`, `Mid`, `Late`, `Triplet`
- `Triplet` mode renders `Early + Mid + Late` in one run.
- Optional diagnostics JSON (`*_daam_diag.json`) with per-term match status and reason codes.
- SDXL-compatible prompt token handling.
- Forge-compatible UNet and text-encoder resolution.
- Grid output mode for multiple attention terms.

## Install

Clone into your WebUI/Forge extension directory:

## Usage

1. Open `txt2img` or `img2img`.
2. Expand `Attention Heatmap`.
3. Enable `Enable DAAM`.
4. Enter target terms in `Attention texts for visualization` (comma separated).
5. Generate.

The extension creates blended heatmap images and, if enabled, a `grid_daam-*.png` in the grid output directory.

Notes:

- Multi-word phrases are matched as one sequence (for example `white dress`).
- You can separate attention targets with `,` or `BREAK`.
- For Dynamic Prompts, DAAM matches against the resolved per-image prompt text.
- DAAM normalizes extra-network tags (for example LoRA) for internal token mapping, so LoRA position (start/middle/end) should not change heatmap term matching.
- For API generation, set `"save_images": true` so save hooks run and DAAM heatmaps are produced.
- `All` means one aggregated heatmap over all denoising steps.
- `Triplet` means three phase heatmaps (`Early`, `Mid`, `Late`) in one generation.
- For `Triplet`, using `Use grid (output to grid dir)` is recommended for easier comparison.

## Forge Notes

### Hires Fix / Upscaling

DAAM works with Hires Fix and non-latent upscalers (for example `4x-UltraSharp`).

For API calls on some Forge builds, include:

```json
"hr_additional_modules": []
```

Without this field, Forge can raise:

```text
TypeError: argument of type 'NoneType' is not iterable
```

This error originates in Forge processing, not in DAAM logic itself.

### ADetailer

ADetailer performs an internal pre-pass `postprocess` callback before final image saving.  
DAAM now defers final cleanup for that dummy callback so heatmaps remain available for the final output save stage.

Current behavior:

- DAAM heatmaps are generated for the main generation pass.
- ADetailer can still change the final image composition after the main pass, so heatmaps should be interpreted as prompt attention guidance, not strict post-inpaint attribution.

### Batch Size / Batch Count

DAAM now supports Forge runs with both:

- `batch_size > 1`
- `n_iter (batch count) > 1`

The extension no longer depends on `Batch pos` PNG metadata (which may be absent on Forge).  
Instead, it resolves per-image batch positions from processing state and seed/filename fallback logic, and handles compact Forge `cond_or_uncond` layouts in attention tracing.
It also keeps per-batch prompt analyzers so variable prompts in one batch resolve terms correctly.

### Output Folder Layout

To avoid missing heatmaps, keep sample and grid output folders separate.

Recommended:

- `outdir_txt2img_samples = outputs/txt2img-images`
- `outdir_img2img_samples = outputs/img2img-images`
- `outdir_txt2img_grids = outputs/txt2img-grids`
- `outdir_img2img_grids = outputs/img2img-grids`

Avoid setting grid dirs to a parent folder that also contains sample dirs (for example `outdir_txt2img_grids = outputs` with samples in `outputs/*`), because this can cause grid/sample misclassification in legacy configs.

## Sample

- Prompt: `A photo of a cute cat wearing sunglasses relaxing on a beach`
- Attention text: `cat, sunglasses, beach`
- Output: original + one heatmap per attention term

<img src="images/00006-2623256163.png" width="150">
<img src="images/00006-2623256163_cat.png" width="150">
<img src="images/00006-2623256163_sunglasses.png" width="150">
<img src="images/00006-2623256163_beach.png" width="150">
