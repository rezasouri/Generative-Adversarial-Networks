"""Launch the optional ForgeGAN interactive defect-synthesis demo.

Install with: python -m pip install -e ".[demo]"
Run with: python demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import gradio as gr
except ImportError as error:  # pragma: no cover - depends on optional extra
    raise SystemExit(
        'Gradio is optional. Install it with: python -m pip install -e ".[demo]"'
    ) from error

from forgegan.synthesis.procedural import DefectSpec, synthesize_defect


def generate(
    image,
    defect_type: str,
    center_x: float,
    center_y: float,
    scale: float,
    opacity: float,
    angle: float,
    seed: int,
):
    if image is None:
        raise gr.Error("Upload a normal product image first.")
    if min(image.size) < 16:
        raise gr.Error("Image dimensions must be at least 16 × 16 pixels.")
    spec = DefectSpec(
        kind=defect_type,
        center_x=center_x,
        center_y=center_y,
        scale=scale,
        opacity=opacity,
        angle_degrees=angle,
    )
    generated, mask = synthesize_defect(image, spec, seed=int(seed))
    return image.convert("RGB"), generated, mask


with gr.Blocks(title="ForgeGAN | Synthetic Defect Studio") as demo:
    gr.Markdown(
        """
# ForgeGAN — Synthetic Defect Studio

Explore controlled, reproducible defect synthesis for industrial visual
inspection. Upload a **normal** product image, choose a defect, and inspect
the generated image and its pixel mask.

> Research prototype: this demo uses handcrafted procedural intensity
> perturbations. It is not a trained GAN, does not certify realistic defects,
> and does not prove downstream anomaly-detection gains. Images are processed
> in this session and are not added to the repository.
"""
    )
    with gr.Row():
        with gr.Column():
            source = gr.Image(label="Normal source image", type="pil")
            defect = gr.Radio(["spot", "scratch"], value="spot", label="Defect shape")
            with gr.Row():
                cx = gr.Slider(0, 1, value=0.5, step=0.01, label="Center X")
                cy = gr.Slider(0, 1, value=0.5, step=0.01, label="Center Y")
            scale = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Scale")
            opacity = gr.Slider(0.05, 1, value=0.75, step=0.05, label="Opacity")
            angle = gr.Slider(0, 180, value=30, step=1, label="Scratch angle")
            seed = gr.Number(value=42, precision=0, label="Random seed")
            run = gr.Button("Generate synthetic defect", variant="primary")
        with gr.Column():
            original = gr.Image(label="Original")
            generated = gr.Image(label="Synthetic defect")
            mask = gr.Image(label="Ground-truth binary mask")
    run.click(
        generate,
        inputs=[source, defect, cx, cy, scale, opacity, angle, seed],
        outputs=[original, generated, mask],
    )
    gr.Markdown(
        """
### Reproducibility

The same source image, parameters, and seed produce the same output.
Only masked pixels are changed; the binary mask identifies the altered region.

[Source code](https://github.com/rezasouri/Generative-Adversarial-Networks)
· [Method and limitations](../docs/PROCEDURAL_SYNTHESIS.md)
· [Matched evaluation protocol](../docs/MATCHED_ANOMALY_BENCHMARK.md)
"""
    )


if __name__ == "__main__":
    demo.launch()
