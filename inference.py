"""FireRed-Image-Edit Inference Demo."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FireRed-Image-Edit inference script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        nargs="+",
        default=[Path("./examples/edit_example.png")],
        help="Path(s) to the input image(s). Supports 1-N images. "
             "When more than 3 images are given the agent will "
             "automatically crop and stitch them into 2-3 composites.",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=Path("output_edit.png"),
        help="Path to save the output image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="在书本封面Python的下方，添加一行英文文字2nd Edition",
        help="Editing prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--recaption",
        action="store_true",
        default=False,
        help="Enable agent-based recaption: expand the editing prompt to "
             "~512 words/characters via Gemini for richer context. "
             "Requires GEMINI_API_KEY environment variable.",
    )
    return parser.parse_args()


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    """Load FireRed image edit pipeline."""
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def main() -> None:
    """Main entry point."""
    args = parse_args()

    pipeline = load_pipeline(args.model_path)
    print("Pipeline loaded.")

    # ── Load all input images ──
    images = [Image.open(p).convert("RGB") for p in args.input_image]
    prompt = args.prompt
    print(f"Loaded {len(images)} image(s).")

    # ── Agent: stitch + recaption when needed ──
    need_stitch = len(images) > 3
    need_recaption = args.recaption

    if need_stitch or need_recaption:
        from agent import AgentPipeline

        agent = AgentPipeline(verbose=True)
        agent_result = agent.run(
            images,
            prompt,
            enable_recaption=need_recaption or need_stitch,
        )
        images = agent_result.images
        prompt = agent_result.prompt
        print(f"Agent produced {len(images)} image(s).")
        print(f"Rewritten prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")

    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        result = pipeline(**inputs)

    output_image = result.images[0]
    output_image.save(args.output_image)

    print("Image saved at:", args.output_image.resolve())


if __name__ == "__main__":
    main()