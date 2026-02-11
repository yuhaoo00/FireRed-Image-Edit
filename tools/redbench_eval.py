"""RedBench evaluation script using Gemini API."""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from PIL import Image
from tqdm import tqdm

# Configure Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "REPLACE_WITH_YOUR_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-3.0-flash",
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 8192,
        "top_p": 0.95,
    },
)


def extract_scores_and_average(entry: str) -> float | None:
    """Extract scores from entry string and compute average."""
    if not isinstance(entry, str):
        print(entry)
        return None

    lines = entry.splitlines()
    scores = []

    for line in lines:
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))

    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def compute_averages(result_json_dict: dict) -> dict:
    """Compute averages for all entries in result dict."""
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
    return result


def compute_edit_type_averages(score_dict: dict, meta_list: list) -> dict:
    """Compute averages grouped by edit type."""
    edit_type_scores = defaultdict(list)

    for idx, score in score_dict.items():
        meta = meta_list[int(idx)]
        edit_type = meta.get("task")
        if edit_type is not None:
            edit_type_scores[edit_type].append(score)

    averaged_by_type = {
        etype: round(sum(scores) / len(scores), 2)
        for etype, scores in edit_type_scores.items()
        if scores
    }
    return averaged_by_type


def load_prompts(prompts_json_path: str) -> dict:
    """Load prompts from JSON file."""
    with open(prompts_json_path, "r") as f:
        return json.load(f)


def call_gemini(
    original_image_path: str,
    result_image_path: str,
    edit_prompt: str,
    edit_type: str,
    prompts: dict,
) -> str:
    """Call Gemini API to evaluate image edit."""
    try:
        # Load images using PIL
        original_image = Image.open(original_image_path)
        result_image = Image.open(result_image_path)

        prompt_template = prompts[edit_type]
        full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

        # Build content with images and text
        content = [
            original_image,
            "This is the original image A",
            result_image,
            "This is the edited image B. Please evaluate.",
            full_prompt,
        ]

        response = model.generate_content(content)
        return response.text

    except Exception as e:
        print(f"Error in calling Gemini API: {e}")
        raise


def process_single_item(
    idx: int,
    item: dict,
    result_img_folder: str,
    prompts: dict,
) -> tuple[str, str]:
    """Process a single edit item."""
    # Build result image path
    result_img_name = item["id"] + ".png"
    task = item["task"]
    result_img_path = os.path.join(result_img_folder, task, result_img_name)

    # Original image path
    origin_img_path = item["source"]

    # Select prompt: prefer Chinese, otherwise English
    edit_prompt = item.get("a_to_b_instructions", "")
    if not edit_prompt:
        edit_prompt = item.get("a_to_b_instructions_eng", "")

    # Edit type
    edit_type = item.get("task", "default")
    res = call_gemini(origin_img_path, result_img_path, edit_prompt, edit_type, prompts)
    print(idx, res)
    return str(idx), res


def process_json(
    edit_infos: list,
    result_img_folder: str,
    num_threads: int,
    prompts: dict,
) -> dict:
    """Process all edit items with thread pool."""
    results = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(
                process_single_item, idx, item, result_img_folder, prompts
            ): idx
            for idx, item in enumerate(edit_infos)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing edits",
        ):
            idx = future_to_idx[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing idx {idx}: {e}")
                results[str(idx)] = {"error": str(e)}

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_img_folder",
        type=str,
        required=True,
        help="Folder containing generated result images",
    )
    parser.add_argument(
        "--edit_json",
        type=str,
        required=True,
        help="Path to edit info jsonl file",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        required=True,
        help="Path to prompts json file",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=50,
        help="Number of concurrent threads",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)

    # Read jsonl file
    edit_infos = []
    with open(args.edit_json, "r") as f:
        for line in f:
            if line.strip():
                edit_infos.append(json.loads(line))

    print(f"Loaded {len(edit_infos)} entries")

    data = process_json(edit_infos, args.result_img_folder, args.num_threads, prompts)
    averaged_data = compute_averages(data)
    averaged_result = compute_edit_type_averages(averaged_data, edit_infos)

    if averaged_result:
        scores = list(averaged_result.values())
        final_score = sum(scores) / len(scores)
    else:
        final_score = 0

    print(averaged_result)
    print(f"Final Score: {final_score}")

    results_path = os.path.join(args.result_img_folder, "result.json")
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    scores_path = os.path.join(args.result_img_folder, "score.json")
    with open(scores_path, "w") as f:
        json.dump(
            {
                "final_score": final_score,
                "averaged_result": averaged_result,
                "averaged_data": averaged_data,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()