import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent

from models import ClarificationRequest, SimulationOutput, SimulationParams


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")


class PromptExtraction(BaseModel):
    model_type: str | None = None
    substructure_type: str | None = None
    num_images: int | None = None
    source_redshift: float | None = None
    lens_redshift: float | None = None
    resolution: int | None = None


def _normalize_label(value: str) -> str:
    # Normalize free-form text to lowercase snake-style token for robust mapping.
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _fallback_parse(prompt: str) -> Dict:
    # Parse common parameter patterns when LLM extraction is unavailable.
    text = prompt.lower()
    result: Dict = {}

    if "model_ii" in text or "model ii" in text:
        result["model_type"] = "Model_II"
    elif "model_i" in text or "model i" in text:
        result["model_type"] = "Model_I"

    if "no_sub" in text or "no sub" in text or "no substructure" in text:
        result["substructure_type"] = "no_sub"
    elif "subhalo" in text or "cdm" in text or "sphere" in text:
        result["substructure_type"] = "subhalo"

    n_match = re.search(r"(\d+)\s+(?:image|images)", text)
    if n_match:
        result["num_images"] = int(n_match.group(1))

    src_match = re.search(r"source\s*redshift\s*([0-9]*\.?[0-9]+)", text)
    if src_match:
        result["source_redshift"] = float(src_match.group(1))

    lens_match = re.search(r"lens\s*redshift\s*([0-9]*\.?[0-9]+)", text)
    if lens_match:
        result["lens_redshift"] = float(lens_match.group(1))

    res_match = re.search(r"(?:resolution|size)\s*(\d+)", text)
    if res_match:
        result["resolution"] = int(res_match.group(1))

    return result


def parse_user_prompt(prompt: str) -> Dict:
    # Extract simulation parameters from natural language using Pydantic AI + OpenAI.
    model_name = os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        # Fall back to regex extraction if API key is not configured.
        return _fallback_parse(prompt)

    extractor = Agent(
        model_name,
        result_type=PromptExtraction,
        system_prompt=(
            "Extract DeepLense simulation parameters from user text. "
            "Return only what is explicitly provided or strongly implied. "
            "Allowed model_type values: Model_I, Model_II. "
            "Allowed substructure_type values: no_sub, subhalo."
        ),
    )

    try:
        # Run synchronous extraction for simple notebook usage.
        result = extractor.run_sync(prompt)
        data = result.data.model_dump(exclude_none=True)
        return data
    except Exception:
        # Gracefully degrade to deterministic parsing on API/network issues.
        return _fallback_parse(prompt)


def validate_params(params: Dict) -> Union[SimulationParams, ClarificationRequest]:
    # Validate extracted parameters and return either full params or clarification questions.
    normalized = dict(params)

    if "model_type" in normalized and normalized["model_type"] is not None:
        mt = _normalize_label(str(normalized["model_type"]))
        mt_map = {
            "model_i": "Model_I",
            "model_1": "Model_I",
            "i": "Model_I",
            "model_ii": "Model_II",
            "model_2": "Model_II",
            "ii": "Model_II",
        }
        normalized["model_type"] = mt_map.get(mt, normalized["model_type"])

    if "substructure_type" in normalized and normalized["substructure_type"] is not None:
        st = _normalize_label(str(normalized["substructure_type"]))
        st_map = {
            "no_sub": "no_sub",
            "no_substructure": "no_sub",
            "no": "no_sub",
            "subhalo": "subhalo",
            "cdm": "subhalo",
            "sphere": "subhalo",
        }
        normalized["substructure_type"] = st_map.get(st, normalized["substructure_type"])

    required_fields = [
        "model_type",
        "substructure_type",
        "num_images",
        "source_redshift",
        "lens_redshift",
    ]
    missing_fields = [f for f in required_fields if f not in normalized or normalized[f] is None]

    if missing_fields:
        # Build one direct clarification question for each missing required field.
        questions_map = {
            "model_type": "Which model should I use: Model_I or Model_II?",
            "substructure_type": "Which substructure type: no_sub or subhalo?",
            "num_images": "How many images should be generated (1-100)?",
            "source_redshift": "What source redshift should be used (0.1-3.0)?",
            "lens_redshift": "What lens redshift should be used (0.1-1.0)?",
        }
        return ClarificationRequest(
            missing_fields=missing_fields,
            questions=[questions_map[f] for f in missing_fields],
        )

    try:
        # Create strongly validated simulation parameters using Pydantic constraints.
        return SimulationParams(**normalized)
    except ValidationError as exc:
        # Convert validation errors into user-friendly clarification prompts.
        fields = []
        questions = []
        for err in exc.errors():
            field = str(err.get("loc", ["unknown"])[0])
            fields.append(field)
            questions.append(f"Please provide a valid value for '{field}': {err.get('msg', 'invalid value')}")
        return ClarificationRequest(missing_fields=fields, questions=questions)


def _generate_single_image(params: SimulationParams) -> np.ndarray:
    # Run one DeepLenseSim generation pass based on selected model and substructure settings.
    from deeplense.lens import DeepLens

    lens = DeepLens(z_halo=params.lens_redshift, z_gal=params.source_redshift)
    lens.make_single_halo(1e12)

    if params.substructure_type == "no_sub":
        lens.make_no_sub()
    else:
        lens.make_old_cdm()

    if params.model_type == "Model_I":
        lens.make_source_light()
        lens.simple_sim()
    else:
        lens.set_instrument("Euclid")
        lens.make_source_light_mag()
        lens.simple_sim_2()

    return np.asarray(lens.image_real)


def _postprocess_image(image: np.ndarray, resolution: int) -> Image.Image:
    # Normalize raw simulated intensity image to uint8 and resize to requested resolution.
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = image.astype(np.float32)

    min_v = float(image.min())
    max_v = float(image.max())
    if max_v > min_v:
        image = (image - min_v) / (max_v - min_v)
    else:
        image = np.zeros_like(image)

    image_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_u8, mode="L")
    return pil_image.resize((resolution, resolution), Image.Resampling.BILINEAR)


def run_simulation(params: SimulationParams) -> List[str]:
    # Generate and save a batch of lensing images using DeepLenseSim and return file paths.
    output_paths: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for i in range(params.num_images):
        # Generate one image from the selected DeepLense model configuration.
        raw = _generate_single_image(params)
        processed = _postprocess_image(raw, params.resolution)

        filename = (
            f"{params.model_type}_{params.substructure_type}_{timestamp}_{i+1:03d}.png"
        )
        file_path = OUTPUT_DIR / filename
        processed.save(file_path)
        output_paths.append(str(file_path))

    return output_paths


def generate_metadata(params: SimulationParams, paths: List[str]) -> SimulationOutput:
    # Build structured metadata describing generated outputs and simulation settings.
    return SimulationOutput(
        params_used=params,
        images_generated=len(paths),
        image_paths=paths,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="success",
    )


def simulation_output_json(output: SimulationOutput) -> str:
    # Return pretty JSON string for notebook-friendly display.
    return json.dumps(output.model_dump(), indent=2)
