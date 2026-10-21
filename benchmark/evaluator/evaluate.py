from __future__ import annotations

import json
import re

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from benchmark.runtime import get_config_value, load_runtime_config, require_runtime_path


class QwenVLMJudge:
    def __init__(self, model_path: str | None = None, config_path: str | None = None):
        config = load_runtime_config(config_path)
        resolved_model_path = model_path or get_config_value("model_paths", "Qwen3.5-9B", config=config)
        self.model_path = require_runtime_path(resolved_model_path, "model_paths.Qwen3.5-9B")
        self.device = get_config_value("devices", "alignment_judge", default="auto", config=config)
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        print(f"[QwenVLMJudge] Loading model from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        print("[QwenVLMJudge] Model loaded successfully.")

    def unload_model(self) -> None:
        if self.model is None:
            return
        print("[QwenVLMJudge] Releasing model and CUDA cache...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        print(f"[QwenVLMJudge] Failed to parse JSON output:\n{text}")
        return {"analysis_A": "Parse error.", "analysis_B": "Parse error.", "winner": "Tie"}

    def _build_judge_instruction(self, prompt_text: str, checklist: list[str]) -> str:
        checklist_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(checklist))
        return f"""You are an impartial judge for text-image alignment.
Your task is to compare Image A and Image B generated from the same prompt.

Evaluation policy:
1. Use the Text Prompt as the primary task definition.
2. Use the Checklist as grounding anchors for the key visible constraints.
3. Judge only text-image alignment. Do not reward general beauty, mood, style, sharpness, or realism unless they directly affect whether the prompt constraints are visibly satisfied.
4. Prefer the image that better satisfies the prompt's core visible constraints, even if the other image looks more attractive overall.
5. If both images satisfy the prompt to a similar degree, or the evidence is insufficient, return Tie.

[Text Prompt]
{prompt_text}

[Checklist]
{checklist_text}

Return STRICT JSON only:
{{
    "analysis_A": "one short sentence about Image A's alignment",
    "analysis_B": "one short sentence about Image B's alignment",
    "winner": "A" or "B" or "Tie"
}}"""

    def _run_single_judgment(self, physical_img_a: str, physical_img_b: str, judge_instruction: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": judge_instruction + "\n\nCompare the two images below.\n\n"},
                    {"type": "image", "image": physical_img_a},
                    {"type": "text", "text": "This is Image A.\n\n"},
                    {"type": "image", "image": physical_img_b},
                    {"type": "text", "text": "This is Image B.\n\nReturn JSON only."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._extract_json(output_text)

    def evaluate_pair(self, image_a_path: str, image_b_path: str, prompt_text: str, checklist: list[str]) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        judge_instruction = self._build_judge_instruction(prompt_text, checklist)
        forward = self._run_single_judgment(image_a_path, image_b_path, judge_instruction)
        backward_raw = self._run_single_judgment(image_b_path, image_a_path, judge_instruction)
        backward = {
            "analysis_A": backward_raw.get("analysis_B", ""),
            "analysis_B": backward_raw.get("analysis_A", ""),
            "winner": "Tie",
        }
        if backward_raw.get("winner") == "A":
            backward["winner"] = "B"
        elif backward_raw.get("winner") == "B":
            backward["winner"] = "A"

        winner_forward = forward.get("winner", "Tie")
        winner_backward = backward.get("winner", "Tie")
        winner_set = {winner_forward, winner_backward}

        if winner_forward == winner_backward:
            winner = winner_forward
            confidence = "high" if winner != "Tie" else "medium"
        elif "A" in winner_set and "B" in winner_set:
            winner = "Tie"
            confidence = "low"
        else:
            winner = "A" if "A" in winner_set else "B"
            confidence = "medium"

        return {
            "analysis_A": forward.get("analysis_A") or backward.get("analysis_A", ""),
            "analysis_B": forward.get("analysis_B") or backward.get("analysis_B", ""),
            "winner": winner,
            "confidence": confidence,
            "swap_consistent": winner_forward == winner_backward,
            "pass_1_winner": winner_forward,
            "pass_2_winner": winner_backward,
        }
