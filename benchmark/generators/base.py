import gc
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image


class BaseImageGenerator(ABC):
    def __init__(self, model_name: str, base_output_dir: str):
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.model_pipeline = None
        self.model_output_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(self.model_output_dir, exist_ok=True)

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_single_image(self, prompt: str, seed: int) -> Image.Image:
        raise NotImplementedError

    def unload_model(self) -> None:
        if self.model_pipeline is not None:
            del self.model_pipeline
            self.model_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[{self.model_name}] Released model weights and cleared CUDA cache.")

    def _mark_model_ready(self, ready_file: Optional[str]) -> None:
        if not ready_file:
            return
        with open(ready_file, "w", encoding="utf-8") as handle:
            handle.write(self.model_name)

    def generate_batch(self, prompt_list: list, ready_file: Optional[str] = None) -> None:
        print(f"========== Running generator: {self.model_name} ==========")
        print(f"[{self.model_name}] Loading weights...")
        self.load_model()
        self._mark_model_ready(ready_file)
        print(f"[{self.model_name}] Model is ready.")

        success_count = 0
        for item in prompt_list:
            prompt_id = item["prompt_id"]
            prompt_text = item["prompt"]
            seed = item["seed"]
            output_path = os.path.join(self.model_output_dir, f"{prompt_id}.png")

            if os.path.exists(output_path):
                print(f"[{self.model_name}] Prompt {prompt_id} already exists. Skipping.")
                success_count += 1
                continue

            try:
                image = self.generate_single_image(prompt_text, seed)
                image.save(output_path)
                success_count += 1
                print(f"[{self.model_name}] Generated prompt {prompt_id}.")
            except Exception as exc:
                print(f"[{self.model_name}] Failed on prompt {prompt_id}: {exc}")

        self.unload_model()
        print(f"========== {self.model_name} complete ({success_count}/{len(prompt_list)}) ==========\n")
