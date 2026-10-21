from __future__ import annotations

from benchmark.generators.registered import RegisteredDiffusersGenerator


class Flux2Klein9BGenerator(RegisteredDiffusersGenerator):
    def __init__(self, base_output_dir: str, config_path: str | None = None):
        super().__init__("FLUX.2-klein-9B", base_output_dir, config_path=config_path)
