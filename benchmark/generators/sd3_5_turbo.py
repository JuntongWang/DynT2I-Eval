from __future__ import annotations

from benchmark.generators.registered import RegisteredDiffusersGenerator


class SD35TurboGenerator(RegisteredDiffusersGenerator):
    def __init__(self, base_output_dir: str, config_path: str | None = None):
        super().__init__("SD3.5-Large-Turbo", base_output_dir, config_path=config_path)
