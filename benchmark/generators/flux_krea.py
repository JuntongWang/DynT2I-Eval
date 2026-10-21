from __future__ import annotations

from benchmark.generators.registered import RegisteredDiffusersGenerator


class FluxKreaGenerator(RegisteredDiffusersGenerator):
    def __init__(self, base_output_dir: str, config_path: str | None = None):
        super().__init__("FLUX.1-Krea-dev", base_output_dir, config_path=config_path)
