"""Differentiable simulation and information analysis of a reconstituted
ten-enzyme EMP glycolysis pathway."""


def init_taichi(arch: str = "cuda") -> None:
    """Single-precision Taichi on the GPU ('cuda') or the CPU ('cpu')."""
    import taichi as ti
    ti.init(arch={"cuda": ti.cuda, "cpu": ti.cpu}[arch], default_fp=ti.f32,
            advanced_optimization=False, offline_cache=False)
