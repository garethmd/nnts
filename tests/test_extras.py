"""Install contract for consumers of nnts.

Core is pandas, pydantic, and requests (nnts.data.tsf downloads the
Monash files with it). Torch is optional, on a range, and the
torch extra does not bring the plotting or experiment-tracking
packages. Notebooks keep the previous stack through the all extra,
including scipy, which is not part of any narrower group.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VIZ = frozenset({"matplotlib", "seaborn", "plotly"})
WANDB = frozenset({"wandb"})
FOUNDATION = frozenset({"transformers"})
# scipy stays available to notebooks via nnts[all]. It is not a torch,
# plotting, tracking, or transformers dependency.
SCIPY = frozenset({"scipy"})


def _poetry():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["poetry"]


def test_core_dependencies_are_pandas_pydantic_and_requests():
    dependencies = _poetry()["dependencies"]
    required = {
        name
        for name, spec in dependencies.items()
        if name != "python" and not (isinstance(spec, dict) and spec.get("optional"))
    }
    assert required == {"pandas", "pydantic", "requests"}


def test_torch_only_import_chain_has_no_top_level_optional_imports():
    # nnts.torch.trainers imports nnts.loggers; the plotting and wandb
    # imports there must stay inside the methods that use them.
    optional = ("matplotlib", "seaborn", "wandb", "plotly", "scipy", "transformers")
    for module in ("nnts/loggers.py", "nnts/torch/trainers.py", "nnts/datasets.py"):
        for line in (ROOT / module).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not line.startswith(" "):
                assert not any(
                    stripped.startswith((f"import {name}", f"from {name}"))
                    for name in optional
                ), f"{module}: {stripped}"


def test_torch_is_optional_on_the_two_four_range():
    spec = _poetry()["dependencies"]["torch"]
    assert spec["optional"] is True
    # 2.6.0 moves the Monash N-HITS numbers, so it stays outside the range.
    assert spec["version"] == ">=2.4,<2.6"


def test_torch_extra_brings_torch_and_not_viz_or_wandb():
    torch_extra = set(_poetry()["extras"]["torch"])
    assert torch_extra == {"torch"}
    assert torch_extra.isdisjoint(VIZ | WANDB | FOUNDATION | SCIPY)


def test_named_extras_are_viz_wandb_foundation_and_all():
    extras = _poetry()["extras"]
    assert set(extras["viz"]) == VIZ
    assert set(extras["wandb"]) == WANDB
    assert set(extras["foundation"]) == FOUNDATION
    assert set(extras["all"]) == {"torch"} | VIZ | WANDB | FOUNDATION | SCIPY
