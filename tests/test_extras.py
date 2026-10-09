"""Install contract for consumers of nnts.

Core is pandas and pydantic. Torch is optional, on a range, and the
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


def test_core_dependencies_are_pandas_and_pydantic():
    dependencies = _poetry()["dependencies"]
    required = {
        name
        for name, spec in dependencies.items()
        if name != "python"
        and not (isinstance(spec, dict) and spec.get("optional"))
    }
    assert required == {"pandas", "pydantic"}


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
