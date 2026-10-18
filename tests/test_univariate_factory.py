"""The benchmark still builds the same nets with the same settings."""

import importlib.util
from pathlib import Path

import nnts
import nnts.benchmark
import nnts.torch.models.dlinear
import nnts.torch.models.nhits
import nnts.torch.models.nlinear
import nnts.torch.models.tide
import nnts.torch.models.tsmixer

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "benchmarking"
    / "run_exp_univariate.py"
)


def _script():
    spec = importlib.util.spec_from_file_location("run_exp_univariate", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata():
    return nnts.datasets.Metadata(
        filename="tourism_quarterly_dataset.tsf",
        dataset="tourism_quarterly",
        context_length=5,
        prediction_length=8,
        freq="QE",
        seasonality=4,
    )


def test_the_benchmark_script_uses_the_package_functions():
    script = _script()
    assert script.model_factory is nnts.benchmark.model_factory
    assert script.get_hyperparams is nnts.benchmark.get_hyperparams


def test_the_five_monash_nets_keep_their_settings():
    metadata = _metadata()
    expected = {
        "nhits": (nnts.torch.models.NHITS, nnts.torch.models.nhits.Hyperparams(), None),
        "tide": (nnts.torch.models.TiDE, nnts.torch.models.tide.Hyperparams(), None),
        "tsmixer": (
            nnts.torch.models.TSMixer,
            nnts.torch.models.tsmixer.Hyperparams(output_channels=1),
            1,
        ),
        "dlinear": (nnts.torch.models.DLinear, nnts.torch.models.dlinear.Hyperparams(), None),
        "nlinear": (nnts.torch.models.NLinear, nnts.torch.models.nlinear.Hyperparams(), None),
    }
    for name, (cls, params, channels) in expected.items():
        built = nnts.benchmark.get_hyperparams(name, metadata)
        assert built.lr == params.lr
        assert built.epochs == params.epochs
        assert built.batch_size == params.batch_size
        assert built.batches_per_epoch == params.batches_per_epoch
        net = nnts.benchmark.model_factory(name, metadata, 1, built)
        assert isinstance(net, cls)
        if channels is not None:
            assert built.output_channels == channels
