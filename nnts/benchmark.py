"""The univariate Monash benchmark's net builder.

Moved unchanged from ``projects/benchmarking/run_exp_univariate.py``.
"""

import nnts
import nnts.datasets
import nnts.torch.models


def model_factory(model_name: str, metadata: nnts.datasets.Metadata, enc_in, params):
    if model_name == "dlinear":
        return nnts.torch.models.DLinear(
            metadata.prediction_length, metadata.context_length, enc_in, params
        )
    elif model_name == "nlinear":
        return nnts.torch.models.NLinear(
            metadata.prediction_length, metadata.context_length, enc_in, params
        )
    elif model_name == "nhits":
        return nnts.torch.models.NHITS(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            configs=params,
        )
    elif model_name == "tide":
        return nnts.torch.models.TiDE(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            configs=params,
        )
    elif model_name == "nbeats":
        return nnts.torch.models.NBeats(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            configs=params,
        )
    elif model_name == "tsmixer":
        return nnts.torch.models.TSMixer(
            sequence_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            input_channels=1,
            params=params,
        )
    elif model_name == "timemixer":
        return nnts.torch.models.TimeMixer(
            sequence_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            params=params,
        )
    elif model_name == "patchtsmixer":
        from nnts.torch.models.patchtsmixer import PatchTSMixer

        return PatchTSMixer(
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            params=params,
        )
    elif model_name == "patchtst":
        return nnts.torch.models.PatchTST(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            c_in=1,
            configs=params,
        )
    elif model_name == "iTransformer":
        return nnts.torch.models.iTransformer(
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            configs=params,
        )
    elif model_name == "timesnet":
        return nnts.torch.models.TimesNet(
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            configs=params,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_hyperparams(model_name: str, metadata):
    if model_name == "dlinear":
        return nnts.torch.models.dlinear.Hyperparams()
    elif model_name == "nlinear":
        return nnts.torch.models.nlinear.Hyperparams()
    elif model_name == "nhits":
        return nnts.torch.models.nhits.Hyperparams()
    elif model_name == "tide":
        return nnts.torch.models.tide.Hyperparams()
    elif model_name == "nbeats":
        return nnts.torch.models.nbeats.Hyperparams()
    elif model_name == "tsmixer":
        return nnts.torch.models.tsmixer.Hyperparams(output_channels=1)
    elif model_name == "timemixer":
        return nnts.torch.models.timemixer.Hyperparams(freq=metadata.freq)
    elif model_name == "patchtsmixer":
        from nnts.torch.models.patchtsmixer import Hyperparams

        patch_len = 16 if metadata.context_length > 16 else metadata.context_length // 2
        return Hyperparams(
            patch_length=patch_len, num_input_channels=1
        )
    elif model_name == "patchtst":
        patch_len = 16 if metadata.context_length > 16 else metadata.context_length
        return nnts.torch.models.patchtst.Hyperparams(patch_len=patch_len)
    elif model_name == "iTransformer":
        return nnts.torch.models.itransformer.Hyperparams(freq=metadata.freq)
    elif model_name == "timesnet":
        return nnts.torch.models.timesnet.Hyperparams(freq=metadata.freq.lower())
