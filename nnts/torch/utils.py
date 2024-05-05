import torch


def count_of_params_in(net: torch.nn.Module):
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return pytorch_total_params
