def choose_device(value='auto'):
    import torch
    value = ('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value
    device = torch.device(value)
    if device.type not in ['cpu', 'cuda']:
        raise ValueError('Supported device: cpu or cuda')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable. Install NVIDIA driver/CUDA PyTorch, or use --device cpu')
    if device.type == 'cuda':
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        device = torch.device('cuda', index)
    return device

def amp_context(device):
    import torch
    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()
