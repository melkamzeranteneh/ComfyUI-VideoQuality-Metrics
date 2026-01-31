import torch
import torch.nn.functional as F
import numpy as np

def tensor_to_hchw(tensor):
    """
    Converts ComfyUI tensor [B, H, W, C] to [B, C, H, W]
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.permute(0, 3, 1, 2)

def tensor_to_bhwc(tensor):
    """
    Converts [B, C, H, W] to [B, H, W, C]
    """
    return tensor.permute(0, 2, 3, 1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def rgb_to_lab(rgb):
    """
    Converts RGB tensor [B, H, W, 3] to LAB tensor [B, H, W, 3]
    Assumes RGB is in range [0, 1]
    """
    # 1. RGB to sRGB (assume it's already sRGB if in [0, 1])
    # 2. sRGB to linear RGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # 3. Linear RGB to XYZ (D65)
    m = torch.tensor([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ], device=rgb.device)
    xyz = torch.matmul(rgb_linear, m.t())
    
    # 4. XYZ to CIELAB
    # Relative to D65 white point
    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=rgb.device)
    xyz_normalized = xyz / xyz_ref
    
    mask = xyz_normalized > 0.008856
    f_xyz = torch.where(mask, xyz_normalized ** (1/3), 7.787 * xyz_normalized + 16/116)
    
    l = (116 * f_xyz[..., 1]) - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    return torch.stack([l, a, b], dim=-1)
