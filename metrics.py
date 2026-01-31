import torch
import torch.nn.functional as F
try:
    from .utils import create_window, tensor_to_hchw
except ImportError:
    from utils import create_window, tensor_to_hchw

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculates PSNR between two images.
    img1, img2: Tensors of shape [B, H, W, C] in range [0, 1]
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculates SSIM between two images.
    img1, img2: Tensors of shape [B, H, W, C] in range [0, 1]
    """
    # Convert to NCHW
    img1 = tensor_to_hchw(img1)
    img2 = tensor_to_hchw(img2)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ciede2000(img1, img2):
    """
    Calculates CIEDE2000 color difference between two images.
    img1, img2: Tensors of shape [B, H, W, 3] in range [0, 1]
    """
    try:
        from .utils import rgb_to_lab
    except ImportError:
        from utils import rgb_to_lab
    
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    # Delta E 2000 algorithm
    # Reference: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
    
    kL, kC, kH = 1, 1, 1
    
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    a1_p = (1 + G) * a1
    a2_p = (1 + G) * a2
    
    C1_p = torch.sqrt(a1_p**2 + b1**2)
    C2_p = torch.sqrt(a2_p**2 + b2**2)
    
    h1_p = torch.atan2(b1, a1_p) * 180 / torch.pi
    h1_p = torch.where(h1_p < 0, h1_p + 360, h1_p)
    
    h2_p = torch.atan2(b2, a2_p) * 180 / torch.pi
    h2_p = torch.where(h2_p < 0, h2_p + 360, h2_p)
    
    dL_p = L2 - L1
    dC_p = C2_p - C1_p
    
    dh_p = h2_p - h1_p
    dh_p = torch.where(torch.abs(dh_p) > 180, torch.where(h2_p <= h1_p, dh_p + 360, dh_p - 360), dh_p)
    
    dH_p = 2 * torch.sqrt(C1_p * C2_p) * torch.sin(dh_p / 2 * torch.pi / 180)
    
    L_bar_p = (L1 + L2) / 2
    C_bar_p = (C1_p + C2_p) / 2
    
    h_bar_p = torch.where(torch.abs(h1_p - h2_p) > 180, (h1_p + h2_p + 360) / 2, (h1_p + h2_p) / 2)
    h_bar_p = torch.where(h_bar_p >= 360, h_bar_p - 360, h_bar_p)
    
    T = 1 - 0.17 * torch.cos((h_bar_p - 30) * torch.pi / 180) + \
        0.24 * torch.cos(2 * h_bar_p * torch.pi / 180) + \
        0.32 * torch.cos((3 * h_bar_p + 6) * torch.pi / 180) - \
        0.20 * torch.cos((4 * h_bar_p - 63) * torch.pi / 180)
    
    d_theta = 30 * torch.exp(-((h_bar_p - 275) / 25)**2)
    Rc = 2 * torch.sqrt(C_bar_p**7 / (C_bar_p**7 + 25**7))
    RT = -torch.sin(2 * d_theta * torch.pi / 180) * Rc
    
    SL = 1 + (0.015 * (L_bar_p - 50)**2) / torch.sqrt(20 + (L_bar_p - 50)**2)
    SC = 1 + 0.045 * C_bar_p
    SH = 1 + 0.015 * C_bar_p * T
    
    dE = torch.sqrt((dL_p / (kL * SL))**2 + (dC_p / (kC * SC))**2 + (dH_p / (kH * SH))**2 + RT * (dC_p / (kC * SC)) * (dH_p / (kH * SH)))
    
    return dE.mean()
