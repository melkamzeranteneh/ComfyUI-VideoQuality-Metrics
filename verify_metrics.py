import torch
import torch.nn.functional as F
import metrics
import utils

# Patch metrics for standalone test
metrics.create_window = utils.create_window
metrics.tensor_to_hchw = utils.tensor_to_hchw
metrics.rgb_to_lab = utils.rgb_to_lab

def test_metrics():
    # Create two identical images
    img1 = torch.rand((1, 256, 256, 3))
    img2 = img1.clone()
    
    print("Testing identical images:")
    psnr = metrics.calculate_psnr(img1, img2).item()
    ssim = metrics.calculate_ssim(img1, img2).item()
    ciede = metrics.calculate_ciede2000(img1, img2).item()
    
    print(f"PSNR: {psnr:.2f} (Expected: 100.00)")
    print(f"SSIM: {ssim:.4f} (Expected: 1.0000)")
    print(f"CIEDE2000: {ciede:.2f} (Expected: 0.00)")
    
    # Create a slightly different image (noise)
    img3 = img1 + 0.05 * torch.randn_like(img1)
    img3 = torch.clamp(img3, 0, 1)
    
    print("\nTesting noisy image:")
    psnr_noisy = metrics.calculate_psnr(img1, img3).item()
    ssim_noisy = metrics.calculate_ssim(img1, img3).item()
    ciede_noisy = metrics.calculate_ciede2000(img1, img3).item()
    
    print(f"PSNR: {psnr_noisy:.2f}")
    print(f"SSIM: {ssim_noisy:.4f}")
    print(f"CIEDE2000: {ciede_noisy:.2f}")

    # Create a color shifted image
    img4 = img1.clone()
    img4[..., 0] += 0.1 # Increase red channel
    img4 = torch.clamp(img4, 0, 1)
    
    print("\nTesting color-shifted image:")
    psnr_color = metrics.calculate_psnr(img1, img4).item()
    ssim_color = metrics.calculate_ssim(img1, img4).item()
    ciede_color = metrics.calculate_ciede2000(img1, img4).item()
    
    print(f"PSNR: {psnr_color:.2f}")
    print(f"SSIM: {ssim_color:.4f}")
    print(f"CIEDE2000: {ciede_color:.2f}")

if __name__ == "__main__":
    test_metrics()
