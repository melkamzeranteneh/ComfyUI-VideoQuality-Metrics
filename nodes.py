import torch
from .metrics import calculate_psnr, calculate_ssim, calculate_ciede2000

class VQ_FullReferenceMetrics:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("psnr", "ssim", "ciede2000", "summary")
    FUNCTION = "measure"
    CATEGORY = "VideoQuality"

    def measure(self, images, reference):
        if images.shape != reference.shape:
            # Basic check, though PSNR/SSIM often require exact match or resizing
            # For now, let's assume they match or suggest resizing
            return (0.0, 0.0, 0.0, "Error: Image dimensions do not match.")

        psnr_val = calculate_psnr(images, reference).item()
        ssim_val = calculate_ssim(images, reference).item()
        ciede_val = calculate_ciede2000(images, reference).item()
        
        summary = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}\nCIEDE2000: {ciede_val:.2f}"
        
        return (psnr_val, ssim_val, ciede_val, summary)

NODE_CLASS_MAPPINGS = {
    "VQ_FullReferenceMetrics": VQ_FullReferenceMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_FullReferenceMetrics": "VQ Full-Reference Metrics (PSNR/SSIM)",
}
