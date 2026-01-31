# ComfyUI Video Quality Metrics

A comprehensive suite of video quality assessment metrics for ComfyUI, based on state-of-the-art research in perceptual quality evaluation.

## Features

### Implemented Metrics ✅
- **PSNR** (Peak Signal-to-Noise Ratio) - Pixel-level fidelity
- **SSIM** (Structural Similarity Index) - Structural fidelity  
- **CIEDE2000** - Perceptually uniform color accuracy

### Coming Soon
- **Temporal Metrics**: Optical flow, warping error, flickering detection
- **Distributional Metrics**: FVD, JEDi
- **Visualization**: Radar charts, statistical reports

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-VideoQuality-Metrics.git
```

2. Install dependencies
```bash
cd ComfyUI-VideoQuality-Metrics
pip install torch torchvision numpy
```

3. Restart ComfyUI

## Usage

### VQ Full-Reference Metrics Node

This node compares two image/video batches and outputs quality metrics.

**Inputs**:
- `images`: Generated or processed frames
- `reference`: Ground truth frames

**Outputs**:
- `psnr`: PSNR value in dB (higher is better)
- `ssim`: SSIM value 0-1 (1 = perfect)
- `ciede2000`: Color difference ΔE (lower is better)
- `summary`: Formatted text summary

**Interpretation Guidelines**:

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| PSNR | > 35 dB | 30-35 dB | 25-30 dB | < 25 dB |
| SSIM | > 0.95 | 0.90-0.95 | 0.85-0.90 | < 0.85 |
| ΔE2000 | < 1.0 | 1.0-2.0 | 2.0-5.0 | > 5.0 |

## Technical Documentation

See [doc.md](doc.md) for a comprehensive analysis of video quality metrics, including:
- Signal-level fidelity (PSNR, SSIM, VMAF)
- Distributional realism (FVD, JEDi)
- Temporal consistency (warping error, flickering)
- Color accuracy (CIEDE2000)
- Statistical significance testing

## Development

Run the verification script to test the implementation:
```bash
python verify_metrics.py
```

Expected output for identical images:
```
PSNR: 100.00 dB
SSIM: 1.0000
CIEDE2000: 0.00
```

## References

This implementation is based on research from:
- CIEDE2000 color difference formula
- Multi-Scale SSIM (Wang et al.)
- VBench and WorldScore benchmarks

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Priority areas:
- VMAF integration (libvmaf packaging)
- Optical flow metrics (RAFT integration)
- Distributional metrics (FVD/JEDi)
