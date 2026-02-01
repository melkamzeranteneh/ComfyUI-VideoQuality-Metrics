"""
Visualization utilities for video quality metrics.

Includes:
- Radar chart generation
- Metric normalization helpers
"""

import torch
from typing import Dict, List, Tuple, Optional
import math


def normalize_metrics(metrics: Dict[str, float], 
                      metric_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Normalize metrics to 0-1 scale for radar chart visualization.
    
    Args:
        metrics: Dictionary of metric_name -> value
        metric_ranges: Optional custom ranges per metric
        
    Returns:
        Dictionary of normalized metrics
    """
    # Default ranges based on typical values
    default_ranges = {
        'psnr': (20.0, 50.0),      # Higher is better
        'ssim': (0.0, 1.0),        # Already 0-1, higher is better
        'ciede2000': (0.0, 50.0),  # Lower is better (will be inverted)
        'warping_error': (0.0, 0.5),  # Lower is better
        'flickering_score': (0.0, 1.0),  # Lower is better
        'smoothness_score': (0.0, 1.0),  # Higher is better
        'fvd': (0.0, 500.0),       # Lower is better
        'fid': (0.0, 300.0),       # Lower is better
    }
    
    ranges = {**default_ranges, **(metric_ranges or {})}
    
    # Metrics where lower is better (need to invert)
    lower_is_better = {'ciede2000', 'warping_error', 'flickering_score', 'fvd', 'fid'}
    
    normalized = {}
    for name, value in metrics.items():
        if name in ranges:
            min_val, max_val = ranges[name]
            norm = (value - min_val) / (max_val - min_val)
            norm = max(0.0, min(1.0, norm))  # Clamp to [0, 1]
            
            if name in lower_is_better:
                norm = 1.0 - norm
            
            normalized[name] = norm
        else:
            # Unknown metric, assume 0-1 range
            normalized[name] = max(0.0, min(1.0, value))
    
    return normalized


def generate_radar_chart_svg(metrics: Dict[str, float], 
                             title: str = "Video Quality Metrics",
                             size: int = 400) -> str:
    """
    Generate SVG radar chart for metric visualization.
    
    Args:
        metrics: Dictionary of metric_name -> normalized_value (0-1)
        title: Chart title
        size: SVG size in pixels
        
    Returns:
        SVG string
    """
    if not metrics:
        return "<svg></svg>"
    
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]
    n = len(labels)
    
    center = size // 2
    radius = size // 2 - 60
    
    # Calculate polygon points
    def polar_to_cart(angle: float, r: float) -> Tuple[float, float]:
        x = center + r * math.cos(angle - math.pi / 2)
        y = center + r * math.sin(angle - math.pi / 2)
        return x, y
    
    angles = [2 * math.pi * i / n for i in range(n)]
    
    # Generate grid lines
    grid_lines = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        points = [polar_to_cart(a, radius * level) for a in angles]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points) + " Z"
        grid_lines.append(f'<path d="{path}" fill="none" stroke="#ddd" stroke-width="1"/>')
    
    # Generate axis lines
    axis_lines = []
    for i, angle in enumerate(angles):
        x, y = polar_to_cart(angle, radius)
        axis_lines.append(f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ccc" stroke-width="1"/>')
    
    # Generate data polygon
    data_points = [polar_to_cart(angles[i], radius * values[i]) for i in range(n)]
    data_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in data_points) + " Z"
    
    # Generate labels
    label_elems = []
    for i, (label, angle) in enumerate(zip(labels, angles)):
        lx, ly = polar_to_cart(angle, radius + 25)
        anchor = "middle"
        if lx < center - 10:
            anchor = "end"
        elif lx > center + 10:
            anchor = "start"
        
        display_label = label.replace('_', ' ').title()
        value_text = f"{values[i]:.2f}"
        label_elems.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="11" font-family="Arial">{display_label}</text>'
        )
    
    # Assemble SVG
    svg = f'''<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{center}" y="25" text-anchor="middle" font-size="16" font-weight="bold" font-family="Arial">{title}</text>
    {"".join(grid_lines)}
    {"".join(axis_lines)}
    <path d="{data_path}" fill="rgba(66, 133, 244, 0.3)" stroke="#4285f4" stroke-width="2"/>
    {"".join(label_elems)}
</svg>'''
    
    return svg


def generate_radar_chart_tensor(metrics: Dict[str, float],
                                 size: int = 256) -> torch.Tensor:
    """
    Generate radar chart as a tensor image for ComfyUI.
    
    Args:
        metrics: Dictionary of metric_name -> normalized_value (0-1)
        size: Image size
        
    Returns:
        Tensor [1, H, W, 3] suitable for ComfyUI IMAGE output
    """
    # Simple rasterization using PyTorch
    n = len(metrics)
    if n == 0:
        return torch.ones(1, size, size, 3)
    
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]
    
    # Create image
    img = torch.ones(size, size, 3)
    
    center = size // 2
    radius = size // 2 - 30
    
    # Draw using simple circle and line approximations
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    
    # Draw grid circles
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        r = int(radius * r_frac)
        for theta in range(360):
            rad = theta * math.pi / 180
            x = int(center + r * math.cos(rad))
            y = int(center + r * math.sin(rad))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = torch.tensor([0.9, 0.9, 0.9])
    
    # Draw data polygon
    points = []
    for i in range(n):
        r = radius * values[i]
        x = int(center + r * math.cos(angles[i]))
        y = int(center + r * math.sin(angles[i]))
        points.append((x, y))
    
    # Draw lines between points
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for s in range(steps + 1):
            x = int(x1 + (x2 - x1) * s / steps)
            y = int(y1 + (y2 - y1) * s / steps)
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = torch.tensor([0.26, 0.52, 0.96])
    
    # Draw center dot
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = center + dy, center + dx
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = torch.tensor([0.2, 0.2, 0.2])
    
    return img.unsqueeze(0)  # [1, H, W, 3]
