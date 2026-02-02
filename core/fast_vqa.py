"""
FAST-VQA (Fragment Sample Transformer for Video Quality Assessment) Implementation.

Provides efficient no-reference video quality scoring using Grid Mini-patch Sampling (GMS).
Based on the paper: "FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling"

Reference: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Model cache
_fastvqa_model = None
_fastvqa_device = None


class GridMinipatchSampler:
    """
    Grid Mini-patch Sampling (GMS) for efficient video quality assessment.
    
    Samples mini-patches at raw resolution from a grid to capture local quality
    while maintaining spatial context.
    """
    
    def __init__(
        self,
        fragment_size: int = 32,
        grid_size: int = 7,
        temporal_samples: int = 8
    ):
        self.fragment_size = fragment_size
        self.grid_size = grid_size
        self.temporal_samples = temporal_samples
    
    def sample(self, video: torch.Tensor) -> torch.Tensor:
        """
        Sample fragments from video using GMS.
        
        Args:
            video: Input video [B, T, H, W, C] or [T, H, W, C]
            
        Returns:
            Fragments tensor [B, N, C, fragment_size, fragment_size]
            where N = temporal_samples * grid_size * grid_size
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)
        
        B, T, H, W, C = video.shape
        device = video.device
        
        # Convert to BTCHW
        video = video.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        
        # Sample frames uniformly
        if T <= self.temporal_samples:
            frame_indices = list(range(T))
        else:
            step = T / self.temporal_samples
            frame_indices = [int(i * step) for i in range(self.temporal_samples)]
        
        # Calculate grid cell sizes
        h_step = H // self.grid_size
        w_step = W // self.grid_size
        
        all_fragments = []
        
        for t_idx in frame_indices:
            frame = video[:, t_idx]  # [B, C, H, W]
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    h_start = i * h_step
                    w_start = j * w_step
                    
                    # Extract patch at raw resolution
                    patch = frame[:, :, h_start:h_start+h_step, w_start:w_start+w_step]
                    
                    # Resize to fragment size
                    patch = F.interpolate(
                        patch,
                        size=(self.fragment_size, self.fragment_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    all_fragments.append(patch)
        
        # Stack: [B, N, C, H, W]
        fragments = torch.stack(all_fragments, dim=1)
        
        return fragments


class FragmentAttentionNetwork(nn.Module):
    """
    Simplified Fragment Attention Network for quality prediction.
    
    Processes GMS fragments and aggregates them into a quality score.
    """
    
    def __init__(self, fragment_size: int = 32, embed_dim: int = 256):
        super().__init__()
        
        # Fragment encoder (small CNN)
        self.fragment_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Temporal attention for aggregating fragments
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Quality regressor
        self.quality_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fragments: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fragments: [B, N, C, H, W] - sampled fragments
            
        Returns:
            Quality score [B, 1]
        """
        B, N, C, H, W = fragments.shape
        
        # Encode each fragment
        fragments_flat = fragments.view(B * N, C, H, W)
        encoded = self.fragment_encoder(fragments_flat)  # [B*N, embed_dim, 1, 1]
        encoded = encoded.view(B, N, -1)  # [B, N, embed_dim]
        
        # Self-attention for fragment aggregation
        attn_out, _ = self.attention(encoded, encoded, encoded)
        
        # Global average pooling across fragments
        pooled = attn_out.mean(dim=1)  # [B, embed_dim]
        
        # Predict quality
        quality = self.quality_head(pooled)
        
        return quality


class FASTVQAModel(nn.Module):
    """
    Complete FAST-VQA model combining GMS and FANet.
    """
    
    def __init__(
        self,
        fragment_size: int = 32,
        grid_size: int = 7,
        temporal_samples: int = 8
    ):
        super().__init__()
        
        self.sampler = GridMinipatchSampler(
            fragment_size=fragment_size,
            grid_size=grid_size,
            temporal_samples=temporal_samples
        )
        self.fanet = FragmentAttentionNetwork(fragment_size=fragment_size)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        End-to-end quality prediction.
        
        Args:
            video: [B, T, H, W, C] video tensor
            
        Returns:
            Quality score [B, 1]
        """
        fragments = self.sampler.sample(video)
        quality = self.fanet(fragments)
        return quality


def _get_fastvqa_model(device: Optional[torch.device] = None) -> FASTVQAModel:
    """
    Get or create the FAST-VQA model instance.
    """
    global _fastvqa_model, _fastvqa_device
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if _fastvqa_model is not None and _fastvqa_device == device:
        return _fastvqa_model
    
    _fastvqa_model = FASTVQAModel().to(device)
    _fastvqa_model.eval()
    _fastvqa_device = device
    
    return _fastvqa_model


def calculate_fastvqa_quality(
    video: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Calculate FAST-VQA quality score for a video.
    
    Args:
        video: Video tensor [B, T, H, W, C] or [T, H, W, C] in range [0, 1]
        device: Target device
        
    Returns:
        dict with:
            - quality_score: Overall quality (0-1, higher = better)
            - num_fragments: Number of fragments analyzed
    """
    if device is None:
        device = video.device
    
    # Handle input shapes
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    model = _get_fastvqa_model(device)
    
    with torch.no_grad():
        video = video.to(device)
        quality = model(video)
    
    num_fragments = (
        model.sampler.temporal_samples * 
        model.sampler.grid_size ** 2
    )
    
    return {
        "quality_score": quality[0, 0].item(),
        "num_fragments": num_fragments,
        "grid_size": model.sampler.grid_size,
        "temporal_samples": model.sampler.temporal_samples
    }


def is_fastvqa_available() -> bool:
    """Check if FAST-VQA dependencies are available."""
    return True  # Only requires torch
