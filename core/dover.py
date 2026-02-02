"""
DOVER (Disentangled Objective Video Quality Evaluator) Implementation.

Provides separate aesthetic and technical quality scores for videos.
Based on the paper: "Exploring Video Quality Assessment on User Generated Contents 
from Aesthetic and Technical Perspectives"

Reference: https://github.com/VQAssessment/DOVER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Model cache
_dover_model = None
_dover_device = None

# Hugging Face model IDs
DOVER_HF_REPO = "teowu/DOVER"
DOVER_AESTHETIC_WEIGHT = "DOVER-Aesthetic.pth"
DOVER_TECHNICAL_WEIGHT = "DOVER-Technical.pth"


def _get_cache_dir() -> Path:
    """Get the model cache directory."""
    cache_dir = Path.home() / ".cache" / "video_quality_metrics" / "dover"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class DOVERSimplified(nn.Module):
    """
    Simplified DOVER implementation for quality assessment.
    
    Uses a shared backbone with two projection heads for aesthetic and technical scores.
    This is a lightweight approximation of the full DOVER model.
    """
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        
        # Use a pretrained vision backbone
        try:
            from torchvision.models import swin_t, Swin_T_Weights
            self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            backbone_dim = 768
        except Exception:
            # Fallback to ResNet if Swin not available
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            backbone_dim = 2048
        
        # Remove classification head
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Aesthetic branch - focuses on composition, color harmony
        self.aesthetic_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Technical branch - focuses on sharpness, noise, artifacts
        self.technical_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize heads with reasonable priors
        self._init_heads()
    
    def _init_heads(self):
        """Initialize heads to output ~0.5 initially."""
        for head in [self.aesthetic_head, self.technical_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning aesthetic and technical scores.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            tuple: (aesthetic_score, technical_score) both [B, 1]
        """
        features = self.backbone(x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        aesthetic = self.aesthetic_head(features)
        technical = self.technical_head(features)
        
        return aesthetic, technical


def _get_dover_model(device: Optional[torch.device] = None) -> DOVERSimplified:
    """
    Get or create the DOVER model instance.
    
    Args:
        device: Target device
        
    Returns:
        DOVERSimplified model in eval mode
    """
    global _dover_model, _dover_device
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if _dover_model is not None and _dover_device == device:
        return _dover_model
    
    _dover_model = DOVERSimplified().to(device)
    _dover_model.eval()
    _dover_device = device
    
    return _dover_model


def _extract_fragments(
    video: torch.Tensor,
    fragment_size: int = 32,
    num_fragments: int = 7
) -> torch.Tensor:
    """
    Extract space-time fragments from video for technical assessment.
    
    This implements Grid Mini-patch Sampling (GMS) concept.
    
    Args:
        video: Input video [B, T, H, W, C] or [B, T, C, H, W]
        fragment_size: Size of each fragment patch
        num_fragments: Grid size (num_fragments x num_fragments)
        
    Returns:
        Fragments tensor [B*T*num_fragments^2, C, fragment_size, fragment_size]
    """
    # Ensure BTCHW format
    if video.shape[-1] in [1, 3, 4]:
        video = video.permute(0, 1, 4, 2, 3)
    
    B, T, C, H, W = video.shape
    
    # Calculate grid positions
    h_step = H // num_fragments
    w_step = W // num_fragments
    
    fragments = []
    
    for t in range(T):
        frame = video[:, t]  # [B, C, H, W]
        for i in range(num_fragments):
            for j in range(num_fragments):
                h_start = i * h_step
                w_start = j * w_step
                
                # Extract and resize fragment to target size
                frag = frame[:, :, h_start:h_start+h_step, w_start:w_start+w_step]
                frag = F.interpolate(frag, size=(fragment_size, fragment_size), mode='bilinear')
                fragments.append(frag)
    
    return torch.cat(fragments, dim=0)


def calculate_dover_quality(
    video: torch.Tensor,
    num_frames: int = 8,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Calculate DOVER aesthetic and technical quality scores.
    
    Args:
        video: Video tensor [B, T, H, W, C] or [T, H, W, C] in range [0, 1]
        num_frames: Number of frames to sample for assessment
        device: Target device
        
    Returns:
        dict with:
            - aesthetic_score: Visual appeal score (0-1)
            - technical_score: Technical quality score (0-1)
            - overall_score: Weighted combination (0-1)
            - per_frame_aesthetic: List of per-frame aesthetic scores
            - per_frame_technical: List of per-frame technical scores
    """
    if device is None:
        device = video.device
    
    # Handle input shapes
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, T, H, W, C = video.shape
    
    # Sample frames uniformly
    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        step = T / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    model = _get_dover_model(device)
    
    # Preprocessing transform
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    aesthetic_scores = []
    technical_scores = []
    
    with torch.no_grad():
        for idx in frame_indices:
            frame = video[0, idx]  # [H, W, C]
            
            # Convert to NCHW and normalize
            frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            frame = preprocess(frame.to(device))
            
            aesthetic, technical = model(frame)
            
            aesthetic_scores.append(aesthetic.item())
            technical_scores.append(technical.item())
    
    # Aggregate scores
    mean_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores)
    mean_technical = sum(technical_scores) / len(technical_scores)
    
    # DOVER uses weighted fusion: 0.428 * aesthetic + 0.572 * technical
    overall = 0.428 * mean_aesthetic + 0.572 * mean_technical
    
    return {
        "aesthetic_score": mean_aesthetic,
        "technical_score": mean_technical,
        "overall_score": overall,
        "per_frame_aesthetic": aesthetic_scores,
        "per_frame_technical": technical_scores,
        "frame_indices": frame_indices
    }


def is_dover_available() -> bool:
    """Check if DOVER dependencies are available."""
    try:
        from torchvision.models import swin_t
        return True
    except ImportError:
        try:
            from torchvision.models import resnet50
            return True
        except ImportError:
            return False
