"""
Statistical utilities for video quality assessment.

Includes:
- Significance testing (Wilcoxon, T-test)
- Confidence intervals
"""

import torch
from typing import Dict, List, Tuple, Optional
import math


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    return mean, std


def t_test(samples1: List[float], samples2: List[float]) -> Dict[str, float]:
    """
    Perform independent two-sample t-test.
    
    Tests if the means of two groups are significantly different.
    
    Args:
        samples1: First group of samples
        samples2: Second group of samples
        
    Returns:
        dict with:
            - t_statistic: T-test statistic
            - p_value: Two-tailed p-value (approximated)
            - significant: Whether p < 0.05
            - mean_diff: Difference in means
    """
    n1, n2 = len(samples1), len(samples2)
    
    if n1 < 2 or n2 < 2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'mean_diff': 0.0
        }
    
    mean1, std1 = mean_std(samples1)
    mean2, std2 = mean_std(samples2)
    
    # Pooled standard error
    se = math.sqrt(std1**2 / n1 + std2**2 / n2)
    
    if se == 0:
        return {
            't_statistic': 0.0,
            'p_value': 1.0 if mean1 == mean2 else 0.0,
            'significant': mean1 != mean2,
            'mean_diff': mean1 - mean2
        }
    
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom (Welch's approximation)
    df_num = (std1**2/n1 + std2**2/n2)**2
    df_denom = (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
    df = df_num / df_denom if df_denom > 0 else n1 + n2 - 2
    
    # Approximate p-value using normal distribution for large df
    # For more accurate results, use scipy.stats.t.sf
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': mean1 - mean2
    }


def wilcoxon_signed_rank(samples1: List[float], samples2: List[float]) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Tests if paired samples have the same distribution.
    
    Args:
        samples1: First group of paired samples
        samples2: Second group of paired samples
        
    Returns:
        dict with:
            - w_statistic: Wilcoxon W statistic
            - p_value: Approximate p-value
            - significant: Whether p < 0.05
    """
    if len(samples1) != len(samples2):
        raise ValueError("Samples must have equal length for paired test")
    
    n = len(samples1)
    if n < 5:
        return {
            'w_statistic': 0.0,
            'p_value': 1.0,
            'significant': False
        }
    
    # Calculate differences
    diffs = [s1 - s2 for s1, s2 in zip(samples1, samples2)]
    
    # Remove zeros
    nonzero_diffs = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    
    if len(nonzero_diffs) == 0:
        return {
            'w_statistic': 0.0,
            'p_value': 1.0,
            'significant': False
        }
    
    # Rank by absolute value
    sorted_diffs = sorted(enumerate(nonzero_diffs), key=lambda x: x[1][0])
    
    # Assign ranks (handling ties with average rank)
    ranks = [0.0] * len(nonzero_diffs)
    i = 0
    while i < len(sorted_diffs):
        j = i
        while j < len(sorted_diffs) and sorted_diffs[j][1][0] == sorted_diffs[i][1][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[sorted_diffs[k][0]] = avg_rank
        i = j
    
    # Calculate W+ and W-
    w_plus = sum(r for r, (_, sign) in zip(ranks, nonzero_diffs) if sign > 0)
    w_minus = sum(r for r, (_, sign) in zip(ranks, nonzero_diffs) if sign < 0)
    
    w_stat = min(w_plus, w_minus)
    
    # Normal approximation for p-value
    n_eff = len(nonzero_diffs)
    mean_w = n_eff * (n_eff + 1) / 4
    std_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
    
    if std_w == 0:
        p_value = 1.0
    else:
        z = (w_stat - mean_w) / std_w
        p_value = 2 * _normal_cdf(z)  # Two-tailed
    
    return {
        'w_statistic': w_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def _normal_cdf(z: float) -> float:
    """
    Approximate cumulative distribution function for standard normal.
    Uses error function approximation.
    """
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def confidence_interval(samples: List[float], 
                        confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for sample mean.
    
    Args:
        samples: List of sample values
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(samples)
    if n < 2:
        mean = samples[0] if n == 1 else 0.0
        return mean, mean, mean
    
    mean, std = mean_std(samples)
    
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    margin = z * std / math.sqrt(n)
    
    return mean, mean - margin, mean + margin


def compare_workflows(metrics_a: Dict[str, List[float]], 
                      metrics_b: Dict[str, List[float]]) -> Dict[str, Dict]:
    """
    Compare two workflows across multiple metrics with significance testing.
    
    Args:
        metrics_a: Dict of metric_name -> list of values for workflow A
        metrics_b: Dict of metric_name -> list of values for workflow B
        
    Returns:
        Dict of metric_name -> comparison results
    """
    results = {}
    
    common_metrics = set(metrics_a.keys()) & set(metrics_b.keys())
    
    for metric in common_metrics:
        vals_a = metrics_a[metric]
        vals_b = metrics_b[metric]
        
        mean_a, std_a = mean_std(vals_a)
        mean_b, std_b = mean_std(vals_b)
        
        # Use t-test for unpaired, Wilcoxon for paired
        if len(vals_a) == len(vals_b):
            test_result = wilcoxon_signed_rank(vals_a, vals_b)
            test_name = "Wilcoxon"
        else:
            test_result = t_test(vals_a, vals_b)
            test_name = "T-test"
        
        results[metric] = {
            'workflow_a': {'mean': mean_a, 'std': std_a, 'n': len(vals_a)},
            'workflow_b': {'mean': mean_b, 'std': std_b, 'n': len(vals_b)},
            'test': test_name,
            'p_value': test_result['p_value'],
            'significant': test_result['significant'],
            'winner': 'A' if mean_a > mean_b else 'B' if mean_b > mean_a else 'Tie'
        }
    
    return results
