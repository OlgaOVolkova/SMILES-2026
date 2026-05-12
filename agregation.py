"""
CHANGES:
    - aggregation:
        multi-layer fusion is active
        pooling methods (mean, max, weighted) are commented out
        

    - geometric / statistical features:
        inter-layer cosine similarity is active
        length and layer-wise activation norms are commented out
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    """
                                #ORIGINE
    layer = hidden_states[-1] 
    real_positions = attention_mask.nonzero(as_tuple=False)  
    last_pos = int(real_positions[-1].item())                 
    feature = layer[last_pos] 
    """   
    
    """
                                #MEAN    
    layer = hidden_states[-1]  # (seq_len, hidden_dim)
    real_mask = attention_mask.bool()
    real_tokens = layer[real_mask]  # (n_real, hidden_dim)
    feature = real_tokens.mean(dim=0)  # (hidden_dim,)
    """

    """
                                #MAX     
    layer = hidden_states[-1]
    real_mask = attention_mask.bool()
    real_tokens = layer[real_mask]
    feature = real_tokens.max(dim=0)[0]  # (hidden_dim,)
    """

    """
                                #WEIGHTED     
    layer = hidden_states[-1]
    real_mask = attention_mask.bool()
    real_tokens = layer[real_mask]
    n_real = real_tokens.shape[0]
    
    weights = torch.arange(1, n_real + 1, device=layer.device).float()
    weights = weights / weights.sum()
    
    feature = (real_tokens * weights.unsqueeze(1)).sum(dim=0)  # (hidden_dim,)
    """


                                #MULTI-LAYER     
    last_pos = attention_mask.nonzero()[-1].item()
    
    features = []
    for layer_idx in [-1, -2, -3, -4]:
        layer = hidden_states[layer_idx]
        features.append(layer[last_pos])
    
    feature = torch.cat(features)  # 896*4 = 3584


    return feature

    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:


    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.

    # Placeholder: returns an empty tensor (no geometric features).
    """
                         # length
    device = hidden_states.device
    n_real = attention_mask.sum().item()
    length = n_real / 512.0
    return torch.tensor([length], device=device)
    """
    """
                         # layer-wise activation norms
    device = hidden_states.device
    last_pos = attention_mask.nonzero()[-1].item()
    
    norms = torch.stack([h[last_pos].norm() for h in hidden_states])
    norm_mean = norms.mean()
    norm_std = norms.std()
    
    norm_mean_norm = torch.tanh(norm_mean / 100.0)
    norm_std_norm = torch.tanh(norm_std / 50.0)
    
    return torch.tensor([norm_mean_norm, norm_std_norm], device=device)
    """
                          # inter-layer cosine similarity
    device = hidden_states.device
    last_pos = attention_mask.nonzero()[-1].item()
    
    cos_sims = []
    for i in range(len(hidden_states) - 1):
        vec1 = hidden_states[i][last_pos]
        vec2 = hidden_states[i+1][last_pos]
        cos = torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        cos_sims.append(cos)
    
    cos_mean = torch.tensor(cos_sims, device=device).mean()
    cos_mean_norm = (cos_mean + 1) / 2 
    
    return torch.tensor([cos_mean_norm], device=device)
    # ------------------------------------------------------------------

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:

    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
