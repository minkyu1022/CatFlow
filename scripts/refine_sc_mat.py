import torch
import numpy as np

# def regularize_singular_values(matrix, min_sv=0.1):

#     U, S, Vh = torch.linalg.svd(matrix)

#     S = torch.clamp(S, min=min_sv)
#     return U @ torch.diag(S) @ Vh

def refine_sc_mat(pred_matrix, min_det=1.0):
    """
    Refine supercell matrix to ensure valid determinant.
    
    Args:
        pred_matrix: torch.Tensor of shape (3, 3) or (B, 3, 3)
        min_det: minimum absolute determinant value (default: 1.0)
    
    Returns:
        torch.Tensor of same shape as input, with integer values and valid determinant
    """
    # Handle single matrix case
    if pred_matrix.dim() == 1:
        pred_matrix = pred_matrix.view(3, 3)
    
    # Handle batch case
    if pred_matrix.dim() == 3:
        return refine_sc_mat_batch(pred_matrix, min_det)
    
    # Single matrix case (original logic)
    m_int = torch.round(pred_matrix)
    det = torch.linalg.det(m_int)

    if abs(det) >= min_det:
        return m_int.int()
    
    candidates = []
    
    for i in range(3):
        for j in range(3):
            for delta in [-1, 1]:
                candidate = m_int.clone()
                candidate[i, j] += delta
                
                cand_det = torch.linalg.det(candidate)
                if abs(cand_det) >= min_det:
                    dist = torch.norm(pred_matrix - candidate)
                    candidates.append((dist, candidate))
    
    if not candidates:
        print("Warning: No valid supercell matrix found. Returning rounded itself.")
        return torch.eye(3, device=pred_matrix.device).int()
    
    best_candidate = min(candidates, key=lambda x: x[0])[1]
    
    return best_candidate.int()

def refine_sc_mat_batch(pred_matrices, min_det=1.0):
    """
    Fully vectorized batch version of refine_sc_mat.
    Removes Python loops for GPU parallel processing.
    """
    B = pred_matrices.shape[0]
    device = pred_matrices.device
    dtype = pred_matrices.dtype
    
    # 1. Round all matrices
    m_int = torch.round(pred_matrices)
    
    # 2. Check determinants
    dets = torch.linalg.det(m_int)
    valid_mask = torch.abs(dets) >= min_det
    
    # If all are valid, return immediately
    if valid_mask.all():
        return m_int.int()

    # 3. Identify matrices that need refinement
    # indices of invalid matrices
    invalid_idx = torch.where(~valid_mask)[0] 
    # (K, 3, 3) - K is number of invalid matrices
    m_invalid = m_int[invalid_idx] 
    pred_invalid = pred_matrices[invalid_idx]
    
    # 4. Create all 18 perturbations (deltas) at once
    # Shape: (18, 3, 3)
    deltas = torch.zeros(18, 3, 3, device=device, dtype=dtype)
    cnt = 0
    for i in range(3):
        for j in range(3):
            for d in [-1, 1]:
                deltas[cnt, i, j] = d
                cnt += 1
                
    # 5. Broadcast addition to create candidates
    # m_invalid: (K, 1, 3, 3)
    # deltas:    (1, 18, 3, 3)
    # candidates: (K, 18, 3, 3)
    candidates = m_invalid.unsqueeze(1) + deltas.unsqueeze(0)
    
    # 6. Compute determinants for all candidates at once
    # (K, 18)
    cand_dets = torch.linalg.det(candidates) 
    cand_valid_mask = torch.abs(cand_dets) >= min_det
    
    # 7. Compute distances (Frobenius norm)
    # pred_invalid: (K, 1, 3, 3)
    # candidates:   (K, 18, 3, 3)
    # dists:        (K, 18)
    dists = torch.norm(candidates - pred_invalid.unsqueeze(1), dim=(-2, -1))
    
    # 8. Mask invalid candidates (set distance to infinity)
    # If a candidate has det=0, we shouldn't pick it.
    dists[~cand_valid_mask] = float('inf')
    
    # 9. Select best candidate
    # min_dists: (K,), best_indices: (K,)
    min_dists, best_cand_indices = torch.min(dists, dim=1)
    
    # 10. Gather results
    # We need to pick the (3,3) matrix corresponding to best_cand_indices
    # best_cand_indices shape (K,) -> expand to (K, 1, 1) for gathering? 
    # Easier way: simple indexing since we have flat candidates
    # candidates[k, best_idx[k]]
    
    final_candidates = candidates[torch.arange(len(invalid_idx), device=device), best_cand_indices]
    
    # Check if any matrix failed to find ANY valid neighbor (all dists are inf)
    # In that case, fallback to original rounded matrix
    failed_refinement = torch.isinf(min_dists)
    if failed_refinement.any():
        # Fallback to Identity matrix if refinement fails
        num_failed = failed_refinement.sum()
        identities = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_failed, -1, -1)
        final_candidates[failed_refinement] = identities
    
    # 11. Update result tensor
    result = m_int.clone()
    result[invalid_idx] = final_candidates
    
    return result.int()