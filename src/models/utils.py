# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Module,
)
from torch.types import Device

LinearNoBias = partial(Linear, bias=False)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class SwiGLU(Module):
    def forward(
        self,
        x,  #: Float['... d']
    ):  # -> Float[' ... (d//2)']:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


def center(atom_coords, atom_mask):
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


def compute_random_augmentation(
    multiplicity, s_trans=1.0, device=None, dtype=torch.float32
):
    R = random_rotations(multiplicity, dtype=dtype, device=device)
    random_trans = (
        torch.randn((multiplicity, 1, 3), dtype=dtype, device=device) * s_trans
    )
    return R, random_trans


def randomly_rotate(coords, return_second_coords=False, second_coords=None):
    R = random_rotations(len(coords), coords.dtype, coords.device)

    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), torch.einsum(
            "bmd,bds->bms", second_coords, R
        ) if second_coords is not None else None

    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Algorithm 19"""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py, Apache-2.0 license
    Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def compatible(self, parameters):
        if len(self.shadow_params) != len(parameters):
            print(
                f"Model has {len(self.shadow_params)} parameter tensors, the incoming ema {len(parameters)}"
            )
            return False

        for s_param, param in zip(self.shadow_params, parameters):
            if param.data.shape != s_param.data.shape:
                print(
                    f"Model has parameter tensor of shape {s_param.data.shape} , the incoming ema {param.data.shape}"
                )
                return False
        return True

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [
            tensor.to(device) for tensor in state_dict["shadow_params"]
        ]

    def to(self, device):
        self.shadow_params = [tensor.to(device) for tensor in self.shadow_params]


# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

def smooth_lddt_loss(pred_coords, true_coords, coords_mask, t):
        """
        Differentiable Soft LDDT Loss (adapted from SimpleFold).
        Returns (B,) loss tensor.
        """
        
        lddt_cutoff = 15.0
        
        B, N, _ = true_coords.shape
        device = pred_coords.device

        true_dists = torch.cdist(true_coords, true_coords)
        
        mask = (true_dists < lddt_cutoff).float()
        
        mask = mask * (1 - torch.eye(N, device=device).unsqueeze(0))
        
        mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

        pred_dists = torch.cdist(pred_coords, pred_coords)
        dist_diff = torch.abs(true_dists - pred_dists)

        eps = (
            torch.sigmoid(0.5 - dist_diff)
            + torch.sigmoid(1.0 - dist_diff)
            + torch.sigmoid(2.0 - dist_diff)
            + torch.sigmoid(4.0 - dist_diff)
        ) / 4.0

        num = (eps * mask).sum(dim=(-1, -2))
        den = mask.sum(dim=(-1, -2)).clamp(min=1) 
        
        lddt_score = num / den # (B,)

        loss = 1.0 - lddt_score

        if isinstance(t, float):
             t_val = torch.tensor(t, device=device)
        else:
             t_val = t
             
        if t_val.dim() > 0 and t_val.shape[0] != B:
             t_val = t_val.view(-1)

        t_weight = 1.0 + 8.0 * torch.relu(t_val - 0.5)
        loss = loss * t_weight

        valid_molecule = (coords_mask.sum(dim=1) >= 2).float()
        loss = loss * valid_molecule
        
        return loss

def lattice_params_to_matrix_torch(lengths, angles):
    """
    lengths: (B, 3) a, b, c
    angles: (B, 3) alpha, beta, gamma (in degrees)
    Returns: (B, 3, 3) Cell matrix (rows are lattice vectors)
    """
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
    
    zeros = torch.zeros_like(a)
    
    # Deg to Rad
    alpha_rad = alpha * torch.pi / 180.0
    beta_rad = beta * torch.pi / 180.0
    gamma_rad = gamma * torch.pi / 180.0
    
    val = (torch.cos(alpha_rad) * torch.cos(beta_rad) - torch.cos(gamma_rad)) / (torch.sin(alpha_rad) * torch.sin(beta_rad))
    # Clamp val to avoid nan in sqrt
    val = torch.clamp(val, -1.0, 1.0)
    
    gamma_star = torch.acos(val)
    
    vector_a = torch.stack([a * torch.sin(beta_rad), zeros, a * torch.cos(beta_rad)], dim=1)
    vector_b = torch.stack([-b * torch.sin(alpha_rad) * torch.cos(gamma_star), b * torch.sin(alpha_rad) * torch.sin(gamma_star), b * torch.cos(alpha_rad)], dim=1)
    vector_c = torch.stack([zeros, zeros, c], dim=1)
    
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=24, cutoff=4.0, start=0.0):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(start, cutoff, n_rbf), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, dists, mask):
        # dists: (B, M, M)
        # mask: (B, M)
        
        dists_expanded = dists.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        
        # (B, M, M, n_rbf)
        rbf = torch.exp(-self.gamma * (dists_expanded - centers)**2)
        
        # Masking
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) 
        
        # Self-interaction exclusion
        B, M = mask.shape
        identity = torch.eye(M, device=mask.device).unsqueeze(0).expand(B, -1, -1)
        pair_mask = pair_mask * (1 - identity)
        
        rbf = rbf * pair_mask.unsqueeze(-1)
        
        # Mask-Normalized Mean Aggregation
        rbf_sum = rbf.sum(dim=2)
        neighbor_counts = pair_mask.sum(dim=2, keepdim=True)
        rbf_mean = rbf_sum / (neighbor_counts + 1e-8)
        
        return rbf_mean