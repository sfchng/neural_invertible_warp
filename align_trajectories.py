import torch
import camera
from scipy.spatial.transform import Rotation as RotLib
import numpy as np
from easydict import EasyDict as edict

from camera import pose_inverse_4x4
from third_party.ATE.align_utils import alignTrajectory


def SO3_to_quat(R: np.ndarray):
    """
    Args:
        R:  (N, 3, 3) or (3, 3) np
    Returns:   
        (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat: np.ndarray):
    """
    Args:
        quat:    (N, 4, ) or (4, ) np
    Returns:  
        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R

def convert3x4_4x4(input: torch.Tensor) -> torch.Tensor:
    """
    Args:
        input:  (N, 3, 4) or (3, 4) torch or np
    Returns: 
        (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output

def backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w):
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    R_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) @ pose_GT_c2w[:, :3, :3]
    t_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) / ssim_est_gt_c2w.s @ (pose_GT_c2w[:, :3, 3:4] - ssim_est_gt_c2w.t)
    pose_GT_c2w_aligned = camera.pose(R=R_GT_c2w_aligned,t=t_GT_c2w_aligned.reshape(-1, 3))
    pose_w2d_recovered = camera.pose.invert(pose_GT_c2w_aligned)
    return pose_w2d_recovered


def align_translations(GT_poses_w2c: torch.Tensor, initial_poses_w2c: torch.Tensor):
    """
    Args:
        GT_poses_w2c (torch.Tensor [B,3,4])
        initial_poses (torch.Tensor [B,4,4])
    """
    GT_poses_w2c_ = torch.eye(4).unsqueeze(0).repeat(GT_poses_w2c.shape[0], 1, 1).to(GT_poses_w2c.device)
    GT_poses_w2c_[:, :3] = GT_poses_w2c
    pose_GT_c2w = camera.pose_inverse_4x4(GT_poses_w2c_) #(B,4,4)

    initial_poses_c2w = camera.pose_inverse_4x4(initial_poses_w2c)  # (B,4,4)

    # initial_poses_c2w = pose_inverse_4x4(initial_poses_w2c)
    # position of camera in world coordinate basically
    trans_error = pose_GT_c2w[:, :3, -1].mean(0).reshape(1, -1) - \
        initial_poses_c2w[:, :3, -1].mean(0).reshape(1, -1)
    
    initial_poses_c2w[:, :3, -1] += trans_error

    translation_scaling = 1.
    initial_poses_w2c = camera.pose_inverse_4x4(initial_poses_c2w)
    return initial_poses_w2c, translation_scaling


def align_ate_c2b_use_a2b(traj_a_c2w: torch.Tensor, traj_b_c2w: torch.Tensor, 
                          traj_c: torch.Tensor=None, method='sim3', 
                          pose_id_to_align=0):
    """Align c to b using the sim3 from a to b.
    Args:
        traj_a:  (torch.Tensor [N0,3/4,4])
        traj_b:  (torch.Tensor [N0,3/4,4])
        traj_c:  None or (N1, 3/4, 4) torch tensor
    Returns:
        (N1, 4, 4) torch tensor
    """
    device = traj_a_c2w.device
    if traj_c is None:
        traj_c = traj_a_c2w.clone()

    traj_a = traj_a_c2w.float().cpu().numpy()
    traj_b = traj_b_c2w.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0,3,3)
    t_a = traj_a[:, :3, 3]  # (N0,3)
    quat_a = SO3_to_quat(R_a)  # (N0,4)

    R_b = traj_b[:, :3, :3]  # (N0,3,3)
    t_b = traj_b[:, :3, 3]  # (N0,3)
    quat_b = SO3_to_quat(R_b)  # (N0,4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method, pose_id_to_align=pose_id_to_align) 

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1,3,3)
    t = t[None, :, None].astype(np.float32)  # (1,3,1)
    s = float(s)


    R_c = traj_c[:, :3, :3]  # (N1,3,3)
    t_c = traj_c[:, :3, 3:4]  # (N1,3,1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1,3,4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1,4,4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    ssim_est_gt_c2w = edict(R=torch.from_numpy(R).to(device), t=torch.from_numpy(t).to(device), s=s)
    return traj_c_aligned, ssim_est_gt_c2w  # (N1, 4, 4)
