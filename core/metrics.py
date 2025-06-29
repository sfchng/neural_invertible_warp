import torch
from easydict import EasyDict as edict

def compute_depth_error_on_rays(var, scaling_factor_for_pred_depth: float=1.):
    """
    Computes depth error between rendered depth at rays and gt depth. 
    Args:
        var (edict): Dic containing all the training parameters
            ::idx (torch.Tensor [B]): the indices of the image
            ::rgb_path (str)
            ::depth_gt (torch.Tensor [B,H,W])
            ::fg_mask
            ::valid_depth_gt (torch.Tensor [B,H,W])
            ::image (torch.Tensor [B,C,H,W])
            ::intr (torch.Tensor [B,3,3])
            ::pose (torch.Tensor [B,3,4])
            ::depth_range (torch.Tensor [B,2])
            ::scene (list)
            ----------------------------------------------------------------
            ::ray_idx (torch.Tensor [B])
            ::rgb (torch.Tensor [B,N,3]) : rendered rgb at rays
            ::depth (torch.Tensor [B,N,1]): rendered depth at rays
            ::opacity

    """
    idx_img_rendered = var.idx
    B = len(idx_img_rendered)

    depth_gt = var.depth_gt[idx_img_rendered].view(B, -1, 1) #(B,HW, 1)
    valid_depth = var.valid_depth_gt[idx_img_rendered].view(B, -1, 1) #(B,HW, 1)
    pred_depth = var.depth
    if 'ray_idx' in var.keys():
        ray_idx = var.ray_idx
        if len(ray_idx.shape) == 2 and ray_idx.shape[0] == B:
            # different ray idx for each image in batch
            n_rays = ray_idx.shape[-1]
            #ray_idx is B, N-rays, ie different ray indices for each image in batch
            batch_idx_flattened = torch.arange(start=0, end=B)\
                .unsqueeze(-1).repeat(1, n_rays).long().view(-1)
            ray_idx_flattened = ray_idx.reshape(-1).long()
            depth_gt = depth_gt[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 1)
            depth_gt = depth_gt.reshape(B, n_rays, -1)  # (nbr_rendered_images, n_rays, 1)

            valid_depth = valid_depth[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 1)
            valid_depth = valid_depth.reshape(B, n_rays, -1)  # (nbr_rendered_images, n_rays, 1)
        else:
            # when random ray, only rendered some rays
            depth_gt = depth_gt[:, ray_idx]  #(B,N,1)
            valid_depth = valid_depth[:, ray_idx] #(B,N,1)
    
    depth_gt = depth_gt[valid_depth]
    pred_depth = pred_depth[valid_depth] * scaling_factor_for_pred_depth

    abs_e = torch.abs(depth_gt - pred_depth)
    abs_e = abs_e.sum() / (abs_e.nelement() + 1e-6) # same than rmse

    rmse = compute_rmse(depth_gt, pred_depth)
    return abs_e, rmse


def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())

def compute_depth_error(var, scaling_factor_for_pred_depth: float=1.):
    """
    Computes depth error between rendered depth and gt depth, for a full image. Here N is HW. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
        output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
        pred_depth: rendered depth (corresponds to depth or depth_fine keys of output_dict), (B, N, 1)
        scaling_factor_for_pred_depth (float): to scale the rendered depth to be similar to gt depth. 
    """
    def compute_metric(depth_gt, depth):
        # mse = torch.sqrt((depth_gt - depth)**2)
        abs_e = torch.abs(depth_gt - depth)
        abs_e = abs_e.sum() / (abs_e.nelement() + 1e-6)
        abs_e = abs_e.item()

        rmse = compute_rmse(depth_gt, depth).item()
        return abs_e, rmse

    idx_img_rendered = 0
    B = 1

    pred_depth = var.depth.view(B,-1,1)  #(B,N,C)
    depth_gt = var.depth_gt[idx_img_rendered].view(B,-1,1) #(B,N,C)
    valid_depth = var.valid_depth_gt[idx_img_rendered].view(B, -1) #(B,N)

    depth_gt = depth_gt[valid_depth]
    pred_depth = pred_depth[valid_depth] 
    if scaling_factor_for_pred_depth != 1.:
        abs_e_no_s, rmse_no_s = compute_metric(depth_gt, pred_depth)
        abs_e_s, rmse_s = compute_metric(depth_gt, pred_depth.clone() * scaling_factor_for_pred_depth)
        abs_e = min(abs_e_no_s, abs_e_s)
        rmse = min(rmse_no_s, rmse_s)
    else:
        abs_e, rmse = compute_metric(depth_gt, pred_depth)

    return abs_e, rmse


def compute_depth_metrics(var, scaling_factor_for_pred_depth):
    abs_err_non_scaled, rms_err_non_scaled = compute_depth_error(var, scaling_factor_for_pred_depth=1)
    abs_err_scaled, rms_err_scaled = compute_depth_error(var, scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
    abs_err = min(abs_err_non_scaled, abs_err_scaled)
    rms_err = min(rms_err_non_scaled, rms_err_scaled)
    return abs_err, rms_err
    