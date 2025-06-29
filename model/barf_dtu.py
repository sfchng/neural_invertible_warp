import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import nerf_dtu
import camera
from align_trajectories import (align_translations, align_ate_c2b_use_a2b, backtrack_from_aligning_the_trajectory)
from utils.torch import get_log_string
import model.nvp.encoder
import model.nvp.nvp_ndr
import roma
import importlib

from model.pose_models.se3 import SE3PoseParams
from model.pose_models.inn import INNPoseParams
from utils.colmap_initialization.sfm import compute_sfm_pdcnet
from utils.camera import pad_poses
# ============================ main engine for training and evaluation ============================

class Pose:
    
    def set_initial_poses(self, opt):
        """ Defines initial pose to be optimized 
        Args: 
            opt (dict)
        Return:
            initial_poses_w2c (torch.Tensor [B,3,4])
            valid_poses_idx (list)
        """
        
        gt_poses_w2c = self.train_data.get_all_camera_poses(opt).to(opt.device) #(B,3,4)
        #self.save_projection_matrix_for_dtu(opt, gt_poses_w2c, mode="gt")
        n_poses = len(self.train_data)
        valid_poses_idx = np.arange(start=0, stop=n_poses, step=1).tolist()
        
        if opt.pose.init == "identity":
            initial_poses_w2c = torch.eye(4, 4)[None, ...].repeat(n_poses, 1, 1).to(opt.device)  #(B,4,4)
            initial_poses_w2c, trans_scaling = align_translations(gt_poses_w2c, initial_poses_w2c) #(B,4,4)
        elif opt.pose.init == "noisy_gt":
            ## similar to blender, corrupt same noise level in rotation and translation ##
            se3_noise = torch.randn(n_poses, 6, device=opt.device) * opt.pose.noise
            pose_noise = camera.lie.se3_to_SE3(se3_noise)
            initial_poses_w2c = camera.pose.compose([pose_noise, gt_poses_w2c]) 
        elif opt.pose.init == "given":
            initial_poses_w2c = self.train_data.all.pose.to(opt.device)                       
        elif opt.pose.init == "colmap":
            log.info("Initialize using COLMAP poses with PDCNET")
            ## Following sparf, we use initial poses obtained by COLMAP with different matches ##
            ## the results from COLMAP will be saved in a common directory, so that the matching just has to be done once ##
            directory_colmap = os.path.join(self.train_data.root, "common/colmap", opt.data.scene)
            os.makedirs(directory_colmap, exist_ok=True)
            
            initial_poses_w2c, valid_poses_idx, index_images_excluded = compute_sfm_pdcnet(opt, self.train_data.all, save_dir=directory_colmap)
            initial_poses_w2c = initial_poses_w2c.to(opt.device).float()
            initial_poses_w2c, ssim_est_gt_c2w = self.prealign_w2c_small_camera_systems\
                (opt, initial_poses_w2c[:, :3], gt_poses_w2c[:, :3])
            trans_scaling = ssim_est_gt_c2w.s      
            log.info("Exclude {} images".format(index_images_excluded))      
        
        else:
            raise ValueError
        return initial_poses_w2c[:,:3], valid_poses_idx
    
    
    def save_projection_matrix_for_dtu(self, opt, poses_w2c, mode="init"):
        
        ## rescale the poses ##
        poses_c2w = camera.pose.invert(poses_w2c).detach().cpu().numpy()

        #poses_c2w = poses_c2w.detach().cpu().numpy()
        poses_c2w[:, :3, 3:] = poses_c2w[:, :3, 3:] / self.train_data.scaling_factor
        poses_c2w[:, :3, 3:] += self.train_data.norm_trans[None] 
        
        K = self.train_data.intrinsics[:,:3,:3]
        num_cameras = poses_w2c.shape[0]        
        
        poses_w2c_ = camera.pose.invert(torch.from_numpy(poses_c2w))
        projection_matrix = K @ poses_w2c_.detach().cpu().numpy()

        cameras_new = {}
        for i in range(num_cameras):
            cameras_new['world_mat_%d' % i ] = np.concatenate((projection_matrix[i], np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)
        np.savez('{0}/{1}_{2}.npz'.format(opt.output_path, "cameras", mode), **cameras_new)
        
    def save_subset_projection_matrix_for_dtu(self, opt, poses_w2c, mode="init"):
        
        ## convert w2c -> c2w 
        poses_c2w = camera.pose.invert(poses_w2c).detach().cpu().numpy()    
        all_poses = self.train_data.all_poses_c2w.shape[0]
        all_indices = np.arange(all_poses)
        test_indices = all_indices[all_indices % opt.data.dtu.dtuhold == 0]
        test_poses = self.train_data.all_poses_c2w[test_indices]
        new_arrs_for_dtu = self.efficient_merge_two_arrays(opt, poses_c2w, test_poses, test_indices)
        
        new_arrs_for_dtu[:, :3, 3:] = new_arrs_for_dtu[:, :3, 3:] / self.train_data.scaling_factor
        new_arrs_for_dtu[:, :3, 3:] += self.train_data.norm_trans[None] 
        
        K = self.train_data.intrinsics[:,:3,:3]
        
        new_arrs_for_dtu_w2c = camera.pose.invert(torch.from_numpy(new_arrs_for_dtu))
        projection_matrix = K @ new_arrs_for_dtu_w2c.detach().cpu().numpy()

        cameras_new = {}
        for i in range(all_poses):
            cameras_new['world_mat_%d' % i ] = np.concatenate((projection_matrix[i], np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)
        np.savez('{0}/{1}_{2}.npz'.format(opt.output_path, "cameras", mode), **cameras_new)
        
        print ('here')
        
    def efficient_merge_two_arrays(self, opt, base_array, insert_array, positions):

        new_arrs = []
        j = 0
        k = 0
        total_size = base_array.shape[0] + len(positions)
        for i in range(total_size):
            if i % opt.data.dtu.dtuhold == 0:
                new_arrs.append(insert_array[j][:3])
                j+=1
            else:
                new_arrs.append(base_array[k])
                k+=1
        
        return np.stack(new_arrs)            
    
    @torch.no_grad()
    def evaluate_poses(self, opt):
        pose, pose_GT = self.get_all_training_poses(opt)
        return self.evaluate_any_poses(opt, pose, pose_GT)
    
    @torch.no_grad()
    def evaluate_any_poses(self, opt, pose_w2c, pose_GT_w2c):
        """Evaluates rotation and translation errors before and after alignment. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor [B,3,4])
            pose_GT_w2c (torch.Tensor [B,3,4])
        """
        stats_dict = {}
        error = self.evaluate_camera_alignment(opt,pose_w2c.detach(),pose_GT_w2c)
        stats_dict['error_R_before_align'] = error.R.mean() * 180. / np.pi  # radtodeg
        stats_dict['error_t_before_align'] = error.t.mean()

        if pose_w2c.shape[0] > 10:
            pose_aligned,_ = self.prealign_w2c_large_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        else:
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT_w2c)
        stats_dict['error_R'] = error.R.mean() * 180. / np.pi  # radtodeg
        stats_dict['error_t'] = error.t.mean()
        return stats_dict
    
    @torch.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned_w2c,
                                  pose_GT_w2c):
        """
        Measures rotation and translation error between aligned and ground-truth world-to-camera poses. 
        Warning:::we want the translation difference between the camera centers in the world 
        coordinate! (not the opposite!)
        Args:
            opt (edict): settings
            pose_aligned_w2c (torch.Tensor [B,3,4])
            pose_GT_w2c (torch.Tensor [B,3,4])
        Returns:
            error: edict with keys 'R' and 't', for rotation (in radian) and translation erorrs (not averaged)
        """
        # just invert both poses from camera to world
        # so that the translation corresponds to the position of the camera in world coordinate frame. 
        pose_aligned_c2w = camera.pose.invert(pose_aligned_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)

        R_aligned_c2w,t_aligned_c2w = pose_aligned_c2w.split([3,1],dim=-1)
        # R_aligned is (B, 3, 3)
        t_aligned_c2w = t_aligned_c2w.reshape(-1, 3)  # (B, 3)

        R_GT_c2w,t_GT_c2w = pose_GT_c2w.split([3,1],dim=-1)
        t_GT_c2w = t_GT_c2w.reshape(-1, 3)

        R_error = camera.rotation_distance(R_aligned_c2w,R_GT_c2w)
        
        t_error = (t_aligned_c2w - t_GT_c2w).norm(dim=-1)

        error = edict(R=R_error,t=t_error)  # not meaned here
        return error
    
    @torch.no_grad()
    def prealign_w2c_large_camera_systems(self,opt,pose_w2c,
                                          pose_GT_w2c):
        """Compute the 3D similarity transform relating pose_w2c to pose_GT_w2c. Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor [B,3,4])
            pose_GT_w2c (torch.Tensor [B,3,4])
        """
        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)

        if opt.pose.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=opt.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=opt.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            try:
                pose_aligned_c2w, ssim_est_gt_c2w = align_ate_c2b_use_a2b(pose_c2w, pose_GT_c2w, method='sim3')
                pose_aligned_w2c = camera.pose.invert(pose_aligned_c2w[:, :3])
                ssim_est_gt_c2w.type = 'traj_align'
            except:
                log.warning("warning: SVD did not converge...")
                pose_aligned_w2c = pose_w2c
                ssim_est_gt_c2w = edict(R=torch.eye(3,device=opt.device).unsqueeze(0), type='traj_align', 
                                        t=torch.zeros(1,3,1,device=opt.device), s=1.)
        return pose_aligned_w2c, ssim_est_gt_c2w

    @torch.no_grad()
    def prealign_w2c_small_camera_systems(self,opt,pose_w2c,pose_GT_w2c):
        """Compute the transformation from pose_w2c to pose_GT_w2c by aligning the each pair of pose_w2c 
        to the corresponding pair of pose_GT_w2c and computing the scaling. This is more robust than the
        technique above for small number of input views/poses (<10). Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        def alignment_function(poses_c2w_from_padded: torch.Tensor, 
                               poses_c2w_to_padded: torch.Tensor, idx_a: int, idx_b: int):
            """Args: FInd alignment function between two poses at indixes ix_a and idx_n

                poses_c2w_from_padded: Shape is (B, 4, 4)
                poses_c2w_to_padded: Shape is (B, 4, 4)
                idx_a:
                idx_b:

            Returns:
            """
            # We take a copy to keep the original poses unchanged.
            poses_c2w_from_padded = poses_c2w_from_padded.clone()
            # We use the distance between the same two poses in both set to obtain
            # scale misalgnment.
            dist_from = torch.norm(
                poses_c2w_from_padded[idx_a, :3, 3] - poses_c2w_from_padded[idx_b, :3, 3]
            )
            dist_to = torch.norm(
                poses_c2w_to_padded[idx_a, :3, 3] - poses_c2w_to_padded[idx_b, :3, 3])
            scale = dist_to / dist_from

            # alternative for scale
            # dist_from = poses_w2c_from_padded[idx_a, :3, 3] @ poses_c2w_from_padded[idx_b, :3, 3]
            # dist_to = poses_w2c_to_padded[idx_a, :3, 3] @ poses_c2w_to_padded[idx_b, :3, 3]
            # scale = onp.abs(dist_to /dist_from).mean()

            # We bring the first set of poses in the same scale as the second set.
            poses_c2w_from_padded[:, :3, 3] = poses_c2w_from_padded[:, :3, 3] * scale

            # Now we simply apply the transformation that aligns the first pose of the
            # first set with first pose of the second set.
            transformation_from_to = poses_c2w_to_padded[idx_a] @ camera.pose_inverse_4x4(
                poses_c2w_from_padded[idx_a])
            poses_aligned_c2w = transformation_from_to[None] @ poses_c2w_from_padded

            poses_aligned_w2c = camera.pose_inverse_4x4(poses_aligned_c2w)
            ssim_est_gt_c2w = edict(R=transformation_from_to[:3, :3].unsqueeze(0), type='traj_align', 
                                    t=transformation_from_to[:3, 3].reshape(1, 3, 1), s=scale)

            return poses_aligned_w2c[:, :3], ssim_est_gt_c2w

        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
        B = pose_c2w.shape[0]

        if opt.pose.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=self.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            # try every combination of pairs and get the rotation/translation
            # take the one with the smallest error
            # this is because for small number of views, the procrustes alignement with SVD is not robust. 
            pose_aligned_w2c_list = []
            ssim_est_gt_c2w_list = []
            error_R_list = []
            error_t_list = []
            full_error = []
            for pair_id_0 in range(min(B, 10)):  # to avoid that it is too long
                for pair_id_1 in range(min(B, 10)):
                    if pair_id_0 == pair_id_1:
                        continue
                    
                    pose_aligned_w2c_, ssim_est_gt_c2w_ = alignment_function\
                        (pad_poses(pose_c2w), pad_poses(pose_GT_c2w),
                         pair_id_0, pair_id_1)
                    pose_aligned_w2c_list.append(pose_aligned_w2c_)
                    ssim_est_gt_c2w_list.append(ssim_est_gt_c2w_ )

                    error = self.evaluate_camera_alignment(opt, pose_aligned_w2c_, pose_GT_w2c)
                    error_R_list.append(error.R.mean().item() * 180. / np.pi )
                    error_t_list.append(error.t.mean().item())
                    full_error.append(error.t.mean().item() * (error.R.mean().item() * 180. / np.pi))

            ind_best = np.argmin(full_error)
            # print(np.argmin(error_R_list), np.argmin(error_t_list), ind_best)
            pose_aligned_w2c = pose_aligned_w2c_list[ind_best]
            ssim_est_gt_c2w = ssim_est_gt_c2w_list[ind_best]

        return pose_aligned_w2c, ssim_est_gt_c2w


class Model(nerf_dtu.Model, Pose):

    def __init__(self,opt):
        super().__init__(opt)

    def build_pose_net(self, opt):
        """
        Defines initial pose and pose parameterization here
        """
        
        ## get initial pose ##
        initial_poses_w2c, valid_poses_idx = self.set_initial_poses(opt)
        #self.save_subset_projection_matrix_for_dtu(opt, initial_poses_w2c, mode="init")
        
        ## save initial poses ##
        if opt.save.init_poses:
            log.info("saving initial poses...")
            self.save_subset_projection_matrix_for_dtu(opt, initial_poses_w2c, mode="gt")
        #initial_poses_c2w = self.preprocess_poses_c2w(opt, initial_poses_w2c)
            
            
        ## get groundtruth pose ##
        gt_poses_w2c = self.train_data.get_all_camera_poses(opt).to(opt.device)
        
        stats_dict = self.evaluate_any_poses(opt, initial_poses_w2c, gt_poses_w2c)
        ## log initial pose errors ##
        message = get_log_string(stats_dict)
        log.warning("All initial pose: {}".format(message))

        if opt.pose.parameterization == "se3":
            self.pose_net = SE3PoseParams(opt, num_poses=len(self.train_data), 
                                          initial_poses_w2c=initial_poses_w2c, device=opt.device)
            ## used in barf ##
        elif opt.pose.parameterization == "inn":
            self.pose_net = INNPoseParams()
            raise NotImplementedError
        else: 
            raise ValueError    

    def build_networks(self,opt):
        
        ## build pose net ##
        self.build_pose_net(opt)

        graph = importlib.import_module("model.{}".format(opt.model))
        log.info("building networks for DTU training...")
        self.graph = graph.Graph(opt, self.pose_net).to(opt.device)
        
    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim_pose = optimizer([dict(params=self.pose_net.parameters(),lr=opt.optim.lr_pose)])
        
        ## setup scheduler ##
        if opt.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            self.sched_pose = scheduler(self.optim_pose,**kwargs)

    def train_iteration(self,opt,var,loader):
        self.optim_pose.zero_grad()
        if opt.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1,self.it/opt.optim.warmup_pose)
        loss = super().train_iteration(opt,var,loader)
        self.optim_pose.step()
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr"] = self.optim_pose.param_groups[0]["lr_orig"] # reset learning rate
        if opt.optim.sched_pose: self.sched_pose.step()
        self.graph.nerf.progress.data.fill_(self.it/opt.max_iter)
        if opt.nerf.fine_sampling:
            self.graph.nerf_fine.progress.data.fill_(self.it/opt.max_iter)
        return loss

    @torch.no_grad()
    def validate(self,opt,ep=None):
        pose, pose_GT = self.get_all_training_poses(opt)
        
        ## computes the alignment between optimized poses and gt ones.
        ## will be used to transform the poses to align with the coordinate system
        ## of optimized poses for the evaluation
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.pose_net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(opt,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.pose_net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(opt,pose,pose_GT)
        super().validate(opt,ep=ep)


    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)

            ## compute pose accuracy 
            stats_dict = self.evaluate_poses(opt)
            self.tb.add_scalar("{0}/error_Rdeg".format(split),stats_dict['error_R'],step)
            self.tb.add_scalar("{0}/error_t".format(split),stats_dict['error_t'],step)            

        # Evaluate pose accuracy
        ## v0a: evaluate pose accuracy using the global transformation ##
        # if split=="train" and opt.data.dataset in ["blender","llff", "llff_modified"]:
        #     pose,pose_GT = self.get_all_training_poses(opt)
        #     pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
        #     error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        #     self.tb.add_scalar("{0}/error_R".format(split),error.R.mean(),step)
        #     self.tb.add_scalar("{0}/error_t".format(split),error.t.mean(),step)
            

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step=step,split=split)
        if opt.visdom:
            if split=="val":
                pose,pose_GT = self.get_all_training_poses(opt)
                util_vis.vis_cameras(opt,self.vis,step=step,poses=[pose,pose_GT])

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(opt.device)
        pose_pred_w2c = self.pose_net.get_w2c_poses()

        return pose_pred_w2c,pose_GT_w2c

    @torch.no_grad()
    def prealign_cameras(self,opt,pose,pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1,1,3,device=opt.device)
        center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
        center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
        try:
            sim3 = camera.procrustes_analysis(center_GT,center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device))
        # align the camera poses
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned,pose_GT):
        # measure errors in rotation and translation
        R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
        R_GT,t_GT = pose_GT.split([3,1],dim=-1)
        R_error = camera.rotation_distance(R_aligned,R_GT)
        t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
        error = edict(R=R_error,t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self,opt):
        self.graph.eval()
        
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt) #(B,3,4),(B,3,4)
        if pose.shape[0] > 9:
            # alignment of the trajectory
            pose_aligned, self.graph.pose_net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(opt,pose,pose_GT)
        else:
            # alignment of the first cameras
            pose_aligned, self.graph.pose_net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(opt,pose,pose_GT)
    
        ## save initialization ##
        self.save_subset_projection_matrix_for_dtu(opt, pose_aligned, mode="pred_barf_se3")
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        print("trans: {:10.5f}".format(error.t.mean()))
        print("--------------------------")
        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,(err_R,err_t) in enumerate(zip(error.R,error.t)):
                file.write("{} {} {}\n".format(i,err_R.item(),err_t.item()))
        # evaluate novel view synthesis
        super().evaluate_full(opt)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self,opt,var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=opt.device))
        optimizer = getattr(torch.optim,opt.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test],lr=opt.optim.lr_pose)])
        #iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in range(opt.optim.test_iter):
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim_pose.step()
            #iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(16,8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []
        for ep in range(0,opt.max_iter+1,opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep!=0:
                try: util.restore_checkpoint(opt,self,resume=ep)
                except: continue
            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            pose_aligned, _ = self.prealign_w2c_large_camera_systems(opt, pose.detach(), pose_ref)
            pose_aligned = pose_aligned.detach().cpu()
            pose_ref = pose_ref.detach().cpu()
            if opt.data.dataset in ["blender", "dtu", "llff", "llff_modified"]:
                dict(
                    dtu=util_vis.plot_save_poses_blender,
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=ep)
            else:
                pose = pose.detach().cpu()
                util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=ep)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 30 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)

# ============================ computation graph for forward/backprop ============================

class Graph(nerf_dtu.Graph):

    def __init__(self,opt, pose_net):
        super().__init__(opt)
        
        self.pose_net = pose_net
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
        self.pose_eye = torch.eye(3,4).to(opt.device)
        
    def get_pose(self,opt,var,mode=None):
        return self.get_w2c_pose(opt, var, mode)
    
    def get_w2c_pose(self,opt,var,mode=None):
        if mode=="train":
            ## get the current estimates of the camera poses, which are optimized  ##
            pose = self.pose_net.get_w2c_poses()  
        elif mode in ["val","eval","test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end 
            # align test pose to refined coordinate system (up to sim3)
            assert hasattr(self.pose_net, 'sim3_est_to_gt_c2w')
            pose_GT_w2c = var.pose
            ssim_est_gt_c2w = self.pose_net.sim3_est_to_gt_c2w
            if ssim_est_gt_c2w.type == 'align_to_first':
                raise NotImplementedError
                pose = backtrack_from_aligning_and_scaling_to_first_cam(pose_GT_w2c, ssim_est_gt_c2w)
            elif ssim_est_gt_c2w.type == 'traj_align':
                pose = backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w)
            else:
                raise ValueError
            # Here, we align the test pose to the poses found during the optimization (otherwise wont be valid)
            # that's pose. And can learn an extra alignement on top
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test,pose])
        else: 
            raise ValueError
        return pose
        
    def get_pose_orig(self,opt,var,mode=None):
        if mode=="train":
            # add the pre-generated pose perturbations
            if opt.data.dataset=="blender":
                if opt.camera.noise:
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose_noise,var.pose])
                else: pose = var.pose
            else: pose = self.pose_eye
            # add learnable pose correction
            var.se3_refine = self.se3_refine.weight[var.idx]
            pose_refine = camera.lie.se3_to_SE3(var.se3_refine)
            pose = camera.pose.compose([pose_refine,pose])
        elif mode in ["val","eval","test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = torch.zeros(1,1,3,device=opt.device)
            center = camera.cam2world(center,var.pose)[:,0] # [N,3]
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
            R_aligned = var.pose[...,:3]@self.sim3.R
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
            pose = camera.pose(R=R_aligned,t=t_aligned)
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test,pose])
        else: pose = var.pose
        return pose
       
    def get_c2w_pose(self, opt, var, mode=None):
        w2c = self.get_w2c_pose(opt, var, mode)
        return camera.pose.invert(w2c)

class NeRF(nerf_dtu.NeRF):

    def __init__(self,opt):
        super().__init__(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def positional_encoding(self,opt,input,L): # [B,...,N]
        input_enc = super().positional_encoding(opt,input,L=L) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=opt.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
            if opt.model == "barf_inn_nvp" and opt.barf_c2f is not None:
                alpha_ratio = weight.sum() / L
                return input_enc, alpha_ratio
        return input_enc
