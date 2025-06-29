import torch
import torch.nn as nn
import torch.nn.functional as F
import camera
import model.nvp.encoder
import model.nvp.nvp_ndr
import roma 

class INNPoseParams(torch.nn.Module):
    def __init__(self,opt,num_poses, initial_poses_w2c, device="cuda"):
        super().__init__()
        
        self.opt = opt
        self.num_poses = num_poses
        self.device = opt.device
        self.initial_poses_w2c = initial_poses_w2c
        self.init_poses_embed()
    
    def init_poses_embed(self):

        self.pose_latent = torch.nn.Embedding(self.num_poses, self.opt.inn.real_nvp.latent_dim).to(self.opt.device)    
        ## get embedding ##
        self.pose_embedding = model.nvp.nvp_ndr.DeformNetwork(d_feature=self.opt.inn.real_nvp.latent_dim, 
                                                              d_in=3, d_out_1=1, d_out_2=3, n_blocks=3, 
                                                              d_hidden=self.opt.inn.real_nvp.d_hidden, n_layers=1,
                                                              skip_in=[], multires=self.opt.inn.real_nvp.multires, 
                                                              weight_norm=True, actfn=self.opt.inn.actfn).to(self.opt.device)
        

        self.pose_global = torch.nn.Embedding(self.num_poses,12).to(self.opt.device)
        
        
    def get_w2c_poses(self):
        """
        TODO: split them into two different functions,
        one is w2c poses, a global transformation
        and another one is center and ray 
        
        get_pose is used in two different spots,
        one is for rendering,
        another one is for pose accuracy evaluation
        """
        return self.pose_global.weight.data.detach().clone().view(-1,3,4)

        
        # if mode == "train":
        #     self.center_init, self.grid_init = camera.get_unwarped_center_and_ray(self.opt, intr=var.intr, ray_idx=var.ray_idx, pose_init=self.initial_poses_w2c) 
        #     center_init, grid_init = self.center_init.detach(), self.grid_init.detach()
                
        #     output_coords = self.forward_inn(center_init, grid_init, iter)
        #     grid_3D_pred = output_coords[:,:len(var.ray_idx),:].squeeze(2)  #(B,N,3)
        #     center_3D_pred = output_coords[:,len(var.ray_idx):,:].squeeze(2)  #(B,N,3)        
        #     ray_pred = grid_3D_pred - center_3D_pred
            
        #     ## solve a rigid transformation ##
        #     self.solve_for_global_transformation(grid_3D_pred, center_3D_pred)
            
        #     return ray_pred, center_3D_pred, grid_3D_pred
        
        # else:
        #     raise NotImplementedError
        
    def get_warped_rays_in_world(self, var, mode=None, iter=None):
        assert mode == "train"
        
        self.center_init, self.grid_init = camera.get_unwarped_center_and_ray(self.opt, intr=var.intr, ray_idx=var.ray_idx, pose_init=self.initial_poses_w2c) 
        center_init, grid_init = self.center_init.detach(), self.grid_init.detach()
            
        output_coords = self.forward_inn(center_init, grid_init, iter)
        grid_3D_pred = output_coords[:,:len(var.ray_idx),:].squeeze(2)  #(B,N,3)
        center_3D_pred = output_coords[:,len(var.ray_idx):,:].squeeze(2)  #(B,N,3)        
        ray_pred = grid_3D_pred - center_3D_pred
        
        ## solve a rigid transformation ##
        self.solve_for_global_transformation(grid_3D_pred, center_3D_pred)
        
        return ray_pred, center_3D_pred, grid_3D_pred
  
        
        
    def forward_inn(self, centers, grids, iter):

        ## call embedding ##
        feat = self.pose_latent.weight #(B,D)
        
        if self.opt.inn.real_nvp.c2f == True:
            alpha_ratio = max(min(iter / self.opt.inn.real_nvp.max_pe_iter,1),0)
        else:
            alpha_ratio = 1
            
        input_coords = torch.cat([grids, centers], axis=1).unsqueeze(2)
        output_coords = self.pose_embedding.forward(feat, input_coords, alpha_ratio=alpha_ratio)
        return output_coords
    
    
    def solve_for_global_transformation(self, grid_pred, center_pred):
        
        source = torch.cat([self.grid_init, self.center_init], dim=1)  #(B,2N,3)
        target = torch.cat([grid_pred, center_pred], dim=1)
        R_global, t_global = roma.rigid_points_registration(target, source) #(B,3,3),(B,3)
        svd_poses = torch.cat((R_global,t_global[...,None]),-1) #(B,3,4)
        self.pose_global.weight.data = svd_poses.detach().clone().view(-1,12)   
        