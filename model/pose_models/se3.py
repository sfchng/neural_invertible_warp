import torch
import torch.nn as nn
import torch.nn.functional as F
import camera


class SE3PoseParams(torch.nn.Module):
    def __init__(self, opt, num_poses, initial_poses_w2c, device="cuda"):
        super().__init__()
        
        self.opt = opt
        self.num_poses = num_poses
        self.device = opt.device
        self.initial_poses_w2c = initial_poses_w2c
        
        ## initialize all the optimziable pose parameters ##
        self.init_poses_embed()
        
    def init_poses_embed(self):
        
        if self.opt.pose.optimize_relative_poses:
            NotImplementedError
        else:
            self.pose_embedding = torch.nn.Parameter(torch.zeros(self.num_poses, 6, device=self.opt.device))
   
            

    def get_w2c_poses(self):
        ## add learnable pose correction ##
        pose_refine = camera.lie.se3_to_SE3(self.pose_embedding)
        
        if self.opt.pose.optimize_relative_poses:
            raise NotImplementedError
        else:
            pose = camera.pose.compose([pose_refine,self.initial_poses_w2c])
        return pose