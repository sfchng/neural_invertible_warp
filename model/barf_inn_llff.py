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
from . import nerf_inn_llff
import camera
import model.nvp.nvp_ndr
import roma
# ============================ main engine for training and evaluation ============================

class Model(nerf_inn_llff.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def build_networks(self,opt):
        super().build_networks(opt)

        if opt.data.dataset == "blender":
            if opt.camera.noise_type =="barf":
                # pre-generate synthetic pose perturbation
                se3_noise = torch.randn(len(self.train_data),6,device=opt.device)*opt.camera.noise_barf
                self.graph.pose_noise = camera.lie.se3_to_SE3(se3_noise)
            else:
                assert opt.camera.noise_type == "l2g"
                so3_noise = torch.randn(len(self.train_data),3,device=opt.device)*opt.camera.noise_l2g_r
                t_noise = torch.randn(len(self.train_data),3,device=opt.device)*opt.camera.noise_l2g_t
                self.graph.pose_noise = torch.cat([camera.lie.so3_to_SO3(so3_noise),t_noise[...,None]],dim=-1) # [...,3,4]
            
        ## v0: build invertible neural network ##
        ## (1): optimizable latent code for the invertible neural network to differentiate one frame from another ##
        if opt.warp_latent.enc_type == "l2fbarf":
            self.graph.warp_latent = torch.nn.Embedding(len(self.train_data), opt.warp_latent.embed_dim).to(opt.device)
        elif opt.warp_latent.enc_type == "posenc":
            self.graph.frame_id = torch.linspace(1, len(self.train_data), len(self.train_data))[:,None] / len(self.train_data)
            self.graph.frame_id = self.graph.frame_id.to(opt.device)   
            opt.warp_latent.embed_dim = 2*opt.warp_latent.posenc.freq_len     
        elif opt.warp_latent.enc_type == "extrinsic":
            self.graph.warp_latent = torch.nn.Embedding(len(self.train_data), 6).to(opt.device)
            #torch.nn.init.zeros_(self.graph.warp_latent.weight)
        else:
            return NotImplementedError


        self.graph.warp_mlp = model.nvp.nvp_ndr.DeformNetwork(d_feature=opt.warp_latent.embed_dim, d_in=3, d_out_1=1, d_out_2=3, n_blocks=3, d_hidden=opt.inn.real_nvp.d_hidden, n_layers=1,
                                                              skip_in=[], multires=opt.inn.real_nvp.multires, weight_norm=True, actfn=opt.inn.actfn).to(opt.device)
 

        if opt.warp_latent.normalize:
            self.graph.frame_id = torch.linspace(1, len(self.train_data), len(self.train_data))[:,None] / len(self.train_data)
            self.graph.frame_id = self.graph.frame_id.to(opt.device)


        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise_type == "l2g":
                pose = camera.pose.compose([pose, self.graph.pose_noise])
            else:
                pose = camera.pose.compose([self.graph.pose_noise, pose])
        else: pose = self.graph.pose_eye[None].repeat(len(self.train_data),1,1)
        
        ## v0a: use Embedding so that the global rigid transformation can be checkpointed
        ## For now, let's just use Pytorch default initialization
        self.graph.global_rigid = torch.nn.Embedding(len(self.train_data),12, _weight=pose.view(-1,12)).to(opt.device)
        
        # auto near/far for blender dataset
        if opt.data.dataset=="blender":
            if not opt.camera.noise_type == "barf": 
                idx_range = torch.arange(len(self.train_data),dtype=torch.long,device=opt.device)
                idx_X,idx_Y = torch.meshgrid(idx_range,idx_range)
                self.graph.idx_grid = torch.stack([idx_X,idx_Y],dim=-1).view(-1,2)

    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(torch.optim,opt.optim.algo)
        
        if opt.inn.optimize.enabled:
            self.optim_pose = optimizer([dict(params=self.graph.warp_mlp.parameters(),lr=opt.optim.lr_pose)])  
        if opt.warp_latent.optimize.enabled and not opt.inn.optimize.enabled:    
            self.optim_pose = optimizer([dict(params=self.graph.warp_latent.parameters(),lr=opt.optim.lr_pose)])            
        elif opt.warp_latent.optimize.enabled and opt.inn.optimize.enabled:    
            self.optim_pose.add_param_group(dict(params=self.graph.warp_latent.parameters(), lr=opt.optim.lr_pose))
        

        # set up scheduler
        if opt.optim.sched_pose:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            if opt.optim.sched_pose.type == "ExponentialLR": kwargs.pop('step_size')
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
        pose,pose_GT = self.get_all_training_poses(opt)
        _,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        super().validate(opt,ep=ep)

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)


        # Evaluate pose accuracy
        ## v0a: evaluate pose accuracy using the global transformation ##
        if split=="train" and opt.data.dataset in ["blender","llff", "llff_modified"]:
            pose,pose_GT = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
            error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split),error.R.mean(),step)
            self.tb.add_scalar("{0}/error_t".format(split),error.t.mean(),step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step=step,split=split)
        if opt.visdom:
            if split=="val":
                pose,pose_GT = self.get_all_training_poses(opt)
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
                util_vis.vis_cameras(opt,self.vis,step=step,poses=[pose_aligned,pose_GT])

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        if opt.data.dataset == "blender":
            pose = pose_GT
            ## initial poses ##
            if opt.camera.noise_type == "l2g":
                pose = camera.pose.compose([pose, self.graph.pose_noise])
            else:
                pose = camera.pose.compose([self.graph.pose_noise,pose])                    
        else: pose = self.graph.pose_eye
        pose_refine = self.graph.global_rigid.weight.data.detach().clone().view(-1,3,4)
        pose = camera.pose.compose([pose_refine, pose])
        return pose,pose_GT

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
        pose,pose_GT = self.get_all_training_poses(opt)
        pose_aligned,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
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
        iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in iterator:
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim_pose.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
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
            if opt.data.dataset in ["blender","llff", "llff_modified"]:
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()
                dict(
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

class Graph(nerf_inn_llff.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
        self.pose_eye = torch.eye(3,4).to(opt.device)

    def get_pose_init(self,opt,var,mode=None,ind=None,iter=None):
        """ ADHOC hack!!! Need to fix that in the future upon code release....
        """
        if mode=="train":
            
            pose_init = None
            if opt.data.dataset=="blender":
                if opt.camera.noise_type == "barf":
                    var.pose_noise = self.pose_noise[var.idx]
                    # left multiplication: transform camera around the object center #
                    pose = camera.pose.compose([var.pose_noise,var.pose])
                elif opt.camera.noise_type == "l2g":
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose, var.pose_noise])
                else: pose = var.pose
                pose_init = pose
            else: 
                pose = self.pose_eye
                pose = pose[None].repeat(len(var.idx),1,1)
            
            return pose            
    

    def get_pose(self,opt,var,mode=None,ind=None,iter=None):
        if mode=="train":
            
            ## COMMENT THIS FROR NOW ##
            pose_init = None
            if opt.data.dataset=="blender":
                if opt.camera.noise_type == "barf":
                    var.pose_noise = self.pose_noise[var.idx]
                    # left multiplication: transform camera around the object center #
                    pose = camera.pose.compose([var.pose_noise,var.pose])
                elif opt.camera.noise_type == "l2g":
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose, var.pose_noise])
                else: pose = var.pose
                pose_init = pose
            else: 
                pose = self.pose_eye
                pose = pose[None].repeat(len(var.idx),1,1)
            
            batch_size = len(var.idx)
            center_cam, grid_cam = camera.get_unwarped_center_and_ray(opt,intr=var.intr,ray_idx=var.ray_idx, pose_init=pose_init)# [B,HW,3] 

            ## v1: impose rigidity constraint on camera center ##
            center_cam = center_cam.detach() #(B,1,2)

            grid_cam = grid_cam.detach() #(B,N,2)

            ## call a forward pass to warp_mlp ##
            if opt.warp_latent.enc_type == "l2fbarf":
                feat = self.warp_latent.weight  #(B,D)
            elif opt.warp_latent.enc_type == "posenc":
                feat = self.positional_encoding(opt, self.frame_id, opt.warp_latent.posenc.freq_len)
            elif opt.warp_latent.enc_type == "extrinsic":
                rotation_vec = self.warp_latent.weight[:,:3]
                translation_vec = self.warp_latent.weight[:,3:]
                rot_feat = self.positional_encoding(opt, rotation_vec, opt.warp_latent.extrinsic.L)   #(B,24)
                trans_feat = self.positional_encoding(opt, rotation_vec, opt.warp_latent.extrinsic.L) #(B,24)
                
                rot_enc = torch.cat([rotation_vec, rot_feat], dim=-1)
                trans_enc = torch.cat([translation_vec, trans_feat], dim=-1)
                
                feat = torch.cat([rot_enc, trans_enc], dim=-1) 
 
            camera_coords_3D = torch.cat([grid_cam, center_cam], axis=1)  #(B,N+N,3)
            
            
            if opt.inn.real_nvp.c2f == True:
                alpha_ratio = max(min(iter / opt.inn.real_nvp.max_pe_iter,1),0)
            else:
                alpha_ratio = 1
            
            camera_coords_3D_warped = self.warp_mlp.forward(feat, camera_coords_3D.unsqueeze(2), alpha_ratio=alpha_ratio)
            grid_3D = camera_coords_3D_warped[:,:len(var.ray_idx),:]  #(B,N,1,3)
            center_3D = camera_coords_3D_warped[:,len(var.ray_idx):,:]  #(B,N,1,3)

            ## v1: expand camera center to ##
            #center_3D = center_3D.repeat(1,len(var.ray_idx), 1, 1) #(B,N,1,3)

            ray = grid_3D - center_3D
            return ray.squeeze(2), center_3D.squeeze(2), grid_3D.squeeze(2), alpha_ratio           
        elif mode == "render_train":
                
            batch_size = 1

            center_cam, grid_cam = camera.get_unwarped_center_and_ray(opt, intr=var.intr[ind].unsqueeze(0))
            center_cam = center_cam.detach()
            grid_cam = grid_cam.detach()

            if opt.warp_latent.enc_type == "l2fbarf":
                feat = self.warp_latent.weight

            num_rays = grid_cam.shape[1]
            camera_coords_3D = torch.cat([grid_cam, center_cam], axis=1)
            camera_coords_3D_warped = self.warp_mlp.forward(self.frame_id[ind][None], feat[ind][None], camera_coords_3D.unsqueeze(2))
            grid_3D = camera_coords_3D_warped[:,:num_rays]
            center_3D = camera_coords_3D_warped[:,num_rays:]
            ray = grid_3D - center_3D

            return ray.squeeze(2), center_3D.squeeze(2)
        
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
    
    def compute_angle_between_two_vectors(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

        cos_angle = dot_product / (mag1 * mag2)
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees
    
    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc

class NeRF(nerf_inn_llff.NeRF):

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
