import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import lpips
from external.pohsun_ssim import pytorch_ssim

import util,util_vis
from util import log,debug

from core.metrics import compute_depth_error_on_rays, compute_depth_metrics
from . import base
import camera
import timeit
import imageio

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device))

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(),lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer

        # training
        if self.iter_start==0: self.validate(opt,0)
        
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for self.it in loader:
            if self.it<self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            #start_time = time.time()
            self.train_iteration(opt,var,loader)
            #print("%s --- %s seconds ----\n" %(self.it, time.time() - start_time))   

            if opt.optim.sched: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
            
            if self.it % opt.freq.early_termination == 0:
                log.info("EARLY TERMINATION....")
                break
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    def train_iteration(self,opt,var,loader):
        # before train iteration
        self.timer.it_start = time.time()

        # train iteration
        self.optim.zero_grad()
        var = self.graph.forward(opt,var,mode="train",iter=self.it)
        loss = self.graph.compute_loss(opt,var,mode="train")
        loss = self.summarize_loss(opt,var,loss)
        loss.all.backward()
        self.optim.step()
        
        #print("%s --- %s seconds ----\n" %(self.it, time.time() - start_time))  
        # after train iteration
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,var,loss,step=self.it+1,split="train")
        if (self.it+1)%opt.freq.vis==0: self.visualize(opt,var,step=self.it+1,split="train")
        self.it += 1

        ## (???) Not sure why set_postfix for nerf takes up much longer time compared to nerf_inn 
        #start_time = time.time()
        #loader.set_postfix(it=self.it,loss="{:.3f}".format(loss.all))
        #print("%s --- %s seconds ----\n" %(self.it, (time.time() - start_time)))  
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        
        ## log learning rate ##
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
            
            ## if depth is available, compute depth error ##
            if 'depth_gt' in var.keys() and 'depth' in var.keys():
                scaling_factor_for_pred_depth = 1
                if 'barf'.lower() in opt.model.lower() and hasattr(self.graph.pose_net, 'sim3_est_to_gt_c2w'):
                    # adjust the scaling of the depth since the optimized scene geometry + poses 
                    # are all valid up to a 3D similarity
                    scaling_factor_for_pred_depth = self.graph.pose_net.sim3_est_to_gt_c2w.s
                abs_depth_err, rms_depth_err = compute_depth_error_on_rays(var, scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
                self.tb.add_scalar("{0}/{1}".format(split, "depth_abs_err"), abs_depth_err, step)
                self.tb.add_scalar("{0}/{1}".format(split, "depth_rms_err"), rms_depth_err, step)
            
            
        ## log PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        if opt.nerf.fine_sampling:
            psnr = -10*loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:

            # ## add visualization for train ##
            # if split == "train":
            #     self.graph.eval()
            #     var = self.graph.forward(opt, var, mode="render_train")
            #     invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            #     rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            #     invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            #     util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
            #     util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map) 
            #     util_vis.tb_image(opt,self.tb,step,split,"image",var.image[var.render_train_idx][None])
            #     if opt.nerf.fine_sampling:
            #         invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
            #         rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            #         invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            #         util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
            #         util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)                
            if not opt.nerf.rand_rays or split!="train":
                util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)                
            
                scaling_factor_for_pred_depth = 1
                if 'barf'.lower() in opt.model.lower() and hasattr(self.graph.pose_net, 'sim3_est_to_gt_c2w'):
                    # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
                    # similarity, compared to the ground-truth. 
                    scaling_factor_for_pred_depth = self.graph.pose_net.sim3_est_to_gt_c2w.s
                    depth_pred = var.depth.view(-1,opt.H,opt.W, 1)* scaling_factor_for_pred_depth #(B,H,W,1)

                    if hasattr(var, 'depth_range') and opt.nerf.depth.param == "metric":
                        depth_range = var.depth_range[0].cpu().numpy().tolist()
                        
                    depth_gt = var.depth_gt[0]  #(H,W)
                    depth_gt_colored = (255 * util_vis.colorize_np(depth_gt.cpu().numpy(), 
                                                          range=depth_range, append_cbar=False)).astype(np.uint8)  #(H,W,3)
                    
                    depth_pred_colored = (255 * util_vis.colorize_np(depth_pred[0].squeeze(-1).cpu().numpy(), 
                                                          range=depth_range, append_cbar=False)).astype(np.uint8)  #(H,W,3)
                    
                    self.tb.add_image("{}/depth_map_gt".format(split), torch.from_numpy(depth_gt_colored).permute(2,0,1), step)
                    self.tb.add_image("{}/depth_map_pred".format(split), torch.from_numpy(depth_pred_colored).permute(2,0,1), step)
                    
                # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                # util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)
                    
                

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None,pose_GT

    @torch.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        res_masked = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        MIN_DEPTH = 1.e-3
        MAX_DEPTH = 10
        for i,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            if 'barf'.lower() in opt.model.lower() and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
            var = self.graph.forward(opt,var,mode="eval")
            
            
            ### evaluate view synthesis
            # rendered image and depth #
            pred_rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            pred_depth = var.depth
            
            ## evaluate rendered image ##
            psnr = -10*self.graph.MSE_loss(pred_rgb_map,var.image).log10().item()
            ssim = pytorch_ssim.ssim(pred_rgb_map,var.image).item()
            lpips = self.lpips_loss(pred_rgb_map*2-1,var.image*2-1).item()
            
            if 'barf'.lower() in opt.model.lower() and hasattr(self.graph.pose_net, "sim3_est_to_gt_c2w"):
                scaling_factor_for_pred_depth = self.graph.pose_net.sim3_est_to_gt_c2w.s
           ## evaluate rendered depthmap ##    
            abs_err, rms_err = float('nan'), float('nan')  
            if 'depth_gt' in var.keys():
                log.info("Evaluating depthmap")  
                if scaling_factor_for_pred_depth != 1.:
                    abs_err, rms_err = compute_depth_metrics(var, scaling_factor_for_pred_depth)
            
            if 'fg_mask' in var.keys():
                log.info("Computing masked metrics ....")
                mask_float = var.fg_mask.float() #(B,3,H,W)
                assert mask_float.shape[1] == 3 or mask_float.shape[1] == 1
                mask = (mask_float == 1.)
                
                pred_rgb_fg = pred_rgb_map * mask_float + (1. - mask_float)
                gt_rgb_fg = var.image * mask_float + (1. - mask_float)  

                psnr_masked = -10*self.graph.MSE_loss(pred_rgb_fg,gt_rgb_fg).log10().item()
                ssim_masked = pytorch_ssim.ssim(pred_rgb_fg,gt_rgb_fg).item()
                lpips_masked = self.lpips_loss(pred_rgb_fg*2-1,gt_rgb_fg*2-1).item()  
                
                          
            pred_depth = pred_depth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) * scaling_factor_for_pred_depth #(B,1,H,W)
            depth_range = None
            if hasattr(var, 'depth_range'):
                depth_range = var.depth_range[0].cpu().numpy().tolist()
                depth_gt = var.depth_gt[0]  #(H,W)

                
                depth_gt_vis =(255 * util_vis.colorize_np(depth_gt.cpu().squeeze().numpy(), 
                                                          range=depth_range, append_cbar=False, cmap_name='magma')).astype(np.uint8) #(H,W,3)
            
                depth_pred_vis = (255 * util_vis.colorize_np(pred_depth[0][0].cpu().squeeze().numpy(), 
                                                          range=depth_range, append_cbar=False, cmap_name='magma')).astype(np.uint8)#(H,W,3)
            
            
            
            ## inverse depthmap ##
            # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            # #rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]

            res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            res_masked.append(edict(psnr_masked=psnr_masked, ssim_masked=ssim_masked, lpips_masked=lpips_masked))
            # dump rendered image and depth #
            
            pred_rgb_map = (pred_rgb_map.permute(0,2,3,1)[0].cpu().numpy() * 255).astype(np.uint8)  #(B,H,W,3)-> (H,W,3)
            rgb_gt = (var.image.permute(0,2,3,1)[0].cpu().numpy() * 255).astype(np.uint8)#(B,H,W,3)-> (H,W,3)
            
            imageio.imwrite("{}/rgb_{}.png".format(test_path, i), pred_rgb_map)
            imageio.imwrite("{}/rgb_GT_{}.png".format(test_path, i), rgb_gt)
            imageio.imwrite("{}/depth_{}.png".format(test_path, i), depth_pred_vis)
            imageio.imwrite("{}/depth_GT_{}.png".format(test_path, i), depth_gt_vis)            
                                 
                        
            # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            # rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            # psnr = -10*self.graph.MSE_loss(rgb_map,var.image).log10().item()
            # ssim = pytorch_ssim.ssim(rgb_map,var.image).item()
            # lpips = self.lpips_loss(rgb_map*2-1,var.image*2-1).item()
            # res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            # # dump novel views
            # torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(test_path,i))
            # torchvision_F.to_pil_image(var.image.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path,i))
            # torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(test_path,i))
        # show results in terminal
        print("---------NVS (UNMASKED)-----------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")

        print("---------NVS (MASKED)-----------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr_masked for r in res_masked])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim_masked for r in res_masked])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips_masked for r in res_masked])))
        
        print("--------------------------")
        print("abs err {:8.2f}, rms err{:8.2f}".format(abs_err, rms_err))
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write("{} {} {} {}\n".format(i,r.psnr,r.ssim,r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10):
        self.graph.eval()
        if opt.data.dataset=="blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,depth_vid_fname))
        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model in ["barf","barf_se3_field"] else pose_GT
            if opt.model in ["barf","barf_se3_field"] and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale).to(opt.device)
            # render the novel views
            # novel_path = "{}/novel_view".format(opt.output_path)
            # os.makedirs(novel_path,exist_ok=True)
            # pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
            # intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics
            # for i,pose in enumerate(pose_novel_tqdm):
            #     ret = self.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
            #           self.graph.render(opt,pose[None],intr=intr)
            #     invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
            #     rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            #     invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            #     torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
            #     torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))
            # # write videos
            # print("writing videos...")
            # rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            # depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            # os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
            # os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)

    def forward(self,opt,var,mode=None,iter=None):
        batch_size = len(var.idx)
        if opt.nerf.depth.param == "inverse":
            depth_range = opt.nerf.depth.range
        else:
            depth_range = var.depth_range[0]
        # render images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:

            # sample random rays for optimization
            ## NOTE::: Assume random sampling for now.. ##
            var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]

            ## get current pose estimates ##
            ray, center, grid_3d = self.get_pose(opt,var,mode=mode, iter=iter) #(:,3,4)            

            ret = self.render_local(opt,ray,center,intr=var.intr,mode=mode,depth_range=depth_range) # [B,N,3],[B,N,1]
            
            ret.update(grid_local=grid_3d, center_local=center, grid_init=self.pose_net.grid_init, center_init=self.pose_net.center_init)
        else:          
            pose_w2c = self.get_pose(opt,var,mode=mode) #(:,3,4)
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose_w2c,intr=var.intr,mode=mode,depth_range=depth_range) if opt.nerf.rand_rays else \
                  self.render(opt,pose_w2c,intr=var.intr,mode=mode,depth_range=depth_range) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb,image)
        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)
        if mode == "train" and opt.loss_weight.global_alignment is not None:
            target = torch.cat([var.grid_local, var.center_local], dim=1)
            source = torch.cat([var.grid_init, var.center_init],dim=1)
            pose_global_w2c = self.pose_net.get_w2c_poses()
            loss.global_alignment = self.MSE_loss(target, camera.cam2world(source, pose_global_w2c))
        return loss

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render_local(self,opt,ray, center,intr=None,ray_idx=None,mode=None,depth_range=None):
        """
        Main rendering function
        
        Args:
            opt (edict)
            ray (torch.Tensor [B,N,3]): ray in world coordinates
            center (torch.Tensor [B,N,3]): ray in world coordinates
            intr (torch.Tensor [B,3,3]): intrinsic matrices
            ray_idx (torch.Tensor): the indices of pixel locations to be rendered
            mode (str)
        """
        batch_size = len(intr)
        
        if ray_idx is not None:
            center, ray = center[:, ray_idx], ray[:, ray_idx]  #(B,N,3)

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
            
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1],depth_range=depth_range) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices_local(self,opt,pose,intr=None,mode=None,depth_range=None):
        ret_all = edict(rgb=[],depth=[],opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,opt.H*opt.W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode,depth_range=depth_range) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all


    def render(self,opt,pose,intr=None,ray_idx=None,mode=None,depth_range=None):
        """
        Main rendering function
        
        Args:
            opt (edict)
            pose (torch.Tensor [B,3,4]): w2c poses at which to render
            intr (torch.Tensor [B,3,3]): intrinsic matrices
            ray_idx (torch.Tensor): the indices of pixel locations to be rendered
            mode (str)
        """
        batch_size = len(pose)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
            
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1],depth_range=depth_range) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices(self,opt,pose,intr=None,mode=None,depth_range=None):
        ret_all = edict(rgb=[],depth=[],opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,opt.H*opt.W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode,depth_range=depth_range) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays=None,depth_range=None):
        """ Sample depths along the ray. The same depth range is used for all the rays
        
        Args:
            opt (edict): Containing all the settings
            batch_size (int)
            num_rays (int)
        
        Returns:
            depth_samples (torch.Tensor [B,num_rays,num_samples,1])
        """
        
        depth_min, depth_max = depth_range

        num_rays = num_rays or opt.H*opt.W
        rand_samples = torch.rand(batch_size,num_rays,opt.nerf.sample_intvs,1,device=opt.device) if opt.nerf.sample_stratified else 0.5 #(B,HW,N,1)
        rand_samples += torch.arange(opt.nerf.sample_intvs,device=opt.device)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0,1,opt.nerf.sample_intvs_fine+1,device=opt.device) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1,device=opt.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]

class NeRF(torch.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        input_3D_dim = 3+6*opt.arch.posenc.L_3D if opt.arch.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.arch.posenc.L_view if opt.arch.posenc else 3
        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(L)-1: k_out += 1
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_rgb)
        feat_dim = opt.arch.layers_feat[-1]
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="all" if li==len(L)-1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self,opt,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self,opt,points_3D,ray_unit=None,mode=None): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
            points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = points_3D
        feat = points_enc
        # extract coordinate-based features
        for li,layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_feat)-1:
                density = feat[...,0]
                if opt.nerf.density_noise_reg and mode=="train":
                    density += torch.randn_like(density)*opt.nerf.density_noise_reg
                density_activ = getattr(torch_F,opt.arch.density_activ) # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[...,1:]
            feat = torch_F.relu(feat)
        # predict RGB values
        if opt.nerf.view_dep:
            assert(ray_unit is not None)
            if opt.arch.posenc:
                ray_enc = self.positional_encoding(opt,ray_unit,L=opt.arch.posenc.L_view)
                ray_enc = torch.cat([ray_unit,ray_enc],dim=-1) # [B,...,6L+3]
            else: ray_enc = ray_unit
            feat = torch.cat([feat,ray_enc],dim=-1)
        for li,layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li!=len(self.mlp_rgb)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]
        return rgb,density

    def forward_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,density_samples = self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = density_samples*dist_samples # [B,HW,N]
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        return rgb,depth,opacity,prob # [B,HW,K]

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
