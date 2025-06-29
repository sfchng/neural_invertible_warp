"""
Tanks_and_Temples dataloader, modified from NOPE
https://github.com/ActiveVisionLab/nope-nerf/blob/main/dataloading/dataset.py

By default, NOPE traines on Tandt images with a resolution of [540, 960]
https://github.com/ActiveVisionLab/nope-nerf/blob/main/configs/Tanks/Francis.yaml

Apart from that, they have this sample_rate parameter.
Need to revisit later.
"""
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log,debug

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 540,960
        super().__init__(opt,split)
        self.root = opt.data.root or "data/tandt"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}/images".format(self.path)
        image_fnames = sorted(os.listdir(self.path_image))

        ## NOTE: poses_raw here correlates with the self.poses 
        ## on L236 in./Documents/project/third_party/nerf_pl/datasets/llff.py
        poses_raw,bounds = self.parse_cameras_and_bounds(opt) ## poses_c2w_opengl
        
        ## Following NOPE-NERF, Spherify_poses ##
        poses_raw, _, bounds = self.spherify_poses(poses_raw, bounds)
        self.list = list(zip(image_fnames,poses_raw,bounds))
        
        # we follow NOPE for the train/test split 
        # determined by the data.val_ratio in the tandt.yaml
        ids = np.arange(len(self.list))
        i_test = ids[int(opt.data.val_ratio/2)::opt.data.val_ratio]
        i_train = np.array([i for i in ids if i not in i_test])
        i_val = i_test[:2]
        
        self.list_all = self.list.copy()
        if split == "train":
            self.list = [self.list_all[i] for i in i_train]
        elif split == "val":
            self.list = [self.list_all[i] for i in i_val]
        else:
            self.list = [self.list_all[i] for i in i_test]
            
        log.info("IDs for test images are {}".format(i_test))
        log.info("IDs for validation images are {}".format(i_val))
        log.info("Validation images count {}/{}".format(len(i_val), len(i_test)))
        # manually split train/val subsets
        # num_val_split = int(len(self)*opt.data.val_ratio)
        # self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        # if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def parse_cameras_and_bounds(self,opt):
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname),dtype=torch.float32)
        # parse cameras (intrinsics and poses)
        cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
        poses_raw = cam_data[...,:4] # [N,3,4]
        poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
        raw_H,raw_W,self.focal = cam_data[0,:,-1]
        assert(self.raw_H==raw_H and self.raw_W==raw_W)
        # parse depth bounds
        bounds = data[:,-2:] # [N,2]
        scale = 1./(bounds.min()*0.75) # not sure how this was determined
        poses_raw[...,3] *= scale
        bounds *= scale
        # roughly center camera poses
        poses_raw = self.center_camera_poses(opt,poses_raw)
        return poses_raw,bounds

    def center_camera_poses(self,opt,poses):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all
    
    def spherify_poses(self, poses, bds):
        
        poses = poses.numpy().astype(np.float32)
        bds = bds.numpy().astype(np.float32)
        p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
        
        rays_d = poses[:,:3,2:3]
        rays_o = poses[:,:3,3:4]

        def min_line_dist(rays_o, rays_d):
            A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
            b_i = -A_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        
        center = pt_mindist
        up = (poses[:,:3,3] - center).mean(0)

        vec0 = self.normalize(up)
        vec1 = self.normalize(np.cross([.1,.2,.3], vec0))
        vec2 = self.normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)

        poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
        
        sc = 1./rad
        poses_reset[:,:3,3] *= sc
        bds *= sc
        rad *= sc
        
        centroid = np.mean(poses_reset[:,:3,3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad**2-zh**2)
        new_poses = []
        
        for th in np.linspace(0.,2.*np.pi, 120):

            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0,0,-1.])

            vec2 = self.normalize(camorigin)
            vec0 = self.normalize(np.cross(vec2, up))
            vec1 = self.normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)

            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        
        new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
        poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
        
        return torch.from_numpy(poses_reset).to(torch.float32)[:,:,:4], new_poses, torch.from_numpy(bds).to(torch.float32)
    
    def normalize(self, x):
        return x / np.linalg.norm(x)

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        """
        NOTE: 
        poses_raw in parse_cameras_and_bounds() (L42) is a camera-to-world matrix and its coordinate system is [right,up,backwards],
        it is taken directly from the dataloader in the original NeRF
    
        Transform the pose_raw
        According to https://github.com/chenhsuanlin/bundle-adjusting-NeRF/issues/5,
        parse_raw_camera() aims to convert the camera information to 
        the standard extrinsic camera matrix (world-to-camera matrix [right,down,forward]), 
        ## 

        Also, refer to https://github.com/google-research/sparf/blob/main/source/datasets/llff.py
        """
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))  ### flips to [right,down,forwards]

        # Transforms OpenGL to OpenCV
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])

        # Gets W2C from C2W matrix
        pose = camera.pose.invert(pose)

        # Right now, the poses are facing in direction -Z. We want them to face direction +Z. 
        # That amounts to a rotation of 180 degrees around the x axis, ie the same pose_flip. 
        # Therefore, when initializing with identity matrix, the identity and ground-truth 
        # poses will face in the same direction. 
        pose = camera.pose.compose([pose_flip,pose])
        return pose
