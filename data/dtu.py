"""
DTU dataloader, modified from SPARF
'''
4xsubsampled version of DTU dataset, using rectified images
Resolution: 300x400
Scenes: 124 scenes
Poses: 49 poses

Loading DTU data in DVR format processed by pixelNeRF

Download dataset from:
https://github.com/sxyu/pixel-nerf

Data convention description not really followed:
https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/FAQ.md

- pixelNeRF uses whole P matrix, world_mat_i in this case is P = K [R|t] !
- IDR also uses DVR convention but different scale matrix

Since the camera matrices are the same for each scene, just load it once for 49 poses

Camera space follows OpenCV convention, x(right), y(down), z(in)
Cameras
#      0-4
#    10 - 5
#   11  -  18
#  27  xx   19
# 28    x    38
#48     -     39

'''
"""
import os
import imageio
import numpy as np
import torch
from PIL import Image
import cv2
import re
from typing import Any, List, Dict, Tuple

from . import base
from . import data_utils

def read_pfm(filename):
    """
    Args:
        filename (str)

    Return:
        np.arr 
    
    """
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



class Dataset(base.Dataset):
    
    def __init__(self,opt,split="train",subset=None):
        self.raw_H, self.raw_W = 300, 400
        super().__init__(opt, split)
        self.root = opt.data.root or "data/dtu"
 
        self.depth_dir = "{}/Depths".format(self.root)
        self.dtu_mask_path = "{}/submission_data/idrmasks".format(self.root)
        self.dtu_data_path = "{}/rs_dtu_4/DTU".format(self.root)

        # This is to rescale the poses and the depth maps. 
        # Here, I hard-coded 1./300. because for all the scenes I considered, the 
        # scaling factor "norm_scale" (see L.236) was always equal to [300., 300., 300.]
        # If this is not the case, this needs to be modified to be equal to 1./norm_scale. 
        # One should also verify that the scaling makes the depth maps consistent with the poses. 
        self.scaling_factor = 1./300.  

        self.near_depth = 1.2
        self.far_depth = 5.2

        self.scene = opt.data.scene
        print(f"Loading scene {self.scene} from DTU Dataset from split {self.split}...")
    
        scene_path = os.path.join(self.dtu_data_path, self.scene)
        _, rgb_files, intrinsics, poses = self.load_scene_data(scene_path)
        self.all_poses_c2w = poses

        ## split the files into train and test
        if opt.data.dtu.split_type == 'pixelnerf':
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            split_indices = {'test': test_idx, 'train': train_idx}
        elif opt.data.dtu.split_type == 'all':
            idx = list(np.arange(49))
            split_indices = {'test': idx, 'train': idx}
        elif opt.data.dtu.split_type == 'pixelnerf_reduced_testset':
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13, 24, 30, 41, 47, 43, 29, 45,
                        34, 33]
            test_idx = [1, 2, 9, 10, 11, 12, 14, 15, 23, 26, 27, 31, 32, 35, 42, 46]
            split_indices = {'test': test_idx, 'train': train_idx}
        else:
            all_indices = np.arange(len(rgb_files))
            split_indices = {
                'test': all_indices[all_indices % opt.data.dtu.dtuhold == 0],
                'train': all_indices[all_indices % opt.data.dtu.dtuhold != 0],
            }

        indices_train = split_indices['train']
        if opt.pose.dtu_reconstruction:
            print("Warning!!! Include all training images for dtu reconstruction !!!")
            indices_train = np.arange(len(rgb_files))
        indices_test = split_indices['test']

        # train split
        if opt.data.dtu.train_sub is not None:
            # here, will take the subset of 1, 3 or 9
            indices_train = indices_train[:self.opt.data.dtu.train_sub]

        if opt.data.dtu.val_sub is not None:
            indices_test = indices_test[:self.opt.data.dtu.val_sub]

        train_masks_files, test_masks_files = self._load_mask_paths(self.scene, indices_train, indices_test)
        # self.training_masks_files, self.test_masks_files

        train_rgb_files = np.array(rgb_files)[indices_train]    #(B)
        train_intrinsics = np.array(intrinsics)[indices_train]  #(B,4,4)
        train_poses = np.array(poses)[indices_train] #(B,4,4)

        test_rgb_files = np.array(rgb_files)[indices_test] 
        test_intrinsics = np.array(intrinsics)[indices_test]
        test_poses = np.array(poses)[indices_test]


        # rendering split 
        if 'train' in self.split:
            render_rgb_files = train_rgb_files
            render_intrinsics = train_intrinsics
            render_poses = train_poses
            img_indices = indices_train
            render_mask_files = train_masks_files
        else:
            render_rgb_files = test_rgb_files
            render_intrinsics = test_intrinsics
            render_poses = test_poses
            img_indices = indices_test
            render_mask_files = test_masks_files

        self.render_rgb_files = render_rgb_files.tolist()
        self.render_poses_c2w = render_poses
        self.render_intrinsics = render_intrinsics
        self.render_masks_files = render_mask_files
        self.render_img_id = img_indices

        print(f"In total there are {len(self.render_rgb_files)} images in this dataset")


    def load_scene_data(self, scene_path: str):
        """
        Load all images and poses 

        Args:
            scene_path: path to scene directory
        Returns: 
            list of file names (list): a list of file names
            rgb_files (list): a list of rgb file paths 
            intrinsics (np.arr [B,4,4])
            poses_c2w (np.arr [B,4,4]): poses defined with OpenCV
        """
        img_path = os.path.join(scene_path, "image")

        if not os.path.isdir(img_path):
            raise FileExistsError(img_path)

        # all images
        file_names = [f.split(".")[0] for f in sorted(os.listdir(img_path))]
        rgb_files = [os.path.join(img_path, f) for f in sorted(os.listdir(img_path))]
        pose_indices = [int(os.path.basename(e)[:-4]) for e in rgb_files] # this way is safer than range

        camera_info = np.load(os.path.join(scene_path, "cameras.npz"))

        intrinsics = []
        poses_c2w = []

        for p in pose_indices:
            P = camera_info[f"world_mat_{p}"] # Projection matrix   #(3,4)
            P = P[:3]  # (3x4) projection matrix 
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3] #(3,3),(3,3),(4,1) ##(OpenCV gives R: world-to-camera transformation, t: is c2w )
            K /= K[2, 2]  # 3x3 intrinsics matrix

            pose_c2w_ = np.eye(4, dtype=np.float32) # camera to world
            pose_c2w_[:3, :3] = R.transpose()
            pose_c2w_[:3, 3] = (t[:3] / t[3])[:, 0]  #(4,4)

            intrinsics_ = np.eye(4)
            intrinsics_[:3, :3] = K
            scale_mat = camera_info.get(f"scale_mat_{p}")  #(4,4)
            
            ## NOTE::: To get back the projection matrix, do:
            #projection_mat = K @ np.linalg.inv(pose_c2w_)[:3]
            if scale_mat is not None:
                self.norm_trans = scale_mat[:3, 3:]
                pose_c2w_[:3, 3:] -= self.norm_trans
                # 1/300, scale the world
                norm_scale = np.diagonal(scale_mat[:3, :3])[..., None]
                # here it is 3 values, but equal to each other!
                assert norm_scale.mean() == 300.
                # I directly use this scaling factor to scale the depth
                # it is hardcoded in self.scaling_factor 
                # If this assertion doesn't hold, them self.scaling_factor should be equal to 1./norm_scale
                # Importantly, the norm_scale must be equal for all directions, otherwise that wouldn't scale
                # the depth map properly. 

            pose_c2w_[:3, 3:] *= self.scaling_factor

            poses_c2w.append(pose_c2w_)
            intrinsics.append(intrinsics_)

        intrinsics = np.stack(intrinsics, axis=0)
        poses_c2w = np.stack(poses_c2w, axis=0)
        self.intrinsics = intrinsics

        return file_names, rgb_files, intrinsics, poses_c2w

    def _load_mask_paths(self, scene: str, train_idx, test_idx):
        """Load masks from disk.
        Args:
            scene (str): scene_name
            train_idx (np.arr)
            test_idx (np.arr) 
        Return:
            train_masks_files (list): List containing the mask files
            test_masks_files  (list): List containing the mask files
        """
        
        
        masks = []

        mask_path = self.dtu_mask_path
        idr_scans = ['scan40', 'scan55', 'scan63', 'scan110', 'scan114']
        if scene in idr_scans:
            maskf_fn = lambda x: os.path.join(  # pylint: disable=g-long-lambda
                mask_path, scene, 'mask', f'{x:03d}.png')
        else:
            maskf_fn = lambda x: os.path.join(  # pylint: disable=g-long-lambda
                mask_path, scene, f'{x:03d}.png')

        train_masks_files = [maskf_fn(i) for i in train_idx]
        test_masks_files = [maskf_fn(i) for i in test_idx]
        return train_masks_files, test_masks_files


    def read_depth(self, filename: str):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600) 

        depth_h *= self.scaling_factor
        # the depth is at the original resolution of the images
        return depth_h

    def get_all_camera_poses(self, args):
        # of the current split
        return torch.inverse(torch.from_numpy(self.render_poses_c2w))[:, :3].float()  # (B, 3, 4)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int)

        Returns:
            a dictionary for each image index containing the following elements: 
                * idx: the index of the image
                * rgb_path: the path to the RGB image. Will be used to save the renderings with a name. 
                * image: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are 
                            normalized to [0, 1] (not [0, 255]). 
                * intr: intrinsics parameters, numpy array of shape [3, 3]
                * pose:  world-to-camera transformation matrix in OpenCV format, numpy array of shaoe [3, 4]
                * depth_range: depth_range, numpy array of shape [1, 2]
                * scene: scene name

                * depth_gt: ground-truth depth map, numpy array of shape [H, W]
                * valid_depth_gt: mask indicating where the depth map is valid, bool numpy array of shape [H, W]
                * fg_mask: foreground segmentation mask, bool numpy array of shape [1, H, W]

        """
        opt = self.opt
        rgb_file = self.render_rgb_files[idx]
        render_pose_c2w = self.render_poses_c2w[idx]
        render_pose_w2c = np.linalg.inv(render_pose_c2w)
        render_intrinsics = self.render_intrinsics[idx]
        img_id = self.render_img_id[idx]
        scene = self.scene

        # read and handle the image to render
        rgb = imageio.imread(rgb_file)
        h, w = rgb.shape[:2]

        mask_file = self.render_masks_files[idx]
        if os.path.exists(mask_file):
            with open(mask_file, 'rb') as imgin:
                mask = np.array(Image.open(imgin), dtype=np.float32)[:, :, :3] / 255.  #(range from 0 to 1)
                mask = mask[:, :, 0]  # (H, W)
                mask = (mask == 1).astype(bool)
        else:
            mask = np.ones_like(rgb[:, :, 0], bool)  # (h, W)


        depth_filename = os.path.join(self.depth_dir, f'{scene}/depth_map_{img_id:04d}.pfm')
        if os.path.exists(depth_filename):
            depth_gt = self.read_depth(depth_filename)
        else:
            print(f'Could not find {depth_filename}')
            depth_gt = np.zeros((h, w), dtype=np.float32)

        rgb, render_intrinsics, depth_gt, mask = \
            self.preprocess_image_and_intrinsics(rgb, intr=render_intrinsics, 
            depth=depth_gt, mask=mask, channel_first=False)

        valid_depth_gt = depth_gt > 0.

        if self.opt.data.dtu.mask_img:
            # we do not want a black background
            # instead white is better
            mask_torch = torch.from_numpy(mask).unsqueeze(-1).float()
            rgb = rgb * mask_torch + 1 - mask_torch
            valid_depth_gt = valid_depth_gt & mask

        near_depth = self.near_depth * (1-self.opt.data.dtu.increase_depth_range_by_x_percent)
        far_depth = self.far_depth * (1+self.opt.data.dtu.increase_depth_range_by_x_percent)
        depth_range = torch.tensor([near_depth, far_depth], dtype=torch.float32)

        assert mask.shape[:2] == rgb.shape[:2]
        assert depth_gt.shape[:2] == rgb.shape[:2]
        assert valid_depth_gt.shape[:2] == rgb.shape[:2]
        ret =  {
            'idx': idx, 
            "rgb_path": rgb_file,
            'depth_gt': depth_gt, # (H, W)
            'fg_mask': np.expand_dims(mask, 0),  # numpy array, (1, H, W), bool
            'valid_depth_gt': valid_depth_gt,  # (H, W) 
            'image': rgb.permute(2, 0, 1), # torch tensor 3, self.H, self.W
            'intr': render_intrinsics[:3, :3].astype(np.float32),
            'pose':  render_pose_w2c[:3].astype(np.float32),   # # 3x4, world to camera
            "depth_range": depth_range,
            'scene': self.scene
        }
        return ret

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])


    def preprocess_image_and_intrinsics(self, image, intr, depth=None, 
                                        mask_depth=None, mask=None, 
                                        new_size=None, channel_first=True):
        """Resizes images to the desired size. Adjust intrinsics accordingly. If provided, resize the depth map
        and masks as well. 
        Return the image as a torch.Tensor, normalized to [0, 1]. The others are numpy arrays. 

        Args:
            image (np.array): (H, W, 3)
            intr (np.array): (3, 3)
            depth (np.array, optional): (H, W). Defaults to None.
            mask_depth (np.array, optional): (H, W). Defaults to None.
            mask (np.array, optional): (H, W). Defaults to None.
            new_size (list, optional): (H_new, W_new). Defaults to None.
            channel_first (bool, optional): _description_. Defaults to True.
        """
        image = np.array(image).astype(np.float32)

        # crop image and modify intrinsics accordingly
        if self.opt.data.dtu.crop_ratio or self.opt.data.dtu.crop:
            if self.opt.data.dtu.crop_ratio is not None:
                crop_H = int(self.raw_H*self.opt.data.dtu.crop_ratio)
                crop_W = int(self.raw_W*self.opt.data.dtu.crop_ratio)
            elif self.opt.data.dtu.crop is not None:
                crop_H, crop_W = self.opt.data.dtu.crop
            # will make sure the size is dividable by 2
            crop_H = crop_H + 1 if crop_H % 2 == 1 else crop_H
            crop_W = crop_W + 1 if crop_W % 2 == 1 else crop_W

            image, intr, depth = data_utils.crop(image, size=(crop_H, crop_W), intr=intr, 
                                    other=[depth, mask_depth, mask], random=self.split == 'train')
 
        # resize
        fn = max if self.opt.data.dtu.resize_by == 'max' else min
        # will make sure the size is dividable by 2, embedded in the functions
        if new_size is not None:
            image, intr = data_utils.resize_image_w_intrinsics(image, new_size, None, intr=intr, fn=fn)
        elif self.opt.data.dtu.resize or self.opt.data.dtu.resize_factor:
            new_size = self.opt.data.dtu.resize
            resize_factor = self.opt.data.dtu.resize_factor
            image, intr = data_utils.resize_image_w_intrinsics(image, new_size, resize_factor, intr=intr, fn=fn)
        
        # convert image to torch.Tensor and normalize to [0, 1]
        image = data_utils.numpy_image_to_torch(image) # shape (3,H,W) and [0, 1] 
        img_shape = image.shape[-2:]  # (H, W)

        if not channel_first:
            image = image.permute(1, 2, 0)  # shape is (H, W, 3) and [0, 1]

        ret = [image, intr]

        if depth is not None:
            depth, _ = data_utils.resize(depth, img_shape, interp='nearest')
            ret.append(depth)
        if mask_depth is not None:
            mask_depth, _ = data_utils.resize(mask_depth.astype(np.float32), img_shape)
            mask_depth = np.floor(mask_depth).astype(bool)
            ret.append(mask_depth)
        if mask is not None:
            mask, _ = data_utils.resize(mask.astype(np.float32), img_shape)
            mask = np.floor(mask).astype(bool)
            ret.append(mask)
        return ret
