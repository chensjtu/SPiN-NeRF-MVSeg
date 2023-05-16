import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import imageio
from skimage.transform import resize
import math

import sys
# sys.path.append('lib/dino')
import dino.dino_utils as utils
import dino.vision_transformer as vits
from reprojector import Reprojector


class Video_Seg_Dino:
    def __init__(self, data_dir, use_reproject=True):
        self.model = vits.__dict__['vit_base'](patch_size=8, num_classes=0)
        print(f"Video Seg Model built.")
        self.model.cuda()
        utils.load_pretrained_weights(self.model, '/workspace/dino_mae/dino/ckpt/dino_vitbase8_pretrain.pth', 'teacher', 'vit_base', 8)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        color_palette = []
        for line in open('/workspace/sa3d/lib/palette.txt', 'r').readlines():
            color_palette.append([int(i) for i in line.split('\n')[0].split(" ")])
        self.color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
        print(len(color_palette), "len pale")
        
        skip = 5
        rgbs_dir = os.path.join(data_dir, 'rgb')
        self.frame_list = self.read_frame_list(rgbs_dir)[::skip]

        self.use_reproject = use_reproject
        if use_reproject:
            depths_dir = os.path.join(data_dir, 'depth')
            self.depth_list = self.read_frame_list(depths_dir)[::skip]
            self.poses = np.loadtxt(os.path.join(data_dir, 'traj_w_c.txt'), delimiter=" ").reshape(-1, 4, 4)[::skip].astype(np.float32)
            img = imageio.imread(self.frame_list[0])
            H, W = img.shape[:2]
            hfov = 90
            focal = W / 2.0 / math.tan(math.radians(hfov / 2.0))
            self.K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ]).astype(np.float32)

    def norm_mask(self, mask):
        c, h, w = mask.size()
        for cnt in range(c):
            mask_cnt = mask[cnt,:,:]
            if(mask_cnt.max() > 0):
                mask_cnt = (mask_cnt - mask_cnt.min())
                mask_cnt = mask_cnt/mask_cnt.max()
                mask[cnt,:,:] = mask_cnt
        return mask
    
    def color_normalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x
    
    def to_one_hot(self, y_tensor, n_dims=None):
        """
        Take integer y (tensor or variable) with n dims &
        convert it to 1-hot representation with n+1 dims.
        """
        if(n_dims is None):
            n_dims = int(y_tensor.max()+ 1)
        _,h,w = y_tensor.size()
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device='cpu').scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(h,w,n_dims)
        return y_one_hot.permute(2, 0, 1).unsqueeze(0)
    
    def restrict_neighborhood(self, h, w):
        size_mask_neighborhood = 12
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w)
        for i in range(h):
            for j in range(w):
                for p in range(2 * size_mask_neighborhood + 1):
                    for q in range(2 * size_mask_neighborhood + 1):
                        if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                            continue
                        if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                            continue
                        mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

        mask = mask.reshape(h * w, h * w)
        return mask.cuda(non_blocking=True)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract one frame feature everytime."""
        out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token
        h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
        dim = out.shape[-1]
        out = out[0].reshape(h, w, dim)
        out = out.reshape(-1, dim)
        if return_h_w:
            return out, h, w
        return out
    
    def label_propagation(self, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
        """
        propagate segs of frames in list_frames to frame_tar
        """
        size_mask_neighborhood = 12
        
        ## we only need to extract feature of the target frame
        feat_tar, h, w = self.extract_feature(model, frame_tar, return_h_w=True)

        return_feat_tar = feat_tar.T # dim x h*w

        ncontext = len(list_frame_feats)
        feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

        feat_tar = F.normalize(feat_tar, dim=1, p=2)
        feat_sources = F.normalize(feat_sources, dim=1, p=2)

        feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
        aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

        if size_mask_neighborhood > 0:
            if mask_neighborhood is None:
                mask_neighborhood = self.restrict_neighborhood(h, w)
                mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
            aff *= mask_neighborhood

        aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
        tk_val, _ = torch.topk(aff, dim=0, k=5)
        tk_val_min, _ = torch.min(tk_val, dim=0)
        aff[aff < tk_val_min] = 0

        aff = aff / torch.sum(aff, keepdim=True, axis=0)

        list_segs = [s.cuda() for s in list_segs]
        segs = torch.cat(list_segs)
        nmb_context, C, h, w = segs.shape
        segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
        seg_tar = torch.mm(segs, aff)
        seg_tar = seg_tar.reshape(1, C, h, w)
        return seg_tar, return_feat_tar, mask_neighborhood

    def read_frame_list(self, video_dir):
        frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.png"))]
        frame_list = sorted(frame_list, key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        return frame_list

    def read_frame(self, frame_dir, scale_size=[480]):
        """
        read a single frame & preprocess
        """
        img = cv2.imread(frame_dir)
        ori_h, ori_w, _ = img.shape
        if len(scale_size) == 1:
            if(ori_h > ori_w):
                tw = scale_size[0]
                th = (tw * ori_h) / ori_w
                th = int((th // 64) * 64)
            else:
                th = scale_size[0]
                tw = (th * ori_w) / ori_h
                tw = int((tw // 64) * 64)
        else:
            th, tw = scale_size
        img = cv2.resize(img, (tw, th))
        img = img.astype(np.float32)
        img = img / 255.0
        img = img[:, :, ::-1]
        img = np.transpose(img.copy(), (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = self.color_normalize(img)
        return img, ori_h, ori_w

    def imwrite_indexed(self, filename, array, color_palette):
        """ Save indexed png for DAVIS."""
        if np.atleast_3d(array).shape[2] != 1:
            raise Exception("Saving indexed PNGs requires 2D array.")

        im = Image.fromarray(array*255)
        # im = Image.fromarray(array)
        im.save(filename, format='PNG')

    def preprocess_mask(self, seg, factor=8, scale_size=[480]):
        seg = Image.fromarray(seg)
        _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
        if len(scale_size) == 1:
            if(_w > _h):
                _th = scale_size[0]
                _tw = (_th * _w) / _h
                _tw = int((_tw // 64) * 64)
            else:
                _tw = scale_size[0]
                _th = (_tw * _h) / _w
                _th = int((_th // 64) * 64)
        else:
            _th = scale_size[1]
            _tw = scale_size[0]
        small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
        small_seg[small_seg >= 0.5] = 1
        small_seg[small_seg < 0.5] = 0
        small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
        print(np.unique(small_seg), "unique small seg")
        return self.to_one_hot(small_seg), np.asarray(seg).astype(np.uint8)

    @torch.no_grad()
    def eval_video_tracking_davis(self, model, frame_list, first_seg, seg_ori, color_palette, save_dir=None):
        """
        Evaluate tracking on a video given first frame & segmentation
        """
        n_last_frames = 15
        size_mask_neighborhood = 12
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # The queue stores the n preceeding frames
        que = queue.Queue(n_last_frames)

        # first frame
        frame1, ori_h, ori_w = self.read_frame(frame_list[0])
        # extract first frame feature
        frame1_feat = self.extract_feature(model, frame1).T #  dim x h*w

        # saving first segmentation
        if save_dir is not None:
            out_path = os.path.join(save_dir, "seg_ori.png")
            self.imwrite_indexed(out_path, seg_ori, color_palette)
        mask_neighborhood = None
        masks = []
        for cnt in tqdm(range(1, len(frame_list))):
            frame_tar = self.read_frame(frame_list[cnt])[0]

            # we use the first segmentation and the n previous ones
            used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
            used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

            frame_tar_avg, feat_tar, mask_neighborhood = self.label_propagation(model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)

            # pop out oldest frame if neccessary
            if que.qsize() == n_last_frames:
                que.get()
            # push current results into queue
            seg = copy.deepcopy(frame_tar_avg)
            que.put([feat_tar, seg])

            # upsampling & argmax
            # patch size 8
            frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=8, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
            frame_tar_avg = self.norm_mask(frame_tar_avg)
            _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

            # saving to disk
            frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
            frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
            masks.append(frame_tar_seg)
            frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
            if save_dir is not None:
                self.imwrite_indexed(os.path.join(save_dir, frame_nm), frame_tar_seg, color_palette)
        
        return np.stack(masks)

    def reproject(self, instance_mask_, idx_ori, save_dir=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        instance_mask = instance_mask_.copy()

        src_rgbs = np.stack([imageio.imread(img) for img in np.array(self.frame_list)]).astype(np.float32)
        src_poses = self.poses.copy()
        src_depths = np.stack([cv2.imread(depth, cv2.IMREAD_UNCHANGED)[..., None] / 1000.0 for depth in self.depth_list]).astype(np.float32)
        
        tgt_rgb, tgt_pose, tgt_depth = None, None, None
        for i in range(len(self.frame_list)):
            idx_img = self.frame_list[i].split('/')[-1].split('_')[-1][:-4]
            if idx_ori == int(idx_img):
                tgt_rgb = imageio.imread(self.frame_list[i])[None]
                tgt_pose = self.poses[i:i+1].copy()
                tgt_depth = cv2.imread(self.depth_list[i], cv2.IMREAD_UNCHANGED)[None, ..., None] / 1000.0
                break
        
        H, W = instance_mask.shape[:2]
        instance_mask = torch.from_numpy(instance_mask[None])
        reprojector = Reprojector(H, W, input_format='opencv', map_is_distance=False)
        src_poses, src_depths = torch.from_numpy(src_poses), torch.from_numpy(src_depths)
        tgt_pose, tgt_depth = torch.from_numpy(tgt_pose), torch.from_numpy(tgt_depth)
        tgt_intrinsic, src_intrinsics = torch.from_numpy(self.K)[None], torch.from_numpy(self.K)[None].repeat(src_poses.shape[0], 1, 1)
        idx_selected = reprojector.select_and_interp_views(instance_mask, tgt_pose, tgt_depth, tgt_intrinsic, \
                        src_poses, src_depths, src_intrinsics, tgt_rgb=tgt_rgb, src_rgb=src_rgbs, save_dir=save_dir)
        idx_selected = np.arange(len(self.frame_list))[idx_selected]

        if save_dir is not None:
            src_rgbs = src_rgbs[idx_selected]
            imageio.mimwrite(os.path.join(save_dir, 'video_src.mp4'), src_rgbs.astype(np.uint8), fps=30, quality=8)
            imageio.mimwrite(os.path.join(save_dir, 'video_tgt.mp4'), tgt_rgb.astype(np.uint8), fps=30, quality=8)
        
        return idx_selected


    def propagate_mask_with_dino(self, instance_mask, idx_ori, save_dir=None):
        '''
        Receive the target 0/1 instance mask and the corresponding original idx in rgbs.
        Return selected indexes and masks, containing the target instance mask.
        '''

        idx_selected_src = self.reproject(instance_mask, idx_ori, save_dir=save_dir)
        idx_selected_all = np.concatenate([np.array([idx_ori//5]), idx_selected_src])
        frame_list = np.array(self.frame_list)[idx_selected_all]
        # frame_list = self.frame_list

        first_seg, seg_ori = self.preprocess_mask(instance_mask)
        gen_masks = self.eval_video_tracking_davis(self.model, frame_list, first_seg, seg_ori, self.color_palette, save_dir=save_dir)
        masks_all = np.concatenate([instance_mask[None], gen_masks], axis=0)

        return idx_selected_all, masks_all




if __name__ == '__main__':

    data_dir = '/datasets/nerf_data/Replica/office_0/Sequence_1/'
    dino_segtor = Video_Seg_Dino(data_dir)
    save_dir = '../tmp_masks_zzw'
    # save_dir = None

    # id_instance = 19 # for test
    # idx_ori = 465
    id_instance = 19 # for test
    idx_ori = 465
    id_mask_dir = '/datasets/nerf_data/Replica/office_0/Sequence_1/semantic_instance/semantic_instance_{}.png'.format(idx_ori)
    id_mask = np.array(Image.open(id_mask_dir))
    instance_mask = id_mask.copy()
    instance_mask[instance_mask != id_instance] = -1
    instance_mask[instance_mask == id_instance] = 0
    instance_mask += 1
    dino_segtor.propagate_mask_with_dino(instance_mask, idx_ori, save_dir=save_dir)