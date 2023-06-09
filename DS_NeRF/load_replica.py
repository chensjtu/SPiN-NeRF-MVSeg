import os
import torch
import numpy as np
import imageio
import json
import scipy
import torch.nn.functional as F
import cv2
import math
import glob
# from .load_features import load_features
from video_segmentor import Video_Seg_Dino


office_0_list = [3,4,7,8,9,10,12,14,17,19,21,23,26,28,29,30,36,37,40,42,44,46,54,55,57,58,61,]
office_1_list = [3,7,9,11,13,14,15,17,23,24,29,32,33,36,37,39,42,44,45,46]
office_2_list = [0,2,8,9,13,14,17,19,23,27,40,41,47,49,51,54,58,60,65,67,70,71,72,73,78,85,90,92,93]
office_3_list = [3,8,11,14,15,18,19,25,29,30,32,33,38,39,43,51,54,55,61,65,72, 76,78,82,87,91,95,96,101,111]
office_4_list = [1,2,6,7,9,11,17,22,23,26,33,34,39,47,49,51,52,53,55,56]
room_0_list = [5,6,7,10,13,14,16,25,32,33,35,46,51,53,55,60,64,67,68,83,86,87,92]
room_1_list = [1,2,4,6,7,9,10,11,16,18,24,28,32,36,37,44,48,52,54,56]
room_2_list = [3,5,6,7,8,9,11,12,16,18,22,26,27,37,38,39,40,43,49,55,56]

cv2gl = np.eye(4)
cv2gl[2,2] = -1
cv2gl[1,1] = -1

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

trans_center = lambda centroid : torch.Tensor([
    [1,0,0,centroid[0]],
    [0,1,0,centroid[1]],
    [0,0,1,centroid[2]],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_gamma = lambda ga : torch.Tensor([
    [np.cos(ga),-np.sin(ga),0,0],
    [np.sin(ga), np.cos(ga),0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()


# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w

def pose_spherical(gamma, phi, t):
    c2w = torch.Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]).float()
    # c2w = trans_t(t)
    # c2w = trans_center(t)
    # c2w = rot_gamma(np.pi) @ c2w
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_gamma(gamma/180.*np.pi) @ c2w
    c2w[:3, 3] = t
    return c2w

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate)*zdelta, 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def rerotate_poses(poses):
    poses = np.copy(poses)
    centroid = poses[:,:3,3].mean(0)

    poses[:,:3,3] = poses[:,:3,3] - centroid

    # Find the minimum pca vector with minimum eigen value
    x = poses[:,:,3]
    mu = x.mean(0)
    cov = np.cov((x-mu).T)
    ev , eig = np.linalg.eig(cov)
    cams_up = eig[:,np.argmin(ev)]
    if cams_up[1] < 0:
        cams_up = -cams_up

    # Find rotation matrix that align cams_up with [0,1,0]
    R = scipy.spatial.transform.Rotation.align_vectors(
            [[0,1,0]], cams_up[None])[0].as_matrix()

    # Apply rotation and add back the centroid position
    poses[:,:3,:3] = R @ poses[:,:3,:3]
    poses[:,:3,[3]] = R @ poses[:,:3,[3]]
    poses[:,:3,3] = poses[:,:3,3] + centroid
    return poses

#####################


def spherify_poses(poses, bds, depths):

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

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    radius = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./radius
    poses_reset[:,:3,3] *= sc
    bds *= sc
    radius *= sc
    depths *= sc

    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, radius, bds, depths


def load_replica_data(basedir='./data/replica/office_0', half_res=False, args=None, movie_render_kwargs={}, bds=[0.1, 5.0], id_instance=0):
    poses = []
    with open(os.path.join(basedir, 'traj_w_c.txt'), 'r') as fp:
        for line in fp:
            tokens = line.split(' ')
            tokens = [float(token) for token in tokens]
            tokens = np.array(tokens).reshape(4, 4)
            poses.append(tokens)
    poses =  np.stack(poses, 0)

    # Ts_full = np.loadtxt(os.path.join(basedir, 'traj_w_c.txt'), delimiter=" ").reshape(-1, 4, 4)

    all_imgs_paths = sorted(os.listdir(os.path.join(basedir, 'rgb')), key=lambda file_name: int(file_name.split("_")[-1][:-4]))


    imgs = []
    for i in range(len(all_imgs_paths)):
        fname = os.path.join(basedir, 'rgb', all_imgs_paths[i])
        imgs.append(imageio.imread(fname))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    hfov = 90
    focal = W / 2.0 / math.tan(math.radians(hfov / 2.0))
    
    depth_list = sorted(glob.glob(os.path.join(basedir, 'depth/depth*.png')), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    depth_list = depth_list
    depths = [cv2.imread(idx_depth, cv2.IMREAD_UNCHANGED) / 1000.0 for idx_depth in depth_list]  # uint16 mm depth, then turn depth from mm to meter
    depths = np.array(depths).astype(np.float32)
    assert(depths.shape[0] == imgs.shape[0])


    centroid = torch.tensor([0, 0, 0]).float()
    centroid[0] += movie_render_kwargs.get('shift_x', 0)
    centroid[1] += movie_render_kwargs.get('shift_y', 0)
    centroid[2] += movie_render_kwargs.get('shift_z', 0)

    # render_poses = torch.stack([pose_spherical(angle, -30.0, 0.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_poses = torch.stack([pose_spherical(angle, -120.0, centroid) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # depths_half_res = np.zeros((depths.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            # depths_half_res[i] = cv2.resize(depths[i], (W, H), interpolation=cv2.INTER_LINEAR)
        imgs = imgs_half_res
        # depths = depths_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    step = 5
    train_ids = np.arange(0, poses.shape[0], step)
    test_ids = np.array([x+step//2 for x in train_ids])
    i_split = [train_ids, test_ids, test_ids]


    # get first mask
    semantic_instance_dir = os.path.join(basedir, 'semantic_instance')
    instance_list = sorted(glob.glob(semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    instances = np.stack([cv2.imread(instance, cv2.IMREAD_UNCHANGED) for instance in instance_list])
    instances_train = instances[train_ids]
    # maskdir = os.path.join(basedir, 'mask_instance') # visualization
    # os.makedirs(maskdir, exist_ok=True)
    counts = (instances_train.reshape(instances_train.shape[0], -1)==id_instance).sum(axis=-1)
    train_idx = np.argmax(counts)
    gt_masks = instances_train.copy()
    gt_masks[gt_masks != id_instance] = -1
    gt_masks[gt_masks == id_instance] = 0
    gt_masks += 1
    first_mask = gt_masks[train_idx].copy()
    # imageio.imwrite(os.path.join(maskdir, 'id_{:03d}_ori_{:03d}.jpg'.format(id_instance, train_idx*5)), (first_mask*255).astype(np.uint8))

    # get all mask, and put the first mask in the first of the list
    if first_mask.sum() == 0:
        raise TypeError('Mask of id {} is empty!! Skip training.'.format(id_instance))
    dino_segtor = Video_Seg_Dino(basedir)
    idx_selected_all, masks_all = dino_segtor.propagate_mask_with_dino(first_mask.astype(np.float32), \
                                                        train_idx*5, save_dir=None) # "/workspace/SPIn-NeRF/MVSeg/debug"
    # only select img with id_instance object
    i_split[0] = i_split[0][idx_selected_all[1:]]
    masks_all = masks_all[1:]
    gt_masks = gt_masks[idx_selected_all[1:]]
    idx_selected_all = idx_selected_all[1:]
    poses = poses @ cv2gl[None]
    # idx_selected_all = None
    # masks_all = np.zeros_like(depths)[train_ids]
    # gt_masks = np.zeros_like(depths)[train_ids]
    return imgs, poses, render_poses, bds, [H, W, focal], i_split, depths, idx_selected_all, masks_all.astype(np.uint16), gt_masks



if __name__ == "__main__":
    load_replica_data()
