import torch
import numpy as np
import os
import sys
import argparse
from trainer import Trainer
sys.path.append('./motion')
sys.path.append('./etc')
sys.path.append('./preprocess')
from Quaternions import Quaternions
import Animation as Animation
import BVH as BVH
from remove_fs import remove_foot_sliding
from utils import ensure_dirs, get_config
from generate_dataset import *
from output2bvh import compute_posture


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mixamo_chosen_joints = np.array([
    0,  # Hips
    1,   2,  3,  4,  # Left whole leg
    5,   6,  7,  8,  # Right whole leg
    9,  10, 11, 12,  # Spine to Head
    13, 14, 15, 16,  # Left whole arm
    17, 18, 19, 20,  # Right whole arm
])

parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19])


def process_mixamo_data(filename, window=120, window_step=60, divide=True):
    anim, names, frametime = BVH.load(filename)

    """ Convert to 60 fps """
    # anim = anim[::2]

    """ Do FK """
    global_xforms = Animation.transforms_global(anim)

    """ Remove trash joints """
    # global_xforms = global_xforms[:, np.array([
    # 0,  # Hips
    # 1,   2,  3,  4,  # Left whole leg
    # 5,   6,  7,  8,  # Right whole leg
    # 9,  10, 11, 12,  # Spine to Head
    # 13, 14, 15, 19,  # Left whole arm
    # 31, 32, 33, 37,  # Right whole arm
    # ])
    #                                            ]

    global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    global_forwards = global_xforms[:, :, :3, 2]
    global_ups = global_xforms[:, :, :3, 1]

    """ Put on Floor """
    fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
    foot_heights = np.minimum(global_positions[:, fid_l, 1], global_positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    global_positions[:, :, 1] -= floor_height
    global_xforms[:, :, 1, 3] -= floor_height

    """ Extract Forward Direction and smooth """
    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across = (
        (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
        (global_positions[:, hip_l] - global_positions[:, hip_r])
        )
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotation = Quaternions.between(forward, target)[:, np.newaxis]

    """ Local Space """
    root_xforms = np.zeros((len(global_xforms), 1, 4, 4))
    root_xforms[:, :, :3, :3] = root_rotation.transforms()
    root_position = global_positions[:, 0:1]
    root_position[..., 1] = 0
    root_xforms[:, :, :3, 3] = np.matmul(-root_xforms[:, :, :3, :3],
                                            root_position[..., np.newaxis]).squeeze(axis=-1)  # root translation
    root_xforms[:, :, 3:4, 3] = 1.0

    local_xforms = global_xforms.copy()
    local_xforms = np.matmul(root_xforms[:-1], local_xforms[:-1])
    local_positions = local_xforms[:, :, :3, 3] / local_xforms[:, :, 3:, 3]
    local_velocities = np.matmul(root_xforms[:-1, :, :3, :3],
                                    (global_positions[1:] - global_positions[:-1])[..., np.newaxis]).squeeze(axis=-1)
    local_forwards = local_xforms[:, :, :3, 2]
    local_ups = local_xforms[:, :, :3, 1]

    root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps    # to angle-axis

    """ Foot Contacts """
    fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
    velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])
    feet_l_x = (global_positions[1:, fid_l, 0] - global_positions[:-1, fid_l, 0])**2
    feet_l_y = (global_positions[1:, fid_l, 1] - global_positions[:-1, fid_l, 1])**2
    feet_l_z = (global_positions[1:, fid_l, 2] - global_positions[:-1, fid_l, 2])**2
    feet_l_h = global_positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)
              & (feet_l_h < heightfactor)).astype(np.float)

    feet_r_x = (global_positions[1:, fid_r, 0] - global_positions[:-1, fid_r, 0])**2
    feet_r_y = (global_positions[1:, fid_r, 1] - global_positions[:-1, fid_r, 1])**2
    feet_r_z = (global_positions[1:, fid_r, 2] - global_positions[:-1, fid_r, 2])**2
    feet_r_h = global_positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)
              & (feet_r_h < heightfactor)).astype(np.float)

    foot_contacts = np.concatenate([feet_l, feet_r], axis=-1).astype(np.int32)

    """ Stack all features """
    local_full = np.concatenate([local_positions, local_forwards, local_ups, local_velocities], axis=-1)  # for joint-wise
    root_full = np.concatenate([root_velocity[:, :, 0:1], root_velocity[:, :, 2:3], np.expand_dims(root_rvelocity, axis=-1)], axis=-1)

    """ Slide over windows """
    local_windows = divide_clip(local_full, window, window_step, divide=divide)
    root_windows = divide_clip(root_full, window, window_step, divide=divide)
    foot_contacts_windows = divide_clip(foot_contacts, window, window_step, divide=divide)

    return local_windows, root_windows, foot_contacts_windows


def initialize_path(config):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    ensure_dirs([config['main_dir'], config['model_dir']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='./model_ours/info/config.yaml',
                        help='Path to the config file.')
    parser.add_argument('--content', 
                        type=str, 
                        default='./output/Anthro_Mocap_Walk.bvh',
                        help='Path to the content bvh file.')
    parser.add_argument('--style', 
                        type=str, 
                        default='./angry_13_000.bvh',
                        help='Path to the style bvh file.')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output')
    parser.add_argument('--remove_fs',
                        type=bool, 
                        default=True)
    args = parser.parse_args()

    # initialize path
    cfg = get_config(args.config)
    initialize_path(cfg)

    # make output path folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # input content and style bvh file
    content_bvh_file = args.content
    style_bvh_file = args.style

    # w/w.o post-processing
    remove_fs = args.remove_fs

    content_name = os.path.split(content_bvh_file)[-1].split('.')[0]
    style_name = os.path.split(style_bvh_file)[-1].split('.')[0]
    recon_name = content_name + '_recon'
    trans_name = 'Style_' + style_name + '_Content_' + content_name
    
    # import norm
    data_norm_dir = os.path.join(cfg['data_dir'], 'norm')
    mean_path = os.path.join(data_norm_dir, "motion_mean.npy")
    std_path = os.path.join(data_norm_dir, "motion_std.npy")
    mean = np.load(mean_path, allow_pickle=True).astype(np.float32)
    std = np.load(std_path, allow_pickle=True).astype(np.float32)
    mean = mean[:, np.newaxis, np.newaxis]
    std = std[:, np.newaxis, np.newaxis]

    # import motion(bvh) and pre-processing
    cnt_mot, cnt_root, cnt_fc = \
        process_mixamo_data(content_bvh_file, divide=False)
    cnt_mot, cnt_root, cnt_fc = cnt_mot[0], cnt_root[0], cnt_fc[0]
    
    sty_mot, sty_root, sty_fc = \
        process_mixamo_data(style_bvh_file, divide=False)
    sty_mot, sty_root, sty_fc = sty_mot[0], sty_root[0], sty_fc[0]

    # normalization
    cnt_motion_raw = np.transpose(cnt_mot, (2, 1, 0))
    sty_motion_raw = np.transpose(sty_mot, (2, 1, 0))
    cnt_motion = (cnt_motion_raw - mean) / std
    sty_motion = (sty_motion_raw - mean) / std
    
    cnt_motion = torch.from_numpy(cnt_motion[np.newaxis].astype('float32'))     # (1, dim, joint, seq)
    sty_motion = torch.from_numpy(sty_motion[np.newaxis].astype('float32'))

    # Trainer
    trainer = Trainer(cfg)
    epochs = trainer.load_checkpoint()
    
    # for bvh
    rest, names, _ = BVH.load(content_bvh_file)
    names = np.array(names)
    names = names[mixamo_chosen_joints].tolist()
    offsets = rest.copy().offsets[mixamo_chosen_joints]
    orients = Quaternions.id(len(parents))
    
    loss_test = {}
    with torch.no_grad():
        cnt_data = cnt_motion.to(device) 
        sty_data = sty_motion.to(device)
        cnt_fc = np.transpose(cnt_fc, (1,0))

        outputs, loss_test_dict = trainer.test(cnt_data, sty_data)
        rec = outputs["recon_con"].squeeze()
        tra = outputs["stylized"].squeeze()
        con_gt = outputs["con_gt"].squeeze()
        sty_gt = outputs["sty_gt"].squeeze()

        # rec = rec.numpy()*std + mean
        # tra = tra.numpy()*std + mean
        # con_gt = con_gt.numpy()*std + mean
        # sty_gt = sty_gt.numpy()*std + mean
        rec = rec.cpu().numpy() * std + mean
        tra = tra.cpu().numpy() * std + mean
        con_gt = con_gt.cpu().numpy() * std + mean
        sty_gt = sty_gt.cpu().numpy() * std + mean

        tra_root = cnt_root

        con_gt_mot, rec_mot = [compute_posture(raw, cnt_root) for raw in [con_gt, rec]]
        tra_mot = compute_posture(tra, tra_root)
        sty_gt_mot = compute_posture(sty_gt, sty_root)
        mots = [sty_gt_mot, con_gt_mot, rec_mot, tra_mot]
        fnames = [style_name, content_name, recon_name, trans_name]
        for mot, fname in zip(mots, fnames):
            local_joint_xforms = mot['local_joint_xforms']

            s = local_joint_xforms.shape[:2]
            rotations = Quaternions.id(s)
            for f in range(s[0]):
                for j in range(s[1]):
                    rotations[f, j] = Quaternions.from_transforms2(local_joint_xforms[f, j])
            
            positions = offsets[np.newaxis].repeat(len(rotations), axis=0)
            positions[:, 0:1] = mot['positions'][:, 0:1]

            anim = Animation.Animation(rotations, positions, 
                                        orients, offsets, parents)

            file_path = os.path.join(output_dir, fname + ".bvh")
            BVH.save(file_path, anim, names, frametime=1.0/60.0)
            if remove_fs and 'Style_' in fname:
                glb = Animation.positions_global(anim)
                anim = remove_foot_sliding(anim, glb, cnt_fc)
                file_path = os.path.join(output_dir, fname + "_fixed.bvh")
                BVH.save(file_path, anim, names, frametime=1.0/60.0)

        for key in loss_test_dict.keys():
            loss = loss_test_dict[key]
            if key not in loss_test:
                loss_test[key] = []
            loss_test[key].append(loss)

        log = f'Load epoch [{epochs}], '
        loss_test_avg = dict()
        for key, loss in loss_test.items():
            loss_test_avg[key] = sum(loss) / len(loss)
        log += ' '.join([f'{key:}: [{value:}]' for key, value in loss_test_avg.items()])
        print(log)


if __name__ == '__main__':
    main()