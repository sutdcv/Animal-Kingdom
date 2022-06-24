# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import logging
import os
import json_tricks as json
from collections import OrderedDict
import json
import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class AnimalKingdomDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        '''
        0: Head_Mid_Top
        1: Eye_Left 
        2: Eye_Right 
        3: Mouth_Front_Top 
        4: Mouth_Back_Left
        5: Mouth_Back_Right
        6: Mouth_Front_Bottom
        7: Shoulder_Left
        8: Shoulder_Right
        9: Elbow_Left
        10: Elbow_Right
        11: Wrist_Left
        12: Wrist_Right
        13: Torso_Mid_Back
        14: Hip_Left
        15: Hip_Right
        16: Knee_Left
        17: Knee_Right
        18: Ankle_Left 
        19: Ankle_Right
        20: Tail_Top_Back
        21: Tail_Mid_Back
        22: Tail_End_Back
        '''
        ### Changes below ###
        self.num_joints = 23
        self.flip_pairs = [[1, 2], [4, 5], [7, 8], [9, 10], [11, 12], [14,15], [16,17], [18,19]]
        self.parent_ids = [3,3,3,6,6,6,13,13,13,7,8,9,10,13,13,13,14,15,16,17,12,20,21]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        self.lower_body_ids = (14, 15, 16, 17, 18, 19, 20, 21, 22)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # Create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json' 
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                # c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
#             # we should first convert to 0-based index
#             c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2]
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # Convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

#         if 'test' in cfg.DATASET.TEST_SET:
#             return {'Null': 0.0}, 0.0

        SC_BIAS = 1
        threshold = 0.1

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               '{}.json' #gt_{}.json'
                               .format(cfg.DATASET.TEST_SET)
                               )
        
    
        with open(gt_file) as f:
            gt_dict = json.load(f)
        
        ### Changes below ###

        # dataset_joints = gt_dict['dataset_joints']
        # jnt_visible = [v for k, v in gt_dict['joints_vis'].items()]
        # pos_gt_src = [v for k, v in gt_dict['joints'].items()]
        # scale = [v for k, v in gt_dict['scale'].items()]

        dataset_joints = [
            [
                "Head_Mid_Top"
            ],
            [
                "Eye_Left"
            ],
            [
                "Eye_Right"
            ],
            [
                "Mouth_Front_Top"
            ],
            [
                "Mouth_Back_Left"
            ],
            [
                "Mouth_Back_Right"
            ],
            [
                "Mouth_Front_Bottom"
            ],
            [
                "Shoulder_Left"
            ],
            [
                "Shoulder_Right"
            ],
            [
                "Elbow_Left"
            ],
            [
                "Elbow_Right"
            ],
            [
                "Wrist_Left"
            ],
            [
                "Wrist_Right"
            ],
            [
                "Torso_Mid_Back"
            ],
            [
                "Hip_Left"
            ],
            [
                "Hip_Right"
            ],
            [
                "Knee_Left"
            ],
            [
                "Knee_Right"
            ],
            [
                "Ankle_Left"
            ],
            [
                "Ankle_Right"
            ],
            [
                "Tail_Top_Back"
            ],
            [
                "Tail_Mid_Back"
            ],
            [
                "Tail_End_Back"
            ]
        ]

        jnt_visible = [x['joints_vis'] for x in gt_dict]
        pos_gt_src = [x['joints'] for x in gt_dict]
        scale = [x['scale'] for x in gt_dict]

        scale=np.array(scale)
        scale=scale*200*math.sqrt(2)

        jnt_visible=np.transpose(jnt_visible, [1, 0])
        pos_pred_src = np.transpose(preds, [1, 2, 0])
        pos_gt_src=np.transpose(pos_gt_src, [1, 2, 0])
        dataset_joints=np.array(dataset_joints)
        head = np.where(dataset_joints == 'Head_Mid_Top')[0][0]
        lsho = np.where(dataset_joints == 'Shoulder_Left')[0][0]
        lelb = np.where(dataset_joints == 'Elbow_Left')[0][0]
        lwri = np.where(dataset_joints == 'Wrist_Left')[0][0]
        lhip = np.where(dataset_joints == 'Hip_Left')[0][0]
        lkne = np.where(dataset_joints == 'Knee_Left')[0][0]
        lank = np.where(dataset_joints == 'Ankle_Left')[0][0]

        rsho = np.where(dataset_joints == 'Shoulder_Right')[0][0]
        relb = np.where(dataset_joints == 'Elbow_Right')[0][0]
        rwri = np.where(dataset_joints == 'Wrist_Right')[0][0]
        rhip = np.where(dataset_joints == 'Hip_Right')[0][0]
        rkne = np.where(dataset_joints == 'Knee_Right')[0][0]
        rank = np.where(dataset_joints == 'Ankle_Right')[0][0]
        
        tmouth = np.where(dataset_joints == 'Mouth_Front_Top')[0][0]
        lmouth = np.where(dataset_joints == 'Mouth_Back_Left')[0][0]
        rmouth = np.where(dataset_joints == 'Mouth_Back_Right')[0][0]
        bmouth = np.where(dataset_joints == 'Mouth_Front_Bottom')[0][0]
        ttail = np.where(dataset_joints == 'Tail_Top_Back')[0][0]
        mtail = np.where(dataset_joints == 'Tail_Mid_Back')[0][0]
        btail = np.where(dataset_joints == 'Tail_End_Back')[0][0]
        
        
        uv_error = pos_pred_src - pos_gt_src

        uv_err = np.linalg.norm(uv_error, axis=1)

#         headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
#         headsizes = np.linalg.norm(headsizes, axis=0)
        scale *= SC_BIAS
        headsizes=scale


        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)

        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 23))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

#         PCKh = np.ma.array(PCKh, mask=False)
#         PCKh.mask[21:22] = True

#         jnt_count = np.ma.array(jnt_count, mask=False)
#         jnt_count.mask[21:22] = True

        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mouth', 0.25 * (PCKh[tmouth] + PCKh[lmouth]+ PCKh[rmouth]+ PCKh[bmouth])),
            ('Tail', (PCKh[ttail] + PCKh[mtail]+ PCKh[btail])/3),
            ('Mean', np.sum(PCKh * jnt_ratio))
#             ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)
        return name_value, name_value['Mean']
