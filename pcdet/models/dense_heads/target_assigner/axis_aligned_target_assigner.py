import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(B, H, W, 1, anchor_num, box_code_size)]
            gt_boxes: (B, M, 8), [x, y, z, dx, dy, dz, heading, class]
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    #* [anchor_total_num,  box_code_size]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    #* selected_classes为符合这个类别的gt的类别
                    selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    #* box_cls_labels是个列表, 列表里面存储每个anchor的对应类别, shape为[1, 200, 176, 2]
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    #* box_reg_targets是个列表, 列表里面存储每个anchor的对应包围框, shape为[1, 200, 176, 2, 7]
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    #* reg_weights是个列表, 列表里面存储每个anchor的对应regression weight, shape为[1, 200, 176, 2]
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                #* target_dict['box_reg_targets']变为[200*176*2*3, 7]
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                #* target_dict['box_cls_labels']变为[200*176*2*3]
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                #* target_dict['reg_weights']变为[200*176*2*3]
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        #* 将多个batch合并在一起, [B, 200*176*2*3, 7]
        bbox_targets = torch.stack(bbox_targets, dim=0)

        #* 将多个batch合并在一起, [B, 200*176*2*3]
        cls_labels = torch.stack(cls_labels, dim=0)
        #* 将多个batch合并在一起, [B, 200*176*2*3]
        reg_weights = torch.stack(reg_weights, dim=0)
        
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        """
        Returns:
            all_targets_dict: 
                box_cls_labels: [B, 200*176*2*3, 7]
                box_reg_targets: [B, 200*176*2*3]
                reg_weights: [B, 200*176*2*3]
        """
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            #* anchor_by_gt_overlap为[anchor_total_num, gt_num], 表示每个anchor和gt的iou
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            #* 存储anchor对应的iou最大的gt框的索引, [anchor_total_num]
            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            #* 存储anchor和对应的iou最大的gt框的iou, [anchor_total_num]
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            #* 存储gt对应的iou最大的anchor的索引, [gt_num]
            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            #* 存储gt和对应的iou最大的anchor的iou, [gt_num]
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            #* 没有对应anchor的gt的掩码, 将这些gt和对应anchor的iou设置为-1
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            #* 找到与gt拥有最大iou的anchor的索引
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            #* 获取这些anchor对应的gt的索引
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            #* 将这些anchor的类别设置成gt的类别
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            #* 和gt的最大iou大于matched_threshold即为positive, 给这些anchor也赋值
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        """
        Returns:
            box_cls_labels: anchor的类别, >0为前景anchor, =0为背景anchor, [anchor_total_num]
            box_reg_targets: anchor的包围框, [anchor_total_num, 7], [xt, yt, zt, dxt, dyt, dzt, rts]
            reg_weights: [anchor_total_num], 回归的权重
        """
        return ret_dict
