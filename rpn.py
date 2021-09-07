import torch, torchvision, utils

from torch import nn
from torch.nn import functional as F


class AnchorGenerator(nn.Module):
    """
    Generate anchors in images using anchor_scale and anchor_aspect_ratio.
    Assume aspect ratio = height / width.
    """
    def __init__(self, anchor_scale=(128, 256, 512), anchor_aspect_ratio=(0.5, 1.0, 2.0), downsample=16, gpu_id=0):
        super(AnchorGenerator, self).__init__()
        torch.cuda.set_device(gpu_id)
        self.gpu = gpu_id
        
        self.anchor_scale = anchor_scale
        self.anchor_aspect_ratio = anchor_aspect_ratio
        self.downsample = downsample
        self.base_anchors = self.generate_base_anchors(anchor_scale, anchor_aspect_ratio).cuda(self.gpu)
        
    def forward(self, features):
        images_anchors = self.generate_images_anchors(features, self.downsample)
        return images_anchors

    def generate_base_anchors(self, anchor_scale, anchor_aspect_ratio):
        anchor_scale, anchor_aspect_ratio = torch.FloatTensor(anchor_scale), torch.FloatTensor(anchor_aspect_ratio)
        anchor_h_ratio = torch.sqrt(anchor_aspect_ratio)
        anchor_w_ratio = 1 / anchor_h_ratio
        
        anchor_ws = (anchor_w_ratio[:, None] * anchor_scale[None, :]).view(-1)
        anchor_hs = (anchor_h_ratio[:, None] * anchor_scale[None, :]).view(-1)
        
        base_anchors = torch.stack([-anchor_ws, -anchor_hs, anchor_ws, anchor_hs], dim=1) / 2
        return base_anchors

    def generate_images_anchors(self, features, downsample):
        features_size = [feature.shape[-2:] for feature in features]
        
        images_anchors = []
        for f_h, f_w in features_size:
            grid_y, grid_x = torch.meshgrid(torch.arange(0, f_h).cuda(self.gpu) * downsample, 
                                            torch.arange(0, f_w).cuda(self.gpu) * downsample)
            grid_y, grid_x = grid_y.reshape(-1), grid_x.reshape(-1)
            grid_xy = torch.stack((grid_x, grid_y, grid_x, grid_y), dim=1)
            image_anchors = (grid_xy.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4)).reshape(-1, 4)
            images_anchors.append(image_anchors)
            
        return torch.stack(images_anchors, dim=0)


class RPNHead(nn.Module):
    """
    Classification and regression for given features.
    """
    def __init__(self, in_channels, num_anchors, gpu_id):
        super(RPNHead, self).__init__()
        torch.cuda.set_device(gpu_id)
        self.gpu = gpu_id
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), 
                                  nn.ReLU()).cuda(self.gpu)
        self.classification = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1).cuda(self.gpu)
        self.bbox_regressor = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1, stride=1).cuda(self.gpu)
        self._initialize_weights()

    def forward(self, features):
        features = self.conv(features)
        objectness = self.classification(features)
        pred_bbox_deltas = self.bbox_regressor(features)
        return objectness, pred_bbox_deltas
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, rpn_head,
                 bbox_reg_weights, 
                 iou_positive_thresh, iou_negative_high, iou_negative_low,
                 batch_size_per_image, positive_fraction,
                 nms_thresh, top_n_train, top_n_test):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        
        self.box_coder = utils.BoxCoder(bbox_reg_weights)
        self.proposal_matcher = utils.Matcher(iou_positive_thresh, iou_negative_high, iou_negative_low, low_quality_match=True)
        self.sampler = utils.Balanced_Sampler(batch_size_per_image, positive_fraction)
        
        self.nms_thresh = nms_thresh
        self.top_n_train = top_n_train
        self.top_n_test = top_n_test
        self.min_size = 0.01

    def assign_gt_to_anchors(self, anchors, gt_labels, gt_bboxs):
        labels, matched_gt_bboxs = [], []
        for anchors_per_img, gt_bboxs_per_img in zip(anchors, gt_bboxs):
            match_quality_matrix = torchvision.ops.box_iou(gt_bboxs_per_img, anchors_per_img)
            matched_idxs_per_img = self.proposal_matcher(match_quality_matrix)
            
            matched_gt_bboxs_per_img = gt_bboxs_per_img[torch.clamp(matched_idxs_per_img, min=0)]
            labels_per_img = (matched_idxs_per_img >= 0).float()

            # Negative
            negative_idxs = matched_idxs_per_img == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_img[negative_idxs] = 0.0

            # Between
            between_idxs = matched_idxs_per_img == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_img[between_idxs] = -1.0

            labels.append(labels_per_img)
            matched_gt_bboxs.append(matched_gt_bboxs_per_img)
        
        labels, matched_gt_bboxs = torch.stack(labels, dim=0), torch.stack(matched_gt_bboxs, dim=0)
        return labels, matched_gt_bboxs
    
    def calculate_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_positive_masks, sampled_negative_masks = self.sampler(labels)
        sampled_masks = sampled_positive_masks | sampled_negative_masks

        sampled_objectness, sampled_labels = objectness[sampled_masks], labels[sampled_masks]
        sampled_deltas, sampled_regression_targets = (pred_bbox_deltas[sampled_positive_masks], 
                                                      regression_targets[sampled_positive_masks])

        rpn_cls_loss = F.binary_cross_entropy_with_logits(sampled_objectness, sampled_labels)
        rpn_loc_loss = F.smooth_l1_loss(sampled_deltas, sampled_regression_targets, beta=1/9)
        
        return rpn_cls_loss, rpn_loc_loss
    
    def convert(self, bbox_cls, bbox_regression):
        """
        Convert convolution output shape (N, A*?, ?, H, W) to shape (N, -1, ?).
        """
        N, Ax4, H, W = bbox_regression.shape
        A = Ax4 // 4
        
        bbox_cls, bbox_regression = bbox_cls.view(N, A, 1, H, W), bbox_regression.view(N, A, 4, H, W)
        bbox_cls, bbox_regression = bbox_cls.permute(0, 3, 4, 1, 2), bbox_regression.permute(0, 3, 4, 1, 2)
        bbox_cls, bbox_regression = bbox_cls.reshape(N, -1), bbox_regression.reshape(N, -1, 4)
        return bbox_cls, bbox_regression
    
    def filter_proposals(self, images, objectness, proposals):
        objectness_prob = torch.sigmoid(objectness)
        filtered_scores, filtered_proposals = [], []
        for img, objectness_prob_per_img, proposals_per_img in zip(images, objectness_prob, proposals):
            # clip to image size
            proposals_per_img = torchvision.ops.clip_boxes_to_image(proposals_per_img, tuple(img.shape[-2:]))

            # remove small proposals
            keep_idx = torchvision.ops.remove_small_boxes(proposals_per_img, self.min_size)
            objectness_prob_per_img, proposals_per_img = objectness_prob_per_img[keep_idx], proposals_per_img[keep_idx]

            # NMS & top-n
            keep_idx = torchvision.ops.nms(proposals_per_img, objectness_prob_per_img, self.nms_thresh)
            keep_idx = keep_idx[:self.top_n()]
            objectness_prob_per_img, proposals_per_img = objectness_prob_per_img[keep_idx], proposals_per_img[keep_idx]
            
            filtered_scores.append(objectness_prob_per_img)
            filtered_proposals.append(proposals_per_img)
        return torch.stack(filtered_scores, dim=0), torch.stack(filtered_proposals, dim=0)

    def forward(self, images, features, gt_labels=None, gt_bboxs=None):
        anchors = self.anchor_generator(features.detach())
        objectness, pred_bbox_deltas = self.rpn_head(features)
        objectness, pred_bbox_deltas = self.convert(objectness, pred_bbox_deltas)
        
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        filtered_scores, filtered_proposals = self.filter_proposals(images, objectness.detach(), proposals)
        
        rpn_cls_loss, rpn_loc_loss = None, None
        if self.training:
            labels, matched_gt_bboxs = self.assign_gt_to_anchors(anchors, gt_labels, gt_bboxs)
            regression_targets = self.box_coder.encode(matched_gt_bboxs, anchors)
            rpn_cls_loss, rpn_loc_loss = self.calculate_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            
        return filtered_proposals, rpn_cls_loss, rpn_loc_loss
    
    def top_n(self):
        if self.training: return self.top_n_train
        return self.top_n_test