import torch, torchvision
import rpn, fast_rcnn

from torch import nn


class FasterRCNN(nn.Module):
    def __init__(self, RPN_config, FastRCNN_config, gpu_id):
        super(FasterRCNN, self).__init__()
        torch.cuda.set_device(gpu_id)
        self.gpu = gpu_id

        self.backbone = self.build_backbone(gpu_id)
        self.RPN =  self.build_RPN(RPN_config, gpu_id)
        self.FastRCNN = self.build_FastRCNN(FastRCNN_config, gpu_id)
        
    def build_backbone(self, gpu_id):
        backbone = torchvision.models.vgg16(pretrained=True).features[:30].cuda(gpu_id)
        for i, children in enumerate(backbone.children()):
            for child in children.parameters():
                child.requires_grad = False
            if i == 9: break
        return backbone
        
    def build_FastRCNN(self, FastRCNN_config, gpu_id):
        classifier = list(torchvision.models.vgg16(pretrained=True).classifier)
        classifier = classifier[:2] + classifier[3:5]
        backbone_fc = nn.Sequential(*classifier).cuda(self.gpu)
        
        roi_head = fast_rcnn.RoIHead(FastRCNN_config['output_size'], FastRCNN_config['downsample'], 
                                     backbone_fc, FastRCNN_config['out_channels'], FastRCNN_config['num_classes'], gpu_id)
        FastRCNN = fast_rcnn.FastRCNN(roi_head,
                                      FastRCNN_config['bbox_reg_weights'],
                                      FastRCNN_config['iou_positive_thresh'], 
                                      FastRCNN_config['iou_negative_high'], FastRCNN_config['iou_negative_low'],
                                      FastRCNN_config['batch_size_per_image'], FastRCNN_config['positive_fraction'],
                                      FastRCNN_config['nms_thresh'], FastRCNN_config['score_thresh'],
                                      FastRCNN_config['detections_per_img'])
        return FastRCNN
        
    def build_RPN(self, RPN_config, gpu_id):
        anchor_generator = rpn.AnchorGenerator(RPN_config['anchor_scale'], RPN_config['anchor_aspect_ratio'], 
                                               RPN_config['downsample'], gpu_id)
        rpn_head = rpn.RPNHead(RPN_config['in_channels'], RPN_config['num_anchors'], gpu_id)
        RPN = rpn.RegionProposalNetwork(anchor_generator, rpn_head, 
                                        RPN_config['bbox_reg_weights'], 
                                        RPN_config['iou_positive_thresh'], 
                                        RPN_config['iou_negative_high'], RPN_config['iou_negative_low'],
                                        RPN_config['batch_size_per_image'], RPN_config['positive_fraction'], 
                                        RPN_config['nms_thresh'], RPN_config['top_n_train'], RPN_config['top_n_test'])
        return RPN

    def forward(self, images, gt_labels=None, gt_bboxs=None):
        if self.training: gt_labels, gt_bboxs = gt_labels.cuda(self.gpu), gt_bboxs.cuda(self.gpu)
        images = images.cuda(self.gpu)
        
        features = self.backbone(images)
        proposals, rpn_cls_loss, rpn_loc_loss = self.RPN(images, features, gt_labels, gt_bboxs)
        labels, scores, detections, roi_cls_loss, roi_loc_loss = self.FastRCNN(images, features, proposals.detach(), 
                                                                               gt_labels, gt_bboxs)
        
        return rpn_cls_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss, labels, scores, detections