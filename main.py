import faster_rcnn, torch
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from voc_eval import voc_eval


class FasterRCNN():
    def __init__(self, RPN_config, FastRCNN_config, TRAIN_config, TEST_config, DEMO_config, gpu_id):
        self.RPN_config = RPN_config
        self.FastRCNN_config = FastRCNN_config
        self.TRAIN_config = TRAIN_config
        self.TEST_config = TEST_config
        self.DEMO_config = DEMO_config
        self.gpu_id = gpu_id
        
        self.model = faster_rcnn.FasterRCNN(RPN_config, FastRCNN_config, gpu_id)
        self.rpn_cls_losses = []
        self.rpn_loc_losses = []
        self.roi_cls_losses = []
        self.roi_loc_losses = []
        self.best_mAP = 0
        
    def train(self, train_loader, test_loader):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key: params += [{'params': [value], 
                                              'lr': self.TRAIN_config['lr'] * 2, 
                                              'weight_decay': 0}]
                else: params += [{'params': [value], 
                                  'lr': self.TRAIN_config['lr'], 
                                  'weight_decay': self.TRAIN_config['weight_decay']}]
                    
        optimizer = optim.SGD(params, momentum=self.TRAIN_config['momentum'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.TRAIN_config['milestones'])
        
        self.model.train()
        for epoch in range(self.TRAIN_config['epochs']):
            print('Epoch {} Started...'.format(epoch+1))
            for i, (images, labels, bboxs) in enumerate(train_loader):
                rpn_cls_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss, _, _, _ = self.model(images, labels, bboxs)
                
                if roi_loc_loss != None: train_loss = rpn_cls_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss
                else: train_loss = rpn_cls_loss + rpn_loc_loss + roi_cls_loss
                    
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                    
                if ((i+1) % self.TRAIN_config['print_freq'] == 0) and (roi_loc_loss != None):
                    rpn_c, rpn_l = rpn_cls_loss.item(), rpn_loc_loss.item()
                    roi_c, roi_l = roi_cls_loss.item(), roi_loc_loss.item()
                    self.rpn_cls_losses.append(rpn_c); self.rpn_loc_losses.append(rpn_l)
                    self.roi_cls_losses.append(roi_c); self.roi_loc_losses.append(roi_l)
            
            scheduler.step()
            
            if epoch % self.TRAIN_config['epoch_freq'] == 0:
                mAP = self.val(test_loader)
                print('Epoch {} mAP : {:.4f}'.format(epoch+1, 100 * mAP))
                if (mAP > self.best_mAP) and (self.TRAIN_config['save']):
                    self.best_mAP = mAP
                    torch.save(self.model.state_dict(), 
                               self.TRAIN_config['SAVE_PATH'] + 'epoch_{}.pt'.format(str(epoch+1).zfill(3)))
                    print('Saved Best Model')
            print()
                
    def val(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            pred_bboxs, pred_labels, pred_scores = [], [], []
            gt_labels, gt_bboxs = [], []
            
            for images, gt_labels_, gt_bboxs_ in test_loader:
                _, _, _, _, pred_labels_, pred_scores_, pred_detections_ = self.model(images)
                gt_labels.append(gt_labels_.view(-1))
                gt_bboxs.append(gt_bboxs_.view(-1, 4))
                
                pred_labels.append(pred_labels_.view(-1))
                pred_scores.append(pred_scores_.view(-1))
                pred_bboxs.append(pred_detections_.view(-1, 4))
        
        self.model.train()
        mAP = voc_eval(pred_bboxs, pred_labels, pred_scores, gt_bboxs, gt_labels, 
                       self.TEST_config['num_classes'], self.TEST_config['iou_thresh'], self.TEST_config['use_07_metric'])
        return mAP
    
    def demo(self, image_dir):
        self.model.eval()
        with torch.no_grad():
            image = Image.open(image_dir).convert('RGB')
            image = transforms.Resize(size=self.DEMO_config['min_size'])(image)
            image_tensor = transforms.ToTensor()(image)
            image_norm_tensor = transforms.Normalize(mean=self.DEMO_config['mean'], std=self.DEMO_config['std'])(image_tensor)
            
            self.model.FastRCNN.score_thresh = self.DEMO_config['score_thresh']
            _, _, _, _, pred_labels_, pred_scores_, pred_detections_ = self.model(image_norm_tensor[None, :, :, :])
            
        return (image, pred_labels_[0].cpu().numpy(), pred_scores_[0].cpu().numpy(), pred_detections_[0].cpu().numpy())