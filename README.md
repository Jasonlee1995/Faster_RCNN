# Faster R-CNN Implementation with Pytorch
- Unofficial implementation of the paper *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```
- Using Single GPU


## 1. Implementation Details
- augmentation.py : augmentation considering images and bounding boxes
- dataset.py : VOC dataset using torchvision
- fast_rcnn.py : Fast R-CNN implementation
- faster_rcnn.py : Faster R-CNN implementation
- main.py : train, test, inference of Faster R-CNN
- rpn.py : Region Proposal Network implementation
- utils.py : class includes decode/encode bounding boxes coordinates, balancing samples, match predicts with ground truth
- train and inference.ipynb : install library, download dataset, preprocessing, train and result
- sample.png : sample image for visualize


## 2. Result Comparison on PASCAL VOC 2007
|Source|Score|Detail|
|:-:|:-:|:-|
|Paper|69.9|VOC 2007 trainval|
|Paper|73.2|VOC 2007 trainval + VOC 2012 trainval|
|Paper|78.8|VOC 2007 trainval + VOC 2012 trainval + COCO|
|Current Repo|67.5454|VOC 2007 trainval|


## 3. Reference
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [[paper](https://arxiv.org/pdf/1506.01497.pdf)] [[code](https://github.com/ShaoqingRen/faster_rcnn)]
