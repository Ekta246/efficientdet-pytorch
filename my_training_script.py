from effdet import create_model
import torch.nn


class ObjectDetectionModel(torch.nn.Module):
    def __init__(self, num_classes, model_name, checkpoint_path=''):
        super(ObjectDetectionModel, self).__init__()
        self.model = create_model(
            model_name,
            bench_task='train',
            num_classes=num_classes,
            pretrained=True,
            pretrained_backbone=True,
            bench_labeler=True,
            checkpoint_path = checkpoint_path,
        )
    def forward(self, x, y):
        return self.model(x, y)
    def freezeBackbone(self):
        for param in self.model.model.backbone.parameters():
            param.requires_grad = False
    def freezeFPN(self):
        for param in self.model.model.fpn.parameters():
            param.requires_grad = False
    def unfreezeBackbone(self):
        for param in self.model.model.backbone.parameters():
            param.requires_grad = True
    def unfreezeFPN(self):
        for param in self.model.model.fpn.parameters():
            param.requires_grad = True
model_name = 'efficientdet_d0'
model = ObjectDetectionModel(num_classes, model_name)
...
#train heads only
model.freezeBackbone()
model.freezeFPN()
for i, (images, bboxes, classes) in enumerate(train_loader):
    targets = dict()
    targets['bbox'] = [bbox.cuda(non_blocking=True).float() for bbox in bboxes] # [y_min, x_min, y_max, x_max]
    targets['cls'] = [cls.cuda(non_blocking=True).float() for cls in classes]
    images = images.cuda(non_blocking=True) # 640x640 for efficientdet_d1. See image_size in https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py
    optimizer.zero_grad()
    # compute output
    outputs = model(images, targets)
    loss = outputs['loss']
    class_loss = outputs['class_loss']
    bbox_loss = outputs['box_loss']
    ...
    loss.backward()
    optimizer.step()
# finetuning step 1
'''model.unfreezeFPN()
# same as train loop above

# finetuning step 2
model.unfreezeBackbone()'''
# same as train loop above
...
# validate
for i, (images, bboxes, classes) in enumerate(val_loader):
    targets = dict()
    targets['bbox'] = [bbox.cuda(non_blocking=True).float() for bbox in bboxes]
    targets['cls'] = [cls.cuda(non_blocking=True).float() for cls in classes]
    # for images 640x640 only. Change it if you are using another image size and scale.
    targets['img_scale'] = torch.ones(images.size(0)).cuda(non_blocking=True).float()
    targets['img_size'] = torch.ones(images.size(0), 2).mul(640).cuda(non_blocking=True).float() # 640 for efficientdet_d1. See image_size in https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py
    images = images.cuda(non_blocking=True)
    # compute output
    outputs = model(images, targets)
    loss = outputs['loss']
    class_loss = outputs['class_loss']
    bbox_loss = outputs['box_loss']
    detections = outputs['detections'].cpu() # xmin, ymin, xmax, ymax, score, class
    ...
    # itarate over images
    for k in range(detections.size(0)):
        # itarate over image bboxes
        for j in range(detections[k].size(0)):
            det = detections[k][j].tolist()
            x_min, y_min, x_max, y_max, score, class_id = det
