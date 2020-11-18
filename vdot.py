import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import torch.utils.data as data
from torch.utils import data
from matplotlib.image import imread
from pycocotools.coco import COCO
from effdet.data.parsers import create_parser
class VdotDataset(data.Dataset):
    def __init__(self, image_dir,  ann_file,  transform=None):
        super(VdotDataset, self).__init__
        
        self.transform = transform
        self.image_dir=image_dir
        self.ann_file = ann_file
        self.coco = None
        #ann = open('/home/ekta/AI_current/vdot/vdot/train_annotations/train_annotations.json', 'r')
        ann = open(self.ann_file)
        data_json = json.load(ann)
        self.yxyx = True  
        self.data_json = data_json
        #image_dir = os.listdir('/home/ekta/AI_current/vdot/vdot/train_set')
        image_dir=os.listdir(self.image_dir)
        total_num_images = len(image_dir)
        self.total_num_images = total_num_images
        self.imgs_list, self.annot_list = self.parse_labels(self.data_json)
        
        self._transform = transform
        self.cat_dicts = [{'id': 1, 'name': 'storm_drain', 'id':2, 'name': 'drop_inlet'}]

    def parse_labels(self, ann_file):

        #annot_lists=[]
        filename_labels = []
        frame_boxes =[]
        det_dict = {}
        all_boxes = []
        
        '''for i,v in self.data_json.items():
            v=np.array(v['assets'], dtype=np.float32)
            #v=v[:,[1,0,3,2]]
            det_dict = {'bbox' : np.concatenate([v['assets'][]]), 'cls' : , 'img_size': (800, 600)}
            frame_boxes.append(det_dict)  '''  
        #all_boxes.append(frame_boxes)
        cls_ids={'0':'background', '1': 'storm_drain', '2':'drop_inlet' }
        cls=[]
        bboxes=[]
        classes=[]
        for k,v in self.data_json.items():
            cls.append(v['assets'])
            filename_labels.append(v['filename'])
        for i in range(len(cls)):
            for j,l in cls[i].items():
                bbox=l
                #yxyx
                [ymin, xmin, ymax, xmax]= [bbox[1], bbox[0], bbox[3], bbox[2]]
                #xyxy
                #[xmin, ymin, xmax, ymax]= [bbox[0], bbox[1], bbox[2], bbox[3]]
                # the model requires in yxyx
                bbox=[ymin, xmin, ymax, xmax]
                #bbox=[xmin,ymin,xmax,ymax]
                #if j=='storm_drain':
                    #clss=1
                    #bbox=np.array([v], dtype=np.float32)
                if j=='drop_inlet':
                    clss=1
                    #bbox=np.array([v], dtype=np.int64)
                bboxes.append(bbox)
                classes.append(clss)
            det_dict = {'bbox' :np.array(bboxes, dtype=np.float32), 'cls':np.array(classes, dtype=np.int64) , 'img_size': (512, 512)}
            bboxes = []
            classes = []
            frame_boxes.append(det_dict)
        '''for i in self.data_json.values():
            if i['filename'] in self.image_dir:
                filename_labels.append(i['filename'])
                bboxes= i['assets']['drop_inlet']
                [ymin, xmin, ymax, xmax]= [bboxes[1], bboxes[0], bboxes[3], bboxes[2]]
                bboxes=[ymin, xmin, ymax, xmax]
                det_dict = {'bbox' :np.array(bboxes, dtype=np.float32), 'cls':np.array([1], dtype=np.int64) , 'img_size': (800, 600)}
            frame_boxes.append(det_dict)'''  
        return filename_labels, frame_boxes

    
    def __len__(self):
       return self.total_num_images
    
    def __getitem__(self, index):
        self.image_name = self.imgs_list[index]
        labels = self.annot_list[index]
        labels['img_id'] = int(index)
        #labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        #img = imread(os.path.join('/home/ekta/AI_current/vdot/vdot/train_set', self.image_name))
        img= Image.open(os.path.join(self.image_dir, self.image_name)).convert('RGB')
        '''img = imread(os.path.join(self.image_dir, self.image_name))
        width = 512
        height = 512
        dim = (width, height)
        img = cv2.resize(img, dim) #interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= img.transpose(2,0,1)
        #img /= 255.0
        self.sample= {'img' : img, 'labels' : labels}'''
        if self.transform is not None:
            img, labels = self.transform(img, labels)
            
        return img, labels
        
        

# self.img_info(self.image_dir)

#np.array([800, 600])
    




#'img_size' : img.size 
#'img_scale' : 









