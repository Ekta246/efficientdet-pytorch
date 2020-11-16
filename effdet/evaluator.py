import torch
import torch.distributed as dist
import abc
import json
from .distributed import synchronize, is_main_process, all_gather_container
from pycocotools.cocoeval import COCOeval
import effdet.evaluation.detection_evaluator as tfm_eval
from vdot import VdotDataset

'''class Evaluator:

    def __init__(self):
        pass

    @abc.abstractmethod
    def add_predictions(self, output, target):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class COCOEvaluator(Evaluator):

    def __init__(self, coco_api, distributed=False):
        super().__init__()
        self.coco_api = coco_api
        self.distributed = distributed
        self.distributed_device = None
        self.img_ids = []
        self.predictions = []

    def reset(self):
        self.img_ids = []
        self.predictions = []

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            #target = all_gather_container(target)
            sample_ids = all_gather_container(target['img_id'])
            if not is_main_process():
                return
        else:
            sample_ids = target['img_id']

        detections = detections.cpu()
        sample_ids = sample_ids.cpu()
        for index, sample in enumerate(detections):
            image_id = int(sample_ids[index])
            for det in sample:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=image_id,
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                self.img_ids.append(image_id)
                self.predictions.append(coco_det)

    def evaluate(self):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            json.dump(self.predictions, open('./temp.json', 'w'), indent=4)
            results = self.coco_api.loadRes('./temp.json')
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = self.img_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = coco_eval.stats[0]  # mAP 0.5-0.95
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset()
        return metric
        class FastMapEvalluator(Evaluator):

    def __init__(self, distributed=False):
        super().__init__()
        self.distributed = distributed
        self.predictions = []

    def add_predictions(self, output, target):
        pass

    def evaluate(self):
        pass'''

#################new_evaluator#####################        

import torch
import torch.distributed as dist
import abc
import json
import logging
import time
import numpy as np

from .distributed import synchronize, is_main_process, all_gather_container
from pycocotools.cocoeval import COCOeval

# FIXME experimenting with speedups for OpenImages eval, it's slow
#import pyximport; py_importer, pyx_importer = pyximport.install(pyimport=True)
import effdet.evaluation.detection_evaluator as tfm_eval
#pyximport.uninstall(py_importer, pyx_importer)

_logger = logging.getLogger(__name__)


__all__ = ['CocoEvaluator', 'PascalEvaluator', 'OpenImagesEvaluator', 'create_evaluator']


class Evaluator:

    def __init__(self, distributed=False, pred_yxyx=False):
        self.distributed = distributed
        self.distributed_device = None
        self.pred_yxyx = pred_yxyx
        self.img_indices = []
        self.predictions = []

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            img_indices = all_gather_container(target['img_id'])
            if not is_main_process():
                return
        else:
            img_indices = target['img_id']

        detections = detections.cpu().numpy()
        img_indices = img_indices.cpu().numpy()
        for img_idx, img_dets in zip(img_indices, detections):
            self.img_indices.append(img_idx)
            self.predictions.append(img_dets)
        
        #print(img_idx)
        #print(img_dets[img_idx][:4])
    def _coco_predictions(self):
        # generate coco-style predictions
        coco_predictions = []
        coco_ids = []
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            img_id = int(self.img_indices[img_idx])
            coco_ids.append(img_id)
            if self.pred_yxyx:
                # to xyxy
                img_dets[:, 0:4] = img_dets[:, [1, 0, 3, 2]]
            # to xywh
            img_dets[:, 2] -= img_dets[:, 0]
            img_dets[:, 3] -= img_dets[:, 1]
            for det in img_dets:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=int(img_id),
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                coco_predictions.append(coco_det)
        return coco_predictions, coco_ids

    @abc.abstractmethod
    def evaluate(self):
        pass

    def save(self, result_file):
        # save results in coco style, override to save in a alternate form
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            json.dump(coco_predictions, open(result_file, 'w'), indent=4)


class CocoEvaluator(Evaluator):

    def __init__(self, dataset, distributed=False, pred_yxyx=False):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self._dataset = dataset.parser
        self.coco_api = dataset.coco

    def reset(self):
        self.img_indices = []
        self.predictions = []

    def evaluate(self):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            json.dump(coco_predictions, open('./temp.json', 'w'), indent=4)
            results = self.coco_api.loadRes('./temp.json')
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = coco_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = coco_eval.stats[0]  # mAP 0.5-0.95
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset()
        return metric


class TfmEvaluator(Evaluator):
    """ Tensorflow Models Evaluator Wrapper """
    def __init__(
            self, dataset, distributed=False, pred_yxyx=False, evaluator_cls=tfm_eval.ObjectDetectionEvaluator):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self._evaluator = evaluator_cls(categories=[{'id': 1, 'name':'drop_inlet'}])
        self._eval_metric_name = self._evaluator._metric_names[0]
        self._dataset = dataset
        self.results = []
        self.det_total=[]
        self.file='./results.json'
    def reset(self):
        self._evaluator.clear()
        self.img_indices = []
        self.predictions = []
        
    def evaluate(self):

        if not self.distributed or dist.get_rank() == 0:
            for img_idx, img_dets in zip(self.img_indices, self.predictions):
                gt = self._dataset.annot_list[img_idx]
                self._evaluator.add_single_ground_truth_image_info(img_idx, gt)
                bbox = img_dets[:, 0:4] if self.pred_yxyx else img_dets[:, [1, 0, 3, 2]]
                det = dict(bbox=bbox, score=img_dets[:, 4], cls=img_dets[:, 5])
                self.results.append(det)
                self._evaluator.add_single_detected_image_info(img_idx, det)
            self.det_total.append(self.results)
            #json.dump(results, open(, 'w'), indent=4)
            #print(self.det_total)   
            metrics = self._evaluator.evaluate()
            _logger.info('Metrics:')
            for k, v in metrics.items():
                _logger.info(f'{k}: {v}')
            map_metric = metrics[self._eval_metric_name]
            if self.distributed:
                dist.broadcast(torch.tensor(map_metric, device=self.distributed_device), 0)
        else:
            map_metric = torch.tensor(0, device=self.distributed_device)
            wait = dist.broadcast(map_metric, 0, async_op=True)
            while not wait.is_completed():
                # wait without spinning the cpu @ 100%, no need for low latency here
                time.sleep(0.5)
            map_metric = map_metric.item()
        self.reset()
        return map_metric


class PascalEvaluator(TfmEvaluator):

    def __init__(self, dataset, distributed=False, pred_yxyx=False):
        super().__init__(
            dataset, distributed=distributed, pred_yxyx=pred_yxyx, evaluator_cls=tfm_eval.PascalDetectionEvaluator)


class OpenImagesEvaluator(TfmEvaluator):

    def __init__(self, dataset, distributed=False, pred_yxyx=False):
        super().__init__(
            dataset, distributed=distributed, pred_yxyx=pred_yxyx, evaluator_cls=tfm_eval.OpenImagesDetectionEvaluator)


def create_evaluator(name, dataset, distributed=False, pred_yxyx=False):
    # FIXME support OpenImages Challenge2019 metric w/ image level label consideration
    if 'coco' in name:
        return CocoEvaluator(dataset, distributed=distributed, pred_yxyx=pred_yxyx)
    elif 'openimages' in name:
        return OpenImagesEvaluator(dataset, distributed=distributed, pred_yxyx=pred_yxyx)
    else:
        return PascalEvaluator(dataset, distributed=distributed, pred_yxyx=pred_yxyx)
