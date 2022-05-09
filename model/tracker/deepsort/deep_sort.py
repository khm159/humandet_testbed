import logging
import math

import numpy as np
import torch

from ..builder import TRACKERS
from .base import BaseTracker
# from .deep.feature_extractor import Extractor
# from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.tracker import Tracker

EXTRACTOR = TRACKERS


@TRACKERS.register_module()
class DeepSortTracker(BaseTracker):
    """DeepSortTracker."""

    def __init__(self, cfg):
        super(BaseTracker).__init__()
        self.device = torch.device('cpu')
        if cfg.verbose:
            self.logger = logging.getLogger('root.tracker')
        else:
            self.logger = None
        self.model_path = cfg.Extractor.checkpoint_path
        self.extractor = EXTRACTOR.build(cfg.Extractor)
        self.min_confidence = cfg.min_confidence
        # self.nms_max_overlap = cfg.nms_max_overlap
        self.target_classes = cfg.target_classes
        self.use_yolo_features = cfg.use_yolo_features
        metric = NearestNeighborDistanceMetric('cosine', cfg.max_dist,
                                               cfg.nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=cfg.max_iou_distance,
            max_age=cfg.max_age,
            n_init=cfg.n_init)

    def to(self, device):
        self.device = torch.device(device=device)
        self.extractor.to(self.device)

    def _update(self, data, **kwargs):
        """Update id tracker using deep features.

        Args:
            data (dict): pipleline data dict
            Required key: 'ori_img', 'det_result'
                        (optional) 'im_crops', 'features'.
            Modifies key: 'track_ids', 'bbox_xyxy'.
        """
        self.height, self.width = data['ori_img'][0].shape[:2]
        # d_object = data['det_result']
        # Filter by class
        objects = data['objects']
        batch_track_ids = []
        for idx, det in enumerate(objects):
            if self.target_classes is not None:
                det = det[(det[:, 5:6] == torch.tensor(
                    self.target_classes)).any(1)]
            # Filter by conf
            det = det[(det[:, 4:5] > self.min_confidence).any(1)]
            # Filter by vaild bbox xyxy
            det = det[(det[:, 2:3] > det[:, 0:1]).any(1)]
            det = det[(det[:, 3:4] > det[:, 1:2]).any(1)]
            bbox_xyxy = det[:, :4]
            confidences = det[:, 4]
            bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)

            # Extrack deep feature
            if self.use_yolo_features:
                # TODO torchvision roi_align. YOLO feature hook.
                features = self._get_yolo_features(bbox_xyxy, data['features'])
            if 'im_crops' in data:
                # TODO preprocssed in yolo
                features = self.extractor(data['im_crops'])
            else:
                features = self._get_features(bbox_xyxy, data['ori_img'][idx])

            # generate detections
            detections = [
                Detection(tlwh, conf, feat)
                for tlwh, conf, feat in zip(bbox_tlwh, confidences, features)
            ]

            self.tracker.predict()
            self.tracker.update(detections)

            # output bbox identities
            track_ids = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                track_ids.append(
                    np.array([x1, y1, x2, y2, track_id], dtype=np.int))
            if len(track_ids) > 0:
                track_ids = np.stack(track_ids, axis=0)
            batch_track_ids.append(track_ids)
        data['batch_track_ids'] = batch_track_ids
        return data

    def overlap_id_assign(self, track_id, new_id):
        self.tracker.overwrite_id(track_id, new_id)

    def change_min_conf(self, conf):
        self.min_confidence = conf

    @staticmethod
    def vaild_check(bbox_xyxy):
        vaild_idx = np.where((bbox_xyxy[:, 2] - bbox_xyxy[:, 0] > 0)
                             & (bbox_xyxy[:, 3] - bbox_xyxy[:, 1] > 0))[0]
        return vaild_idx

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_xywh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_xywh = bbox_xyxy.clone()
        bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
        bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
        bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_xywh

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_xywh_and_tlwh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_xywh = bbox_xyxy.copy()
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_xywh = bbox_xyxy.clone()
            bbox_tlwh = bbox_xyxy.clone()
        bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
        bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
        bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        bbox_tlwh[:, 2] = bbox_xywh[:, 2]
        bbox_tlwh[:, 3] = bbox_xywh[:, 3]
        return bbox_xywh, bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_xyxy, ori_img):
        # Modify by hobeom, bbox_xywh to bbox_xyxy
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _get_yolo_features(self, bbox_xyxy, yolo_features):
        features = []
        for box in bbox_xyxy:
            feature = self._roi_align(box, yolo_features)
            features.append(feature)
        return np.array(features)

    def _roi_align(self, box, feature):
        # bbox with format [x1, y1, x2, y2]
        # bbox width, height is (1280, 720)
        # feature tensor shape is (1, 3, 12, 20, 85)
        # do roi align for feature
        x1, y1, x2, y2 = box
        # rescale bbox to 0~1
        # rescale bbox to feature map size
        x1 = math.floor(x1 / 1280 * feature.shape[3])
        x2 = math.ceil(x2 / 1280 * feature.shape[3])
        y1 = math.floor(y1 / 720 * feature.shape[2])
        y2 = math.ceil(y2 / 720 * feature.shape[2])
        if x1 == x2 or y1 == y2:
            logging.error(f'{x1,x2,y1,y2} error with extrack yolo feature')
        # feature to numpy array
        feature = feature.numpy()
        # get roi
        roi = feature[:, :, y1:y2, x1:x2, :]
        # get roi mean
        roi_mean = np.mean(roi, axis=(2, 3))
        return roi_mean
