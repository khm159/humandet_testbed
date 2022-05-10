import torch
import numpy as np
from model.metaclass import ObjTracker_base
from model.tracker.deepsort.sort.detection import Detection

class Model(ObjTracker_base):
    """
    deepsort wrapping class

    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        super().__init__(args)
        self._build_model(**kwargs) 
    
    def _update_metadat(self,exp):
        """update metadata from yolox exp config file"""
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        print("    - bbox conf threshold : {}".format(self.confthre))
        print("    - bbox nms threshold : {}".format(self.nmsthre))
        print("    - bbox test size : {}".format(self.test_size))
    
    def _build_model(self,**kwargs):
        """building yolox model from exp config file"""
        super()._build_model(**kwargs)
        from model.tracker.deepsort.sort.tracker import Tracker
        from model.tracker.deepsort.sort.nn_matching import NearestNeighborDistanceMetric

        # metric, matching_threshold, budget=None
        metric = NearestNeighborDistanceMetric(
            metric = 'cosine', 
            matching_threshold = self.args.tracker_max_dist,
            budget = self.args.tracker_budget
            )
        self.tracker = Tracker(
            metric,
            max_iou_distance=self.args.tracker_max_iou_distance,
            max_age=self.args.tracker_max_age,
            n_init=self.args.tracker_n_init
        )
        if self.args.extractor is not None:
            if self.args.extractor == 'default':
                from model.tracker.deepsort.extractors.feature_extractor import DefaultExtractor
            else: 
                NotImplementedError
            self.extractor = DefaultExtractor(
                checkpoint_path = self.args.exctractor_ckpt
            )
            self.extractor.to(self.args.device)
    
    def change_confthre(self, confthre):
        """change confthresh if not use default setting"""
        self.confthre = confthre

    def change_nmsthre(self, nmsthre):
        """change nms thresh if not use default setting"""
        self.nmsthre =nmsthre
    
    def _img_preproc(self, img):
        """input law np frames"""
        img = torch.from_numpy(img)  # c, h, w
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def _get_features(self, bbox_xyxy, ori_img):
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
    

    def _update(self, data, **kwargs):
        """Update id tracker using deep features.

        Args:
            data (dict): pipleline data dict
            Required key: 'ori_img', 'det_result'
                        (optional) 'im_crops', 'features'.
            Modifies key: 'track_ids', 'bbox_xyxy'.
            TODO: Batch support
        """
        self.height, self.width = data['img'][0].shape[:2]
        
        # Filter by class
        objects = data['objects']
        bbox_xyxy = []
        confidences = []
        track_ids = []

        objects = [x.cpu().numpy() for x in objects]
        objects = np.asarray(objects)


        if self.args.detector =='yolox':
            # Filter by conf 
            objects = objects[objects[:,4]*objects[:,5] > self.min_confidence]
            # Filter by vaild bbox xyxy
            objects = objects[objects[:,2] > objects[:,0]] 
            objects = objects[objects[:,3] > objects[:,1]]
            confidences = objects[:,4]*objects[:,5]
            bbox_xyxy = objects[:,0:4]
            bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
            bbox_tlwh = torch.from_numpy(bbox_tlwh).float().to(self.args.device)
            
        # Extrack deep feature
        if 'im_crops' in data:
            # TODO preprocssed in yolo
            features = self.extractor(data['im_crops'])
        else:
            features = self._get_features(bbox_xyxy, data['ori_img'])
    
        # generate detections
        detections = [
            Detection(tlwh, conf, feat)
            for tlwh, conf, feat in zip(bbox_tlwh, confidences, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_ids.append(
                np.array([x1, y1, x2, y2, track_id], dtype=np.int))

        return np.asarray(track_ids)
    
    @staticmethod
    def _xywh_to_xyxy(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_xyxy = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_xyxy = bbox_xywh.clone()
        bbox_xyxy[:,2] = bbox_xywh[:,0] + bbox_xywh[:,2]
        bbox_xyxy[:,3] = bbox_xywh[:,1] + bbox_xywh[:,3]
        return bbox_xyxy

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
