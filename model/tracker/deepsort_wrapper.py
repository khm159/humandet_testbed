import torch
from model.metaclass import ObjTracker_base

class Model(ObjTracker_base):
    """
    deepsort wrapping class

    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        super().__init__(args)
        self._show_config = args.show_tracker_config
        self._show_model_layers = args.show_tracker_model_layers
        self._build_model(**kwargs) 
    
    def _build_model(self,**kwargs):
        """building deepsort"""
        super()._build_pipeline(**kwargs)
        from model.tracker.deepsort.sort.tracker import Tracker
        from model.tracker.deepsort.sort.nn_matching import NearestNeighborDistanceMetric
        cfg = self._load_cfg()
        metric = NearestNeighborDistanceMetric(
            'cosine', cfg.max_dist,cfg.nn_budget
            )
        self.tracker = Tracker(
            metric,
            max_iou_distance=cfg.max_iou_distance,
            max_age=cfg.max_age,
            n_init=cfg.n_init)
            
    def _load_cfg(self):

    def _load_pretrained(self):
        """loading pretrained model"""
        ckpt_file = self.args.detector_ckpt
        ckpt = torch.load(ckpt_file, map_location=self.device)
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        print("- Detector Checkpoint Loaded from {}".format(ckpt_file))

    def _load_config(self):
        """loading config file from model name"""
        

    def change_confthre(self, confthre):
        """change confthresh if not use default setting"""
        self.confthre = confthre

    def change_nmsthre(self, nmsthre):
        """change nms thresh if not use default setting"""
        self.nmsthre =nmsthre

    def _preprocess(self, img):
        """yolox pre-process code for images"""
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self._augmentation(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        return img, img_info
    
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, COCO_CLASSES)
        return vis_res

    def run(self,img):
        """inference code for images"""
        ## batch-inference will be updated soon.
        img, img_info = self._preprocess(img)
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, 
                self.num_classes, 
                self.confthre,
                self.nmsthre, 
                class_agnostic=True
            )
            ## leave only target class
            if outputs[0] is not None: ## TODO: Make Efficient.
                if self.args.target_classes != None:
                    cls_name = outputs[0][:, 6]
                    remain_inds = []
                    for i, elem in enumerate(cls_name):
                        if elem.item() in self.args.target_classes:
                            remain_inds.append(i)
                    outputs[0] = outputs[0][remain_inds]
        return outputs, img_info
