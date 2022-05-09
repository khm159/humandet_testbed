import torch
from model.metaclass import ObjDetector_base
from model.detector.YOLOX.yolox.utils import postprocess, vis
from model.detector.YOLOX.yolox.data.data_augment import ValTransform
from model.detector.YOLOX.yolox.data.datasets import COCO_CLASSES

class Model(ObjDetector_base):
    """
    yolox wrapping class
    developed using yolox 0.3.0 version.
    please modify this wrapper if you want to use other version.

    Created by H. M. Kim (22.05.09)
    """
    def __init__(self, args, **kwargs):
        super().__init__(args)

        self._show_config = args.show_detector_config
        self._show_model_layers = args.show_detector_model_layers
        self._augmentation = ValTransform(legacy=False) 
    
    def upate_config(self,configdict):
        self.config = configdict
    
    def _update_metadata(self,exp):
        """update metadata from yolox exp config file"""
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        print("    - bbox conf threshold : {}".format(self.confthre))
        print("    - bbox nms threshold : {}".format(self.nmsthre))
        print("    - bbox test size : {}".format(self.test_size))
  
    def build_model(self,**kwargs):
        """building yolox model from exp config file"""
        super().build_model(**kwargs)
        from model.detector.YOLOX.yolox.exp.build import get_exp
        exp = get_exp(self._load_config())
        self._update_metadata(exp)
        if self._show_config : print(exp)
        self.model = exp.get_model()
        if self._show_model_layers : print(self.model)
        self.model.to(self.args.device)
        self._load_pretrained()
        self.model.eval()
    
    def _load_pretrained(self):
        """loading pretrained model"""
        ckpt_file = self.args.detector_ckpt
        ckpt = torch.load(ckpt_file, map_location=self.device)
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        print("- Detector Checkpoint Loaded from {}".format(ckpt_file))

    def _load_config(self):
        """loading config file from model name"""
        if self.args.exp_config is not None:
            return 'model/detector/yolox/exps/default/'+\
            self.configdict['model']
        else:
            return 'model/detector/yolox/exps/default/'+\
            self.args.detector_model
    
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
