import torch
from model.metaclass import ObjDetector_base
from model.detector.YOLOX.yolox.utils import postprocess, vis
from model.detector.YOLOX.yolox.data.data_augment import ValTransform
from model.detector.YOLOX.yolox.data.datasets import COCO_CLASSES
import cv2
import numpy as np

class Model(ObjDetector_base):
    """
    yolox wrapping class
    developed using yolox 0.3.0 version.
    please modify this wrapper if you want to use other version.

    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        super().__init__(args)
        self._show_config = args.show_detector_config
        self._show_model_layers = args.show_detector_model_layers
        self._build_model(**kwargs) 
        self._augmentation = ValTransform(legacy=False) 
        # True if you use legacy checkpoint. not support in this version.
        self.compute_color_for_labels = lambda x: [
            int((p * (x**2 - x + 1)) % 255) for p in self.palette
        ]  # noqa
        self.palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    
    def _update_metadat(self,exp):
        """update metadata from yolox exp config file"""
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
    
    def _build_model(self,**kwargs):
        """building yolox model from exp config file"""
        super()._build_model(**kwargs)
        from model.detector.YOLOX.yolox.exp.build import get_exp
        exp = get_exp(self._load_config())
        self._update_metadat(exp)
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
        self.img_info = {"id": 0}
        height, width = img.shape[:2]
        self.img_info["height"] = height
        self.img_info["width"] = width
        self.img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        self.img_info["ratio"] = ratio

        img, _ = self._augmentation(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        return img
    
    def rescale_bbox(self, bbox):       
        
        box_num = bbox.shape[0]
        ratio = torch.tensor([self.img_info['ratio']]*box_num).to(self.device)
        print(ratio.shape, bbox[:,0].shape)
        bbox[:, 0] /= ratio
        bbox[:, 1] /= ratio
        bbox[:, 2] /= ratio
        bbox[:, 3] /= ratio
        return bbox

    def draw_bbox(self,img,bboxes,identities=None,offset=(0, 0)):
        """
        drawing bbox on image 
        args :
            img : cv2 array (H,W,3)
            bboxes : list or array of the bbox_xyxy [(x1,y1,x2,y2),...]
            identities : list or array of the identities
            offset : offset for bbox
        """
        # Drwa Boxes
        for i, bbox in enumerate(bboxes):
            # Every IDs
            x1, y1, x2, y2 = [int(i) for i in bbox]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
           
            # Draw ID text, Bbox
            if len(identities) == 0 and len(bboxes) !=0:
                label = "unknonw"
                color = self.compute_color_for_labels(0)
            else:
                id = int(identities[i]) if len(identities) != 0 else 0
                # Bbox ID 색상 계산
                color = self.compute_color_for_labels(id)
                # Bbox 텍스트 생성
                label = '{}'.format(id)

            # 화면 그리기
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(img, (x1, y1),
                (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color,-1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN, 3, [196, 196, 196], 1)

        return img

    def run(self,img):
        """inference code for images"""
        ## batch-inference will be updated soon.
        img = self._preprocess(img)
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
            if outputs[0] is not None and len(outputs) != 0:
                ## TODO: Make Efficient.
                if self.args.target_classes != None:
                    cls_name = outputs[0][:, 6]
                    remain_inds = []
                    for i, elem in enumerate(cls_name):
                        if elem.item() in self.args.target_classes:
                            remain_inds.append(i)
                    outputs[0] = outputs[0][remain_inds]
            else:
                return [torch.zeros([0,7]).to(self.device)]
        #rescale bbox
        outputs[0][:,0:4] = self.rescale_bbox(outputs[0][:,0:4])
        return outputs
