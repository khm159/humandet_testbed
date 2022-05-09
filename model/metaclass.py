import cv2
from abc import ABC, abstractmethod

class ObjDetector_base(ABC):
    """
    Object Detector Abstract Class
    Initialize the basic object detector-related value and method.
    childern object detectors should be inherited from this class

    - Example: 
            class Yolo(ObjDetector_base):
                def __init__(self,args,**kwargs):
                    super().__init__(args)
                    self.args = args
                
                def _build_model(self,**kwargs):
                    # model building-related codes. 
                
                def run(self,input):
                    # model inference-related codes.
    >> The wrapper class can be modified by any other implementations of the detector.
    Created by H. M. Kim (22.05.09)
    """
    def __init__(self, args,**kwargs):
        self.args = args
        self.target_classes = args.target_classes
        self.device = args.device
        self.fp16   = args.fp16
    
    @abstractmethod
    def _build_model(self,**kwargs):
        self.detector_type = self.args.detector
    
    @abstractmethod
    def run(self, input):
        pass

class ObjTracker_base(ABC):
    """
    Object Tracker Abstract Class
    Initialize the basic object tracker-related value and method.
    childern object tracker should be inherited from this class

    - Example: 

    >> The wrapper class can be modified by any other implementations of the tracker.
    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        self.args = args
    
    @abstractmethod
    def _build_model(self,**kwargs):
        self.tracker_type = self.args.tracker
        print("- tracker_type : ", tracker_type)
    
    @abstractmethod
    def run(self):
        pass

class MetaVideoPipe_base(ABC):
    """
    Default Object Detector + Tracker Application abstract class
    
    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        self.args = args

    def _build_pipeline(self,**kwargs):
        import importlib
        detector_wrapper = 'model.detector.'+self.args.detector+'_wrapper'
        print("- import detector : ", detector_wrapper)
        detector_wrapper = importlib.import_module(detector_wrapper)
        detector_class = getattr(detector_wrapper, 'Model')
        self.detector = detector_class(self.args, **kwargs)

        #tracker_wrapper = 'model.tracker.'+self.args.tracker+'_wrapper'
        #print("- import tracker : ", tracker_wrapper)
        #tracker_wrapper = importlib.import_module(tracker_wrapper)
        #tracker_class = getattr(tracker_wrapper, 'Model')
        #self.tracker = tracker_class(self.args, **kwargs)

    @abstractmethod
    def __call__(self, input):
        pass

    @staticmethod
    def _get_vid_info(video):
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(video.get(cv2.CAP_PROP_FPS))
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        return video_width, video_height, video_fps, video_length
