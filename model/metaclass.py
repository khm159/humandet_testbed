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
    def __init__(self, args, **kwargs):
        self.args   = args
        self.device = args.device
        self.fp16   = args.fp16
    
    @abstractmethod
    def build_model(self,**kwargs):
        pass
    
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
    """
    def __init__(self,args,**kwargs):
        self.args = args
        self.device = args.device
    
    @abstractmethod
    def _build_model(self,**kwargs):
        self.tracker_type = self.args.tracker
        print("- tracker_type : ", self.tracker_type)
    
    @abstractmethod
    def _update(self, *args, **kwargs):
        """update tracking object every call."""
        raise NotImplementedError
    
    @abstractmethod
    def to(self, device):
        """Define the device where the computation will be performed."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self._update(*args, **kwargs)

class MetaVideoPipe_base(ABC):
    """
    Default Object Detector + Tracker Application abstract class
    
    Created by H. M. Kim (22.05.09)
    """
    def __init__(self,args,**kwargs):
        self.args   = args
        self.device = args.device
        self.fp16   = args.fp16
    
    def _load_config_dict(self):
        """load config dict from python file"""
        import importlib
        config_wrapper = self.args.exp_config
        config_wrapper = importlib.import_module(config_wrapper)

        print(" - loaded experiment config file : ", config_wrapper)
        self.detector_config = getattr(config_wrapper, 'detector')
        self.tracker_config  = getattr(config_wrapper, 'tracker')
        print("\n\n    [Detector config] ")
        print(self.detector_config)
        print("\n    [Tracker config]  ")
        print(self.tracker_config,'\n\n')
        
    def _build_pipeline(self,**kwargs):
        import importlib
        if self.args.exp_config is None:
            raise ValueError("Please set the experiment config file.")

        # config file loading 
        self._load_config_dict()

        # detector model building
        detector_wrapper = \
            'model.detector.'+self.detector_config['type']+'_wrapper'
            
        print("- import detector : ", detector_wrapper)
        detector_wrapper = importlib.import_module(detector_wrapper)
        detector_class = getattr(detector_wrapper, 'Model')
        self.detector = detector_class(self,args, **kwargs)
        self.detector.upate_config(self.detector_config)
        self.detector.build_model()

        # tracker model building
        tracker_wrapper = \
            'model.tracker.'+self.tracker_config['type']+'_wrapper'
        print("- import tracker : ", tracker_wrapper)
        tracker_wrapper = importlib.import_module(tracker_wrapper)
        tracker_class = getattr(tracker_wrapper, 'Model')
        self.tracker = tracker_class(self.args, **kwargs)
        self.tracker.upate_config(self.tracker_config)
        self.tracker.build_model()

            
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
