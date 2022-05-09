import os
import cv2
from model.metaclass import MetaVideoPipe_base

class DemoAPP(MetaVideoPipe_base):
    """
    Object Detector and Tracker Video-demo Application
    """
    def __init__(self, args, **kwargs):
        super().__init__(args)
        # strip if not video file by endwitch
        self.video_list = os.listdir(args.video_dir)
        self.supported_video_ext = ['mp4','avi']
        self.video_list = [x for x in self.video_list if x.split('.')[-1] in self.supported_video_ext]
        self.video_list.sort()
        self.visualize_frame = args.visualize_frame
        if len(self.video_list) ==0:
            raise Exception("No video file in {}".format(args.video_dir))

    def _build_pipeline(self,**kwargs):
        super()._build_pipeline(**kwargs)
        self.detector.change_confthre(self.args.detector_confthresh)

    def __call__(self):
        print("- Video DEMO APP")
        print("- Number of Videos in {} : {}".format(self.args.video_dir, len(self.video_list)))

        for video_name in self.video_list:
            print(" [Start Processing : {}]".format(video_name))
            # open video 
            video_path = os.path.join(self.args.video_dir, video_name)
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise Exception("Cannot open video file : {}".format(video_path))
            # get video info
            w, h, fps, vid_len = self._get_vid_info(video)
            if self.args.show_video_info:
                print("    - Video Info : {}".format(video_name))
                print("    - Video Size : {} x {}".format(w, h))
                print("    - Video FPS : {}".format(fps))
                print("    - Video Length : {}".format(vid_len))
            # get video frame
            video_frame = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                video_frame += 1
                # run detector
                pred, img_info = self.detector.run(frame)
                result_frame = self.detector.visual(pred[0], img_info, self.detector.confthre)
                result_frame = cv2.resize(result_frame, (w, h))

                if self.visualize_frame:
                    cv2.imshow("yolox", result_frame)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break



    

