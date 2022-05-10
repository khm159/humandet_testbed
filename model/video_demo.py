import os
import cv2
import torch
import numpy as np
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
    
    def get_bbox_ids_from_tracker(self, track_ids):

        bbox_xyxy = np.array([], dtype=np.int64)
        identities = np.array([], dtype=np.int64)
        bbox_xyxy = track_ids[:,:4]
        identities = track_ids[:,-1]

        return bbox_xyxy, identities 

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

            prv_bbox = np.asarray([])
            prv_identities = np.asarray([])
            identities = np.asarray([])
            # get video frame
            video_frame = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                video_frame += 1

                # run detector
                pred = self.detector.run(frame)

                # ====
                if int(pred[0].shape[0]) == 0:
                    if len(prv_bbox) != 0 and len(prv_identities) != 0:
                        bbox_xyxy = prv_bbox
                        identities = prv_identities
                    else:
                        bbox_xyxy = np.array([])
                        identities = np.array([])
                    data = None
                else:
                    # run tracker   
                    data = dict(
                        img=self.tracker._img_preproc(frame), 
                        ori_img=frame, 
                        idxs=[video_frame],
                        objects = pred[0] 
                    )
                    track_ids = self.tracker(data)

                    # bbox to bbox_xyxy for visualization 
                    if len(track_ids) != 0:
                        bbox_xyxy, identities = self.get_bbox_ids_from_tracker(track_ids)
                        prv_bbox = bbox_xyxy
                        prv_identities = identities
                    else:
                        if len(prv_bbox) != 0 and len(prv_identities) != 0:
                            bbox_xyxy = prv_bbox
                            identities = prv_identities      
                        else:
                            bbox_xyxy = np.array([])
                            identities = np.array([])
                
                # draw bbox 
                result_frame = self.detector.draw_bbox(
                    img=frame,
                    bboxes=bbox_xyxy,
                    identities=identities
                )

                # visualize
                if self.visualize_frame:    
                    cv2.imshow("yolox", result_frame)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
