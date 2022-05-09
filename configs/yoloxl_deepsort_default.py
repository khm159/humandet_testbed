detector = dict(
    type='yolox',
    model='yolox_l',
    cfg=dict(
        checkpoint_path='pretrained_models/detector/yolox/yolox_l.pth', 
        conf_thres=0.30,
        iou_thres=0.45,
        agnostic_nms=False,
        target_classes=[0],
        max_det=100,
        verbose=False,
        batch_size=8,
    ),
)

tracker = dict(
    type='deepsort',
    cfg=dict(
        Extractor=dict(
            type='DefaultExtractor', 
            checkpoint_path='pretraied_models/tracker/deepsort/ckpt.t7',
        ),
        max_dist=0.2,
        min_confidence=0.3,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        target_classes=[0],
))