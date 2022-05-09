# argparser 
import argparse

parser = argparse.ArgumentParser()

# === tracker configures ===
parser.add_argument('--tracker',type=str, default='deepsort',choices=['deepsort','sort'])
parser.add_argument('--tracker_model' ,type=str, default='[mobilenetv2_1.0]_[market1501]')

# === detector configures 
parser.add_argument('--detector',type=str, default='yolox',choices=['yolox','yolor','yolov5'])
parser.add_argument('--detector_model',type=str, default='yolox_l')
parser.add_argument('--target_classes',type=list, default=[0], help='choosing in the COCO classes names')
parser.add_argument('--detector_ckpt',type=str, default='pretraied_models/detector/yolox/yolox_l.pth')
parser.add_argument('--detector_confthresh',type=float, default=0.30)

# === setting display ===
parser.add_argument('--show_video_info',type=bool, default=True)
parser.add_argument('--show_detector_config',type=bool, default=True)
parser.add_argument('--show_detector_model_layers',type=bool, default=False)

# === visualization ===
parser.add_argument('--visualize_frame',type=bool, default=True)

# === data_related ===
parser.add_argument('--video_dir',type=str, default='videos')
parser.add_argument(
    "--fp16",dest="fp16",default=False,action="store_true",
    help="Adopting mix precision evaluating."
)
# === device ===
parser.add_argument('--device',type=str, default='cuda:0')

args = parser.parse_args()