# argparser 
import argparse

parser = argparse.ArgumentParser()

# === setting display ===
parser.add_argument('--show_video_info',type=bool, default=True)
parser.add_argument('--show_detector_config',type=bool, default=True)
parser.add_argument('--show_tracker_config',type=bool, default=True)
parser.add_argument('--show_detector_model_layers',type=bool, default=False)
parser.add_argument('--show_tracker_model_layers',type=bool, default=False)

# === visualization ===
parser.add_argument('--visualize_frame',type=bool, default=True)

# === exp_config ===
parser.add_argument('--exp_config',type=str, default='configs.yoloxl_deepsort_default')

# === data_related ===
parser.add_argument('--video_dir',type=str, default='videos')
parser.add_argument(
    "--fp16",dest="fp16",default=False,action="store_true",
    help="Adopting mix precision evaluating."
)
# === device ===
parser.add_argument('--device',type=str, default='cuda:0')

args = parser.parse_args()