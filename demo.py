import os 
from opt import args
from model.video_demo import DemoAPP

def process(args):
    # build app
    app = DemoAPP(args)
    app._build_pipeline()
    # test videos
    app()

if __name__ == "__main__":
    process(args)
