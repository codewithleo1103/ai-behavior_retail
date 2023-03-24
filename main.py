import cv2
import yaml
import numpy as np
import time
import argparse
import os
import datetime

from utils.utils import get_monitor_size


def make_parser():
    parser = argparse.ArgumentParser("Behavior retail")
    #camera
    parser.add_argument("--input_path", action="store", type=str, default=0)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser

def main():
    # init parser
    args = make_parser().parse_args() 
    file = open(r'./config.yml')
    config = yaml.full_load(file)
    
    # init_video_debug
    debug = args.debug
    if debug:
        time_save = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        video_path =[f"{config['debug']['save_result']}/{time_save}/video_in", 
                     f"{config['debug']['save_result']}/{time_save}/video_out"]
        for each_path in video_path:
            if not os.path.exists(each_path):
                os.makedirs(each_path, exist_ok=False)

        mo_wid, mo_hei = get_monitor_size()[:2]
        writer_in = cv2.VideoWriter(video_path[0], config['debug']['fourcc'], config['debug']['fps'], (mo_wid, mo_hei))
        writer_out = cv2.VideoWriter(video_path[1], config['debug']['fourcc'], config['debug']['fps'], (mo_wid, mo_hei))

    # init model
    cap = cv2.VideoCapture(args.input_path)
    while(True):
        ret, frame = cap.read()
        if ret:
    
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()