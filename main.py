import cv2
import copy
import yaml
import numpy as np
import time
import argparse
import os
import torch
import datetime
import logging
from threading import Thread, Lock
from queue import Queue

from __init__ import (config, area_cf, person_track_cf, person_detect_cf, pose_cf, 
                      item_detect_cf, item_track_cf, behavior_cf, debug_cf, visual_cf)
from utils.utils import get_monitor_size, convert_boxobject2nparr, filter_object, CLASSES, COLOR
from utils import logger

from predictor import PoseWithMobileNetDetector, ItemDetector, PersonDetector
from behavior import Behavior
from detector.models.with_mobilenet import PoseEstimationWithMobileNet
from ultralytics import YOLO
from objects import Person, Item, Point, Box, Area, Frame

from bytetrack_person.tracker.byte_tracker import BYTETrackerPerson
from bytetrack_item.tracker.byte_tracker import BYTETrackerItem


def make_parser():
    parser = argparse.ArgumentParser("Behavior retail")
    #camera
    parser.add_argument("--input_path", action="store", type=str, default=0)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser

class Processor():
    def __init__(self) -> None:
        # init parser
        self.args = make_parser().parse_args() 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.logger = logger.set_logger(level=logging.INFO)
        
        self.logger.info('Loading model yolov8s person detection ...')
        self.person_detector = PersonDetector(model=YOLO,
                                weight=person_detect_cf['weight'],
                                device=self.device,
                                classes=person_detect_cf['categories'],
                                conf=person_detect_cf['conf'],
                                iou_thes=person_detect_cf['iou'],
                                agnostic_nms=person_detect_cf['agnostic_nms'],
                                args=person_detect_cf)

        # self.logger.info('Loading model pose estimation ...')
        # self.pose_detector = PoseWithMobileNetDetector(model=PoseEstimationWithMobileNet(), 
        #                                    weight=pose_cf['weight'],
        #                                    device=self.device, 
        #                                    stride=pose_cf['stride'],
        #                                    height_size=pose_cf['height_size'],
        #                                    upsample_ratio=pose_cf['upsample_ratio'],
        #                                    delay=pose_cf['delay'])
        
        self.logger.info('Loading model yolov8 item detection ...')
        self.item_detector = ItemDetector(model=YOLO,
                              weight=item_detect_cf['weight'],
                              device=self.device,
                              classes=item_detect_cf['categories'],
                              conf=item_detect_cf['conf'],
                              iou_thes=item_detect_cf['iou'],
                              agnostic_nms=item_detect_cf['agnostic_nms'],
                              args=item_detect_cf)
        
        self.logger.info('Loading tracking bytetrack person ...')
        self.person_tracker = BYTETrackerPerson(track_thresh=person_track_cf['track_thresh'],
                                          match_thresh=person_track_cf['match_thresh'],
                                          track_buffer=person_track_cf['track_buffer'],
                                          frame_rate  =person_track_cf['frame_rate'],
                                          mot20       =person_track_cf['mot20'])
        
        self.logger.info('Loading tracking bytetrack item ...')
        self.item_tracker = BYTETrackerItem(track_thresh  =item_track_cf['track_thresh'],
                                          match_thresh=item_track_cf['match_thresh'],
                                          track_buffer=item_track_cf['track_buffer'],
                                          frame_rate  =item_track_cf['frame_rate'],
                                          mot20       =item_track_cf['mot20'])
        self.frame_queue = Queue(maxsize=config['max_queue'])
        self.behavior = Behavior(behavior_cf['num_frame_ignore'], behavior_cf['num_frame_process'], Area)

        # init_video_debug
        self.mo_wid_save, self.mo_hei_save = get_monitor_size()[:2]
        self.writer_in = None
        self.writer_out = None
        self.area = None

    def read_video(self):
        cap = cv2.VideoCapture(self.args.input_path) 
        if self.args.debug:
            time_save = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            video_paths =[f"{debug_cf['save_result']}/{time_save}/video_in", 
                        f"{debug_cf['save_result']}/{time_save}/video_out"]
            for each_path in video_paths:
                if not os.path.exists(each_path):
                    os.makedirs(each_path, exist_ok=False)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if debug_cf['video_in']:
                self.writer_in = cv2.VideoWriter(f"{video_paths[0]}/input.avi", fourcc, debug_cf['fps'], (frame_width, frame_height))
            if debug_cf['video_out']:
                self.writer_out = cv2.VideoWriter(f"{video_paths[1]}/output.avi", fourcc, debug_cf['fps'], (frame_width, frame_height))
        
        while True:
            ret, img = cap.read()
            if img is None:
                self.logger.info(f"No frame to read from: {self.args.input_path}")
                time.sleep(3)
                os._exit(1)
            while self.frame_queue.full():
                time.sleep(0.02)
            if ret:
                self.frame_queue.put(img)
        
    def run(self):
        self.thread_input = Thread(target=self.read_video)
        self.thread_input.start() 
        log_person_results, log_item_results = [], []
        log_person_behavior, log_item_behavior = [], []
        frame_id = 0
        while True:
            sys_start_time = time.time()
            # print(f"=========FRAME==========={frame_id}")
            org_frame =  self.frame_queue.get()
            img = copy.deepcopy(org_frame)
            frame = Frame(frame_id, img)
            self.area = Area(frame.img, area_cf)

            persons, fps_person = self.person_detector.detect(frame.img)
            if len(persons):
                person_boxes = convert_boxobject2nparr(persons)
                online_targets = self.person_tracker.update(person_boxes, frame.shape[:2], person_track_cf['input_size'])
                person_trackers = filter_object(online_targets, person_track_cf['min_box_area'], person_track_cf['aspect_ratio_thresh'])
            else:
                person_trackers = []
            for tracker in person_trackers:
                box = Box(tl=Point(x=max(0, tracker.tlbr[0]), y=max(0, tracker.tlbr[1])),
                          br=Point(x=min(frame.width, tracker.tlbr[2]), y=min(frame.height, tracker.tlbr[3])))
                person = Person(track_id    = tracker.track_id,
                                id_object   = -1,
                                name_object = CLASSES[-1],
                                box         = box,
                                conf        = tracker.score)
                
                if person.stand_in(self.area.shelve):
                    person.local     = 'shelve'
                    person.in_shelve = True
                    person.paid      = False
                elif person.stand_in(self.area.item_detect) or person.box.has_center().inside(self.area.item_detect):
                    person.local = 'item_detect'
                elif person.stand_in(self.area.payment):
                    person.local = 'payment'
                    
                if person in log_person_results:
                    index = log_person_results.index(person)
                    log_person_results[index].box  = person.box
                    log_person_results[index].conf  = person.conf
                    log_person_results[index].local = person.local
                    if log_person_results[index].local == 'payment':
                        log_person_results[index].item_holding += person.item_holding

                    if log_person_results[index].in_shelve == False:
                        log_person_results[index].in_shelve = person.in_shelve
                    log_person_results[index].paid = person.paid
                else:
                    log_person_results.append(person)
                
                # # get human for behavior get item
                # for each_person in log_person_results:
                #     if each_person.local=='item_detect' and each_person.in_shelve and not each_person in log_item_behavior:
                #         log_item_behavior.append(each_person)
                if visual_cf['person']:
                    person.draw_box(frame.img, label=f"id: {person.track_id}")    
                
            items, fps_item = self.item_detector.detect(frame.img)
            if len(items):
                item_boxes = convert_boxobject2nparr(items)
                online_targets_item = self.item_tracker.update(item_boxes, frame.img.shape[:2], item_track_cf['input_size'])
                item_trackers = filter_object(online_targets_item, item_track_cf['min_box_area'], item_track_cf['aspect_ratio_thresh'])
            else:
                item_trackers = []
            
            for tracker in item_trackers:
                box = Box(tl=Point(x=max(0, tracker.tlbr[0]), y=max(0, tracker.tlbr[1])),
                          br=Point(x=min(frame.width, tracker.tlbr[2]), y=min(frame.height, tracker.tlbr[3])))
                item = Item(track_id    = tracker.track_id,
                            id_object   = tracker.id_object,
                            name_object = CLASSES[tracker.id_object],
                            box         = box,
                            conf        = tracker.score)
                if item.inside(self.area.shelve):
                    item.local = 'shelve'
                elif item.inside(self.area.item_detect):
                    person.local = 'item_detect'
                elif item.inside(self.area.payment):
                    item.local = 'payment'
                else: 
                    item.local = None
                if not item in log_item_results and item.inside(self.area.item_detect):
                    log_item_results.append(item)
                
                if visual_cf['item'] and not item.box.has_center().over_line_has_2_point(self.area.line_shelve.points[0], self.area.line_shelve.points[1]):
                    item.draw_box(frame.img, label=f"{item.track_id}: {item.name_object}")

                # print(len(log_item_results))
            # for item in log_item_results:
            #     print(f"id: {item.track_id}   item: {item.name_object}")


            step_1 = self.behavior.get_item(log_person_results, log_item_results, frame)
            step_2 = self.behavior.to_pay(step_1)


            sys_end_time = time.time()
            sys_fps = round(1/(sys_end_time - sys_start_time), 2)
            # SET UP FRAME:
            cv2.putText(frame.img, f"FPS: {sys_fps}",
                    (int(1/20*frame.width), int(1/17*frame.height)),
                    fontScale = 2,
                    color=COLOR.yellow,
                    thickness=2,
                    fontFace=cv2.LINE_AA
                    )
            if visual_cf['area']:
                    self.area.draw()
            if self.args.debug:
                if debug_cf['video_in']:
                    self.writer_in.write(org_frame) 
                if debug_cf['video_out']:
                    self.writer_out.write(frame.img) 
            if self.args.show:
                frame.img = cv2.resize(frame.img, (int(self.mo_wid_save*1/3), int(self.mo_hei_save*2/3)), interpolation=cv2.INTER_AREA)
                cv2.imshow("Result", frame.img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            frame_id += 1


            

if __name__ == "__main__":
    processor = Processor()
    processor.run()

 
    
    
   
 
    