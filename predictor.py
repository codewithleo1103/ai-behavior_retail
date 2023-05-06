import time
import torch
import cv2
import numpy as np
import copy
from typing import List
from detector.modules.load_state import load_state
# from detector.modules.keypoints import extract_keypoints, group_keypoints
# from detector.modules.pose import Pose
# from detector.val import normalize, pad_width

from utils.utils import COLOR, CLASSES, xywh_to_xyxy

from objects import Person, KeyPoint, Item, Box, Point

from __init__ import pose_cf, item_detect_cf, person_detect_cf, visual_cf


class Detector:
    '''
    Detector:
    Args:
        weight_path: path of parameter file .pt or .pth
        model_load: Architecture of model    
    '''
    def __init__(self, model, weight, device) -> None:
        self.model = model
        self.device = device
        self.weight = weight

    def pre_process(self, img, args=None):
        '''
        Args:
            input:
        Output:
            img process
        '''
        return img
    
    def post_process(self, dets, args=None) -> List:
        '''
        pre-process rule, convert output to standart output
        Args:
            det: Object detected from detect function
        
        Output: List of object converted to standard.
        '''
        return dets
    
    def detect(self, img, args=None):
        img_procced = self.pre_process(img)

        start_time = time.time()
        with torch.no_grad():
            det = self.model(img_procced)
        end_time = time.time()
        fps = round(1/(end_time-start_time), 3)

        det = self.post_process(det=det, original_img=img)
        return det, fps
    

class ItemDetector(Detector):
    def __init__(self, model, weight, device, classes, conf, iou_thes, agnostic_nms, args) -> None:
        super().__init__(model, weight, device)
        self.model        = model(weight) 
        if args['ex_onnx']:
            self.model.export(format='onnx')

        self.classes      = []
        for obj in classes:
            for id_obj, name_obj in CLASSES.items():
                if name_obj[0] == obj:
                    self.classes.append(id_obj)
                    break
        self.conf         = conf
        self.iou_thes     = iou_thes
        self.agnostic_nms = agnostic_nms
    
    def detect(self, orig_img):
        start_time = time.time()
        img = orig_img.copy()
        img = self.pre_process(img)
        dets = self.model.predict(img, 
                                  classes       = self.classes, 
                                  conf          = self.conf, 
                                  iou           = self.iou_thes,
                                  agnostic_nms  = self.agnostic_nms, 
                                  verbose       = False
                                  )
        dets = self.post_process(dets)
        end_time = time.time()
        fps = round(1/(end_time-start_time), 2)
        return dets, fps

    def post_process(self, dets) -> List:
        dets_final = []
        boxes = dets[0].boxes.xyxy
        classes = dets[0].boxes.cls
        confs = dets[0].boxes.conf
        for box, clas, conf in zip(boxes.tolist(), classes.tolist(), confs.tolist()):

                bbox = Box(tl=Point(x=box[0], y=box[1]),
                          br=Point(x=box[2], y=box[3]))
                item = Item(track_id=None,
                            id_object=int(clas),
                            name_object=CLASSES[int(clas)][0],
                            box=bbox,
                            conf=conf,
                            price=CLASSES[int(clas)][0][1])
                dets_final.append(item)
        return dets_final


class PersonDetector(ItemDetector):
    def __init__(self, model, weight, device, classes, conf, iou_thes, agnostic_nms, args) -> None:
        super().__init__(model, weight, device, classes, conf, iou_thes, agnostic_nms, args)
        self.classes = [0]
        if args['ex_onnx']:
            self.model.export(format='onnx')

    def post_process(self, dets) -> List:
        dets_final = []
        boxes = dets[0].boxes.xyxy
        classes = dets[0].boxes.cls
        confs = dets[0].boxes.conf
        for box, clas, conf in zip(boxes.tolist(), classes.tolist(), confs.tolist()):

                bbox = Box(tl=Point(x=box[0], y=box[1]),
                          br=Point(x=box[2], y=box[3]))
                person = Person(track_id=None,
                                id_object=-1,
                                name_object=CLASSES[-1][0],
                                box=bbox,
                                conf=conf)
                dets_final.append(person)
        return dets_final

    
class PoseWithMobileNetDetector(Detector):
    def __init__(self, model, weight, device, stride, height_size, upsample_ratio, delay) -> None:
        super().__init__(model, weight, device)
        self.model = model
        checkpoint = torch.load(weight, map_location='cpu')
        load_state(self.model, checkpoint)
        self.model.eval()
        self.stride = stride
        self.height_size = height_size
        self.upsample_ratio = upsample_ratio
        self.delay = delay
        self.device = device
        self.num_keypoints = Pose.num_kpts
        self.body_parts = [pose_cf['pose_left_hand'], pose_cf['pose_right_hand'], pose_cf['pose_left_foot'], pose_cf['pose_right_foot']]

        if self.device != 'cpu':
            self.model.cuda()
    
    def detect(self, person, img):
        start_time = time.time()
        orig_img = img.copy()
        img_person = self.pre_process(person, orig_img)
        heatmaps, pafs, scale, pad = infer_fast(self.model, img_person, self.height_size, self.stride, self.upsample_ratio, cpu=False)
        
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        current_poses = self.post_process(current_poses, person)
        end_time = time.time()
        fps = round(1/(end_time-start_time), 2)
        return current_poses, fps
    
    def pre_process(self, person, args):
        img = copy.deepcopy(args)
        person_img = img[person.box.tl.y:person.box.br.y, person.box.tl.x:person.box.br.x]
        return person_img 
        
    def post_process(self, dets, args) -> List:
        person = copy.deepcopy(args)
        best_kps = max(dets, key=lambda ele: ele.confidence)
        best_kps.keypoints[best_kps.keypoints[:, 0]!=-1, 0] +=  person.box.tl.x
        best_kps.keypoints[best_kps.keypoints[:, 1]!=-1, 1] +=  person.box.tl.y

        pose_body_parts = []
        non_point_x, non_point_y = (best_kps.keypoints[:, 0]==-1).sum(), (best_kps.keypoints[:, 1]==-1).sum()
        if non_point_x<pose_cf['num_kps']-3 and non_point_y<pose_cf['num_kps']-3:
            for cf_pose_part in self.body_parts:
                pose_body_part = []
                for each_part in cf_pose_part:
                    x_kp, y_kp = best_kps.keypoints[each_part:each_part+1, :][0]
                    if x_kp!=-1 and y_kp!=-1:
                        pose = KeyPoint(x=x_kp, y=y_kp,
                                        id_= each_part, name=None,
                                        conf=None, color=COLOR.red, swap=None)
                        pose_body_part.append(pose)
                pose_body_parts.append(pose_body_part)
        return pose_body_parts, best_kps
    

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu=False,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad