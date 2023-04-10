from typing import List, Dict, Tuple
from utils.utils import COLOR, CLASSES
import cv2
import numpy as np
import math
import os
import random
import copy

class Point:
    def __init__(self, x, y) -> None:
        self.x = int(x)
        self.y = int(y)
    
    def inside(self, polygon) -> bool:
        '''Validate point inside a polygon area'''
        # import ipdb; ipdb.set_trace()
        coutour = np.array(polygon.points).reshape(4, 1, 2)
        score = cv2.pointPolygonTest(contour=coutour, pt=(self.x, self.y), measureDist=True)
        return True if score > 0 else False
    
    def over_line_has_2_point(self, point1:Tuple, point2:Tuple):
        points = [point1, point2]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords)[0]
        x_at_line = (self.y - c)//m
        return True if self.x < x_at_line else False


class Box:
    def __init__(self, tl:Point, br:Point) -> None:
        self.tl = tl
        self.br = br

    def has_center(self) -> Point:
        '''return a center point a bounding box'''
        return Point((self.tl.x+self.br.x)//2, (self.tl.y+self.br.y)//2)
    
    def draw_box_in(self, image, color:tuple=COLOR.green, thickness:int=1, label:str='') -> None:
        '''
        draw object bounding box.
        '''
        const_edge_len = 50
        # top left
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y), 
                (self.box.tl.x, self.box.tl.y + const_edge_len), 
                color, thickness) 
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y),
                (self.box.tl.x + const_edge_len, self.box.tl.y),
                color, thickness) 

        # bottom_right
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x - const_edge_len, self.box.br.y),
                color, thickness) 
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x, self.box.br.y - const_edge_len),
                color, thickness) 

        # put label text of the bounding box
        cv2.putText(image, label,
                    (self.box.tl.x, self.box.tl.y-15),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )


class Polygon():
    def __init__(self, points:List) -> None:
        self.points = [(int(pt[0]), int(pt[1])) for pt in points]
    
    def draw_box(self, image:np.array, color:tuple=COLOR.yellow, thickness:int=1, label:str='') -> None:
        '''draw polygon in frame'''
        # drawing...
        points = np.array(self.points)
        cv2.polylines(img=image, pts=[points], isClosed=True, color=color, thickness=thickness)
        
        # draw_label
        top_point = min(self.points, key=lambda x:x[1])
        cv2.putText(image, label,
                    (top_point[0], top_point[1]-15),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )


class Object:
    def __init__(self, track_id, id_object, name_object, box:Box, conf) -> None:
        self.track_id    = track_id
        self.id_object   = id_object
        self.name_object = name_object
        self.box         = box            # top left bottom right
        self.conf        = round(conf, 3)           # confidence of object
        self.local       = None
    
    def overlap_with(self, object_) -> float:
        # Compute S area 2 boxes
        S_self = (self.box.br.x-self.box.tl.x) * (self.box.br.y-self.box.tl.y)
        S_object = (object_.box.br.x-object_.box.tl.x) * (object_.box.br.y-object_.box.tl.y)
        # Compute coor overlap area
        xx = max(self.box.tl.x, object_.box.tl.x)
        yy = max(self.box.tl.y, object_.box.tl.y)
        aa = min(self.box.br.x, object_.box.br.x)
        bb = min(self.box.br.y, object_.box.br.y)
        # Compute S overlap area
        w = max(0, aa-xx)
        h = max(0, bb-yy)
        intersection_area = w*h
        # Compute S 2 boxes merge
        union_area = S_self + S_object - intersection_area
        Iou = intersection_area/union_area
        return Iou
    
    
    def draw_box(self, image:np.array, color=COLOR.blue, thickness=2, label:str=''):
        cv2.rectangle(image,
                    (self.box.tl.x, self.box.tl.y),
                    (self.box.br.x, self.box.br.y),
                    color=color,
                    thickness=thickness
                    )
        cv2.putText(image, label,
                (self.box.tl.x, self.box.tl.y-15),
                fontScale = 0.8,
                color=color,
                thickness=thickness,
                fontFace=cv2.LINE_AA
                )


class Item(Object):
    def __init__(self, track_id, id_object, name_object, box: Box, conf) -> None:
        super().__init__(track_id, id_object, name_object, box, conf)
        self.cnt_in_area_detect = 0

    def inside(self, area:Polygon) -> bool:
        point_1 = copy.deepcopy(self.box.tl)
        point_2 = Point(x=self.box.br.x, y=self.box.tl.y)
        point_3 = copy.deepcopy(self.box.br)
        point_4 = Point(x=self.box.tl.x, y=self.box.br.y)
        return 2 < point_1.inside(area)+point_2.inside(area)+point_3.inside(area)+point_4.inside(area) 

    def __eq__(self, __value: object) -> bool:
        return self.track_id == __value.track_id


class Hand(Object):
    def __init__(self, _track_id, id_object, name_object, box: Box, conf, id_person) -> None:
        super().__init__(_track_id, id_object, name_object, box, conf)
        self.id_person = id_person

    def touch(self, item:Item, thres=0.5) -> bool:
        iou_score = self.overlap_with(item)
        return True if iou_score >= thres else False

class Person(Object):
    def __init__(self, track_id, id_object, name_object, box: Box, conf) -> None:
        super().__init__(track_id, id_object, name_object, box, conf)
        self.left_hand_kp   = []
        self.right_hand_kp  = []
        self.left_leg_kp    = []
        self.right_leg_kp   = []

        self.item_holding   = []
        self.item_paid     = []
        self.hold_item_flag = True if len(self.item_holding)>0 else False 

        self.in_shelve              = False
        self.payment_flag           = False
        self.cnt_in_area_pay        = 0
        self.paid                   = False
    
    def stand_in(self, area:Polygon) -> bool:
        '''
        human stand in area and not?
        '''
        cen_point_2_leg = Point(x=(self.box.tl.x+self.box.br.x)//2, y=self.box.br.y-random.randint(4, 10))
        return cen_point_2_leg.inside(area) 
    
    def hold(self, item:Item, thres:float) -> List[Item]:
        item_holdings = []
        l_cnt += sum(map(lambda each_kp: each_kp.in_box(item.box), self.left_hand_kp))
        r_cnt += sum(map(lambda each_kp: each_kp.in_box(item.box), self.right_hand_kp))
        
        if l_cnt/len(self.left_hand_kp)>=thres or r_cnt/len(self.right_hand_kp)>=thres:
            for each_item in item_holdings:
                if each_item.box.tl.x != item.box.tl.x:
                    item_holdings.append(item)
        return item_holdings

    def draw_box(self, image: np.array, color=COLOR.blue, thickness=2, label: str = ''):
        const_edge_len = 60
        # top left
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y), 
                (self.box.tl.x, self.box.tl.y + const_edge_len), 
                color, thickness) 
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y),
                (self.box.tl.x + const_edge_len, self.box.tl.y),
                color, thickness) 
        # bottom_right
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x - const_edge_len, self.box.br.y),
                color, thickness) 
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x, self.box.br.y - const_edge_len),
                color, thickness) 
        # visual Label
        cv2.putText(image, label,
                    (self.box.tl.x+10, self.box.tl.y+30),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )
        
    def detect_items_around_hand(self, model_item, img_org, ratio_box_det:list, ratio_distance_hand:float) -> List:
        dets = []
        wid_box, hei_box = ratio_box_det[0]*img_org.shape[1], ratio_box_det[1]*img_org.shape[0]
        all_kps_hand = self.left_hand_kp + self.right_hand_kp
        # if len(all_kps_hand) > 1:
        #     dis_2hands = math.sqrt((self.left_hand_kp[0].x-self.right_hand_kp[0].x)**2 + (self.left_hand_kp[0].y-self.right_hand_kp[0].y)**2)
        #     height_person = math.sqrt((self.box.br.x-self.box.tl.x)**2 + (self.box.br.y-self.box.tl.y)**2)
        #     if dis_2hands < height_person*ratio_distance_hand:
        #         all_kps_hand = [all_kps_hand[0]]

        # for kp_hand in all_kps_hand:
        if len(all_kps_hand):
            kp_hand = all_kps_hand[0]
            box_det = [max(0, kp_hand.x-wid_box//2), max(0, kp_hand.y-hei_box//2),                                                   # top_left
                        min(img_org.shape[1], kp_hand.x+wid_box//2), min(img_org.shape[0], kp_hand.y+hei_box//2)]                    # bot_right
            box_det = list(map(int, box_det))
            img_det = img_org[box_det[1]:box_det[3], box_det[0]:box_det[2]]
            # print(f"{self.track_id}---------{kp_hand.x}  { kp_hand.y}      {box_det}                     {img_org.shape}")
            # cv2.imwrite(f"debug/debug_{str(img_org.shape[1])}.jpg", img_det)
            det = model_item.detect(img_det)
            for each in det[0]:
                dets.append(each)
        return dets
    
    def __eq__(self, __value: object) -> bool:
        return self.track_id == __value.track_id

class Frame:
    def __init__(self, id_, img) -> None:
        if isinstance(img, str):
            self.path = img
            self.name = os.path.basename(img)
            self.img  = cv2.imread(img)
        else:
            self.img  = img
            self.path = None
            self.name = None
        self.id = id_
        self.shape   = self.img.shape
        self.width   = self.shape[1]
        self.height  = self.shape[0]
        self.channel = self.shape[2] if len(self.shape) > 2 else None
    

class KeyPoint(Point):
    def __init__(self, x, y, id_:int, name:str, conf:float, color:tuple, swap:str) -> None:
        self.x     = x
        self.y     = y
        self._id   = id_
        self.name  = name
        self.conf  = conf
        self.color = color
        self.swap  = swap

    def in_box(self, box:Box) -> bool:
        return self.x>box.tl.x and self.y>box.tl.y and self.x<box.br.x and self.y<box.br.y


class Area:
    def __init__(self, image, args) -> None:
        self.img         = image
        img_h, img_w     = self.img.shape[:2]
        self.shelve      = Polygon([[x*img_w, y*img_h] for x, y  in args['shelve']])
        self.line_shelve = Polygon([[x*img_w, y*img_h] for x, y  in args['line_shelve']])
        self.payment     = Polygon([[x*img_w, y*img_h] for x, y  in args['payment']])
        self.item_detect = Polygon([[x*img_w, y*img_h] for x, y  in args['item_detect']])
        
    def draw(self):
        self.shelve.draw_box(self.img, COLOR.red, 1, f"shelve")
        self.line_shelve.draw_box(self.img, COLOR.pink, 1, f"line_shelve")
        self.payment.draw_box(self.img, COLOR.green, 1, f"payment")
        self.item_detect.draw_box(self.img, COLOR.green, 1, f"item_detect")



