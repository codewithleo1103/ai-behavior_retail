import tkinter as tk
import cv2
from typing import List
import numpy as np
import copy

# from object import Person

class COLOR: 
    red     = (0, 0, 255)
    blue    = (255, 0, 0)
    green   = (0, 255, 0)
    yellow  = (255, 255, 0)
    fuchsia = (255, 0, 255)
    pink    = (241, 167, 228)


CLASSES = {
    -1: ['person', -1],
    0: ['dark_noodles', 20000],
    1: ['g7', 30000],
    2: ['haohao', 10000],
    3: ['modern', 10000],
    4: ['nabati', 15000],
    5: ['nescafe', 30000],
    6: ['oreo', 30000],
    7: ['passiona', 100000],
}    


def get_monitor_size():
    root = tk.Tk()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()
    width_mm = root.winfo_screenmmwidth()
    height_mm = root.winfo_screenmmheight()
    # 2.54 cm = in
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    width_dpi = width_px/width_in
    height_dpi = height_px/height_in
    return width_px, height_px, width_mm, height_mm, width_in, height_in, width_dpi, height_dpi


def convert_boxobject2nparr(objects:List) -> np.array:
    dets = [[object_.box.tl.x, object_.box.tl.y, object_.box.br.x, object_.box.br.y, object_.conf, object_.id_object] 
            for object_ in objects]
    return np.array(dets)


def filter_object(out_tracker, min_box_area, aspect_ratio_thresh):
    tracks = []
    for t in out_tracker:
        tlwh = t.tlwh
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        # print(f"vertical: {vertical}              {tlwh[2] * tlwh[3] > min_box_area}")
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            tracks.append(t)
    return tracks

def xywh_to_xyxy(img, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+w, y1+h
    x2 = x2 if x2 <= img.shape[1] else img.shape[1]
    y2 = y2 if y2 <= img.shape[0] else img.shape[0]
    return x1, y1, x2, y2


def visualize_payment(image, result:list):
    title = "*****PAYMENT*****"
    x_start_person = 10
    y_start_person = 30
    cv2.putText(image,title,(55,y_start_person),cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR.red, 1)
    x_start_item = x_start_person + 30
    y_start_person += 40
    for person in result:
        cv2.putText(image,f"person {person.track_id}",(x_start_person,y_start_person),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (111, 0, 5), 1)
        y_start_item = y_start_person + 30
        if len(person.item_paid):
            for item in person.item_paid:
                cv2.putText(image,f"+ {item.name_object}: {item.price} VND",(x_start_item,y_start_item),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR.blue, 1)
                y_start_item += 40

            cv2.putText(image,f"Total money:  {person.compute_sum_item_price()} VND",(x_start_item,y_start_item),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR.red, 1)
        y_start_item += 20       
        cv2.putText(image,f"______________________________________________",(x_start_item,y_start_item),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR.pink, 1)
        y_start_item += 10

        y_start_person = y_start_item + 40
    