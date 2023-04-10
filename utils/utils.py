import tkinter as tk
import cv2
from typing import List
import numpy as np

# from object import Person

class COLOR: 
    red     = (0, 0, 255)
    blue    = (255, 0, 0)
    green   = (0, 255, 0)
    yellow  = (255, 255, 0)
    fuchsia = (255, 0, 255)
    pink    = (241, 167, 228)


CLASSES = {
    -1: 'person',
    0: 'dark_noodles',
    1: 'g7',
    2: 'haohao',
    3: 'modern',
    4: 'nabati',
    5: 'nescafe',
    6: 'oreo',
    7: 'passiona',
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