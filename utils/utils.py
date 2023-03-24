import tkinter as tk
import cv2

class COLOR: 
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    fuchsia = (255, 0, 255)
    pink = (241, 167, 228)


class CLASSES:
    dark_noodles = {'id': 0, "name":'dark_noodles'}
    g7           = {'id': 1, "name":'g7'}
    haohao       = {'id': 2, "name":'haohao'}
    modern       = {'id': 3, "name":'modern'}
    nabati       = {'id': 4, "name":'nabati'}
    nescafe      = {'id': 5, "name":'nescafe'}
    oreo         = {'id': 6, "name":'oreo'}
    passiona     = {'id': 7, "name":'passiona'}


class AREA:
    '''
        - Area coordinates is the ratio of that point on the image to the real coordinates
        - Coordinate formated ratio: ((tl), (tr), (br), (bl))
    '''
    # shelve = ((0.01, 0.14), (0.08, 0.1), (0.29, 0.87), (0.08, 1))
    selection = ((0.309, 0.0333), (0.421, 0.103), (0.5333, 0.7111), (0.313, 0.6981))
    # attend = ((0.14, 0.5), (0.4, 0.17), (0.73, 0.33), (0.58, 0.97))
    payment = ((0.5375, 0.572), (0.751, 0.561), (0.843, 0.929), (0.663, 0.983))

    line_shelve = ((0.378, 0), (0.498, 0.687))


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

