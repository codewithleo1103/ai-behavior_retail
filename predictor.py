import time
import torch
from typing import List

class Detector:
    '''
    Detector:
    Args:
        weight_path: path of parameter file .pt or .pth
        model_load: Architecture of model    
    '''
    def __init__(self, weight, device) -> None:
        self.device = device
        self.weight = weight

    def pre_process(self, inputs):
        '''
        Args:
            input:
        Output:
            img process
        '''
        return inputs
    
    def post_process(self, dets) -> List:
        '''
        pre-process rule, convert output to standart output
        Args:
            det: Object detected from detect function
        
        Output: List of object converted to standard.
        '''
        return dets
    
    def detect(self, img):
        img_procced = self.pre_process(img)

        start_time = time.time()
        with torch.no_grad():
            det = self.model(img_procced)
        end_time = time.time()
        fps = round(1/(end_time-start_time), 3)

        det = self.post_process(det=det, original_img=img)
        return det, fps