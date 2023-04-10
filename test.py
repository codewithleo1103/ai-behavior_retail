# listOfElems = [11, 22, 33, 45, 66, 77, 88, 99, 101]
# # Count odd numbers in the list
# count = sum(map(lambda x : x%2 == 1, listOfElems))
# print('Count of odd numbers in a list : ', count)
# import numpy as np
# a = np.array([[       1995,           0,         342,       858.8,      19.041]])
# b = np.array([       1995,           0,         342,       858.8])

# print((a[:, :4] == b).sum())


# from ultralytics import YOLO
# import glob
# import cv2
 
# model = YOLO("./models/person_weight/yolov8m.pt")  # load a pretrained YOLOv8n model
 
# # model.train(data="coco128.yaml")  # train the model
# # model.val()  # evaluate model performance on the validation set
# for img in glob.glob(f"/media/ubuntu/DATA/SANG_DEP_TRAI_O_DAY/DataTest_behavior_retail/data/datatraining/data/images/*"):
#     img = cv2.imread(img)
#     result = model.predict(source=img, 
#                 classes=[0],
#                 show=True,
#                 conf=0.6)  # predict on an image


# model.export(format="onnx") 


         # print(f"id: {person.track_id}      in_shelve: {person.in_shelve}")
                # if person.stand_in(self.area.shelve):
                    # kps_person, raw_kps = self.pose_detector.detect(person, frame)[0]
                    # if len(kps_person):
                    #     person.left_hand_kp, person.right_hand_kp, person.left_leg_kp, person.right_leg_kp = kps_person
                    # Indentify local of current person   
                
                    # if person.in_shelve:
                    #     kps_person, raw_kps = self.pose_detector.detect(person, frame)[0]
                    #     if visual_cf['keypoint']:
                    #         raw_kps.draw(frame)
                    #     if len(kps_person):
                    #         person.left_hand_kp, person.right_hand_kp, person.left_leg_kp, person.right_leg_kp = kps_person
            



# import numpy as np

# a = np.array([[141, 208],
#                 [141, 312],
#                 [185, 401],
#                 [111, 223],
#                 [ 81, 342],
#                 [ 96, 446],
#                 [ -1,  -1],
#                 [ -1,  -1],
#                 [ -1,  -1],
#                 [ -1,  -1]])

# a[a[:, 0] > 0, 0] = 50
# print(a)






from collections import Counter

my_list = [1, 2, 3, 4, 2, 3, 3, 3, 3, 1, 2, 2, 2]
most_common_elements = [element for element, count in Counter(my_list).most_common() if count == Counter(my_list).most_common(1)[0][1]]

print(most_common_elements)