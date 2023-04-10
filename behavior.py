from typing import List 
from objects import Area
from __init__ import behavior_cf, area_cf
from queue import Queue



class Behavior:
    def __init__(self, num_frame_ignore, num_frame_process, area) -> None:
        self.status = None
        self.area              = area
        self.num_frame_process = num_frame_process
        self.num_frame_ignore  = num_frame_ignore
        self.persons_process   = []
        self.frames_process    = []
        self.items_process     = []
        self.results            = []

    def get_item(self, persons:List, items:List, frame):
        self.frames_process.append(frame)
        self.persons_process.append(persons)
        self.items_process.append(items)
        while (len(self.frames_process) > self.num_frame_process):
            self.frames_process  = self.frames_process[-1*self.num_frame_process:]
            self.persons_process = self.persons_process[-1*self.num_frame_process:]
            self.items_process   = self.items_process[-1*self.num_frame_process:]

        # START PROCESS
        results = []
        if len(self.persons_process):
            for persons, items, frame in zip(self.persons_process, self.items_process, self.frames_process):
                for person in persons:
                    print(f"id: {person.track_id} ======== holding: {len(person.item_holding)}")
                    if person.in_shelve and person.local=='item_detect':
                        for item in items:
                            if item.inside(Area(frame.img, area_cf).item_detect) and \
                                                    item.overlap_with(person) < 0.6 and item not in person.item_holding:
                                item.cnt_in_area_detect += 1
                                if item.cnt_in_area_detect >= behavior_cf['ratio_frame_to_hold_item']* self.num_frame_process:
                                    person.item_holding.append(item)
                                    break
                    results.append((frame, person))
        return results
    

    def to_pay(self, result_holding:List):
        result = []
        if len(result_holding):
            for frame, person in result_holding:
                if person.stand_in(Area(frame.img, area_cf).payment):
                    person.cnt_in_area_pay += 1
                    if person.cnt_in_area_pay >= behavior_cf['ratio_frame_to_hold_item']*self.num_frame_process:
                        person.in_shelve = False
                        person.paid      = True
                        for item in person.item_holding:
                            if item in person.item_holding:  
                                person.item_holding.remove(item)
                                person.item_paid.append(item)
                # if len(person.item_holding) == 2:
                #     import ipdb; ipdb.set_trace()
                print(f"id: {person.track_id}     hold: {len(person.item_holding)}     paid: {len(person.item_paid)}")


                    # print(f"frame: {frame.id}    person id: {person.track_id}      len items: {len(person.item_holding)}")
                    # if not person in self.result:


        # if len(self.frames_process)==len(self.persons_process)==self.num_frame_process:
        #     for persons_e_frame, items_each_frame, frame in zip(self.persons_process, self.items_process, self.frames_process):
        #         for person in persons_e_frame:
        #             all_keypoint_hand = [kp for kp in person.left_hand_kp]+[kp for kp in person.right_hand_kp]
        #             if person.local == 'shelve':                        
        #                 for item in items_each_frame:
        #                     cnt_kp = 0
        #                     for kp in all_keypoint_hand:

        #                         if kp.in_box(item.box):
        #                             print(item.name_object)
        #                             break
                                #     cnt_kp += 1
                                #     print(cnt_kp)
                                # if cnt_kp > 1/9*self.num_frame_process:
                        #             person.cnt_hand_touch_item += 1
                        #             # break
                        #     # check condition to add item to human
                        #     # if human.flag_in_selection_area and human.flag_hand_over_line and human.cnt_hand_touch_item > 1/6*NUM_FRAME_SET:
                        #     if person.cnt_hand_touch_item > 1/7*self.num_frame_process:
                        #         # print("DA LAY VAT")
                        #         if item not in person.item_holding:
                        #             person.item_holding.append(item)
                        #             person.status = 'holding_item'
                        # if not person in self.result:
                        #     self.result.append(person)    
                        
                        # if len(self.result) != 0:
                        #     for rs_person in self.result:
                        #         if rs_person == person and len(person.item_holding) > len(rs_person.item_holding):
                        #             for rs_item in person.item_holding:
                        #                 if not rs_item in rs_person.item_holding:
                        #                     rs_person.item_holding.append(rs_item)
                
                # for person in self.result:
                #     print(f"-------person_id: {person.track_id} -------holding: {person.item_holding}")

                # print("======================================================================")
        
                            
    # def to_pay(self, rs_num_frame_consecutive, all_human_in_retail):
    #     for rs_frame in rs_num_frame_consecutive:
    #         payment_area = [convert2Real(each[0], each[1], rs_frame.width, rs_frame.height) for each in AREA.payment]
    #         payment_area = [Point(x=each[0], y=each[1]) for each in payment_area]
    #         payment_area = Polygon(payment_area)
    #         for rs_human in rs_frame.humans:
    #             rs_human.cnt_in_area_pay+=1 if rs_human.stand_in(payment_area) else 0
    #             if rs_human.cnt_in_area_pay > 3:
    #                 for human in all_human_in_retail:
    #                     if human == rs_human and len(human.item_holding) > 0:
    #                         human.was_paid = True
    #                         break
    #     for human in all_human_in_retail:
    #         print(f"-------human_id: {human.track_id} -------was paid: {human.was_paid}   -------holding: {human.item_holding}")

    #     print("======================================================================")