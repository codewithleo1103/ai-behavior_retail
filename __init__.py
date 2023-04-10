import yaml

file_ = open(r'./config.yml')
config = yaml.full_load(file_)

area_cf = config['area']

person_detect_cf = config['person_detection']
person_track_cf = config['person_tracking']
pose_cf = config['pose_detection']
behavior_cf = config['behavior']

item_detect_cf = config['item_detection']
item_track_cf = config['item_tracking']

visual_cf = config['visualization']
debug_cf = config['debug']