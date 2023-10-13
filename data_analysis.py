import os
import json




def qd_coco_coverage(class_relationship):

    with open(class_relationship) as clrf:
        sketch_to_coco = json.load(clrf)
    
    cnt = 0
    for class_name, mapped in sketch_to_coco.items():
        if mapped is not None:
            cnt += 1
    print("matched sketch count:", cnt)





class_relationship = 'qd_coco_files/quickdraw_to_coco_v2.json'
qd_coco_coverage(class_relationship)


path = 'coco-records/valid'
cnt = 0

for img in os.listdir(path):
    cnt += 1

print("cnt ", cnt)