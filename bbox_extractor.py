import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import imageio

from PIL import Image


def plot_bounding_box(img_dir, filename, sketch_bboxes, qd_class_ids):

    
    img = imageio.imread(os.path.join(img_dir, filename))

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'purple', 'pink', 'brown', 'white',
              'cyan', 'magenta', 'tan']
    color_index = 0
    x_y_lst = []
    
    for i, bbox in enumerate(sketch_bboxes[1:]):
        x, y, x_max, y_max = bbox
        w = x_max - x 
        h = y_max - y
        bbox_area = (x_max - x) * (y_max - y) 
        found_cls = qd_class_ids[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=0.8, edgecolor=colors[color_index], facecolor='none')

        if [x, y] in x_y_lst or [x, y] == [316, 54]:
            ax.text(x + 3*w/4, y-4, found_cls,
                    bbox={'facecolor': 'black', 'alpha': 0, 'edgecolor': 'none'},
                    color=colors[color_index])

        else:
            ax.text(x, y-4, found_cls,
                    bbox={'facecolor': 'black', 'alpha': 0, 'edgecolor': 'none'},
                    color=colors[color_index])

        ax.add_patch(rect)
        color_index = (color_index + 1) % len(colors)
        x_y_lst.append([x, y])

    plt.savefig(os.path.join(img_dir, "scene_with_bboxes.png"), dpi=500)
    # plt.show()
    

data_path = "coco-records-latest/valid"
img_id = 473199
img_dir = os.path.join(data_path, str(img_id))
filename = "0_scene.png"
with open(os.path.join(img_dir, "data_info.json"), "r") as f:
    data_info = json.load(f)

sketch_bboxes = data_info["sketch_bboxes"]
qd_class_ids = data_info["qd_class_ids"]
scene_x_min, scene_y_min, scene_x_max, scene_y_max = sketch_bboxes[0]


plot_bounding_box(img_dir, filename, sketch_bboxes, qd_class_ids)

