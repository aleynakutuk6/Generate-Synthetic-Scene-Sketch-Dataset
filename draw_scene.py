from utils.sketch import *
import copy
import numpy as np
import cv2


def draw_sketch(
    sketch: np.ndarray, save_path: str=None, margin: int=10, keep_size: bool=False,
    is_absolute: bool=False, max_dim: int=512, white_bg: bool=False):
    
    assert max_dim > 2*margin # pads 50 to each side
    
    if white_bg:
        canvas = np.full((max_dim, max_dim, 1), 255, dtype=np.uint8)
        fill_color = (0, 0, 0)
    else:
        canvas = np.zeros((max_dim, max_dim, 1), dtype=np.uint8)
        fill_color = (255, 255, 255)
    
    
    if not is_absolute:
        abs_sketch = relative_to_absolute_customized(copy.deepcopy(sketch))
    else:
        abs_sketch = copy.deepcopy(sketch)
    
    if not keep_size:
        xmin, ymin, xmax, ymax = get_absolute_bounds_customized(abs_sketch)
        abs_sketch[:,0] -= xmin
        abs_sketch[:,1] -= ymin
        abs_sketch = normalize_to_scale_customized(
            abs_sketch, is_absolute=True, scale_factor=max_dim-2*margin)
        if abs_sketch is None:
            return None
        abs_sketch[:,:2] += margin # pads margin px to top and left sides
    
    for i in range(1, abs_sketch.shape[0]):
        if abs_sketch[i-1, -1] > 0.5: continue # stroke end
        px, py = int(abs_sketch[i-1, 0]), int(abs_sketch[i-1, 1])
        x, y   = int(abs_sketch[i, 0]), int(abs_sketch[i, 1])
        canvas = cv2.line(canvas, (px, py), (x, y), color=fill_color, thickness=2)
    
    if save_path is not None:
        cv2.imwrite(save_path, canvas)
    
    return canvas
 
    
def draw_scene_with_cv2(data, png_filename=None, obj_points=None, factor=0.0005):

    min_x, max_x, min_y, max_y = get_absolute_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    img = np.zeros((int(dims[1]), int(dims[0]), 3), dtype=np.uint8)
    img.fill(255)

    # black , blue, red, green, yellow, magenta, brown, pink
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (120, 255, 255),
              (255, 0, 255), (139, 69, 19), (255, 20, 147), (0, 0, 100)]
    color_id = 0
    thickness = 2
    prev_x, prev_y = None, None
    stroke_cnt = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor + 25
        y = float(data[i, 1]) / factor + 25
        next_x = int(x - min_x)
        next_y = int(y - min_y)
        if prev_x is None:
            prev_x = next_x
            prev_y = next_y
            continue
            
            
        img = cv2.line(img, (prev_x, prev_y), (next_x, next_y), colors[color_id], thickness)
        lift_pen = data[i, 2]

        if lift_pen == 1:
            prev_x = None
            stroke_cnt += 1
            if obj_points is not None and stroke_cnt in obj_points:
                color_id += 1
                color_id %= len(colors)

        else:
            prev_x = next_x
            prev_y = next_y
            
    if png_filename is not None:
        cv2.imwrite(png_filename, img)

    return img
