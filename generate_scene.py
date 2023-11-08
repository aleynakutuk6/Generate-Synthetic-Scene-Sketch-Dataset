from utils.hparams import *
from utils.sketch import *
from rdp import rdp
from tqdm import tqdm
from draw_scene import *

import numpy as np
import copy


def viz_given_scene(scene_img, save_path):

    sketch = np.asarray(scene_img["scene_strokes"])
    obj_points = scene_img['object_divisions']

    img = draw_scene_with_cv2(sketch, save_path, obj_points, 1.0)

    
def simplification(data, scale_factor=255.0):
    
    data = np.asarray(data, dtype=float)
    
    # get bounds of sketch
    min_x, max_x, min_y, max_y = get_absolute_bounds(data)
    scale = scale_factor / max([max_x - min_x, max_y - min_y, 1])
    
    # align the drawing to the top-left corner, to have minimum values of 0.
    data[:, 0] -= min_x
    data[:, 1] -= min_y
    
    """
    for i in range(1, data.shape[0]):
        data[i, 0] -= data[0, 0]
        data[i, 1] -= data[0, 1]
    data[0, 0] = 0.0
    data[0, 1] = 0.0
    """
    
    # uniformly scale the drawing, to have a maximum value of 255.
    data[:, :2] *= scale
    
    # simplify all strokes using the Ramer–Douglas–Peucker algorithm with an epsilon value of 2.0.
    sketch_relative = to_relative(data)
    sketch_rdp = apply_RDP(sketch_relative)  # apply RDP
    simplified = normalize(sketch_rdp)

    return simplified
  

def read_quickdraw_npz(filepath: str, partition: str=None, idx=None):

    assert partition is not None or (partition is None and idx is None)
    assert partition is None or partition in ['train', 'valid', 'test']
    assert idx is None or idx == "random" or idx >= 0
    
    if not os.path.isfile(filepath):
        raise ValueError(f"No NPZ file exists in: {filepath}")
    
    sketches = np.load(filepath, allow_pickle=True, encoding="bytes")
    
    def clean_sketch(sketch: np.ndarray) -> np.ndarray:
        sk_new = sketch.astype(float)
        if sk_new[0, -1] == 1:
            sk_new = sk_new[1:, :]
        
        end_checks = sk_new[1:, -1] + sk_new[:-1, -1]
        vals = np.where(end_checks > 1)[0]
        if vals.shape[0] > 0:
            for j in np.flip(vals):
                sk_new = np.delete(sk_new, j+1, axis=0)
        
        return sk_new

    
    if partition is not None:
        sketches = sketches[partition]
        if idx is None:
            return [clean_sketch(sk) for sk in sketches]
        elif idx == "random":
            n_samples = len(sketches)
            idx = random.randint(0, n_samples-1)
            return clean_sketch(sketches[idx]), idx
        else:
            return clean_sketch(sketches[idx])
    else:
        tr = [clean_sketch(sk) for sk in sketches["train"]]
        val = [clean_sketch(sk) for sk in sketches["valid"]]
        test = [clean_sketch(sk) for sk in sketches["test"]]
        return tr, val, test


def generate_scene_from_single_img(d):
    generated_scene = {}
    info, y = d["data"], d["label"]
    img_id = info["image_id"]

    scene_sketch, sketch_divisions, sketch_bboxes = [], [0], []
    o_idx = 0
    for obj in info["objects"]:
        o_idx += 1
        # Spatial Information
        x_i, y_i, h_i, w_i = obj["x"], obj["y"], obj["h"], obj["w"]

        x_c_i = x_i + w_i / 2
        y_c_i = y_i + h_i / 2
        ratio_i = h_i / w_i
        
        # Identity Information
        sketch_temp = obj["stroke-3"]
        sketch = copy.deepcopy(sketch_temp)
        sketch = relative_to_absolute(sketch)

        x_vals = sketch[:, 0]
        y_vals = sketch[:, 1]
        min_x, min_y, max_x, max_y = min(x_vals), min(y_vals), max(x_vals), max(y_vals)
        h_j = max_y - min_y
        w_j = max_x - min_x
        x_c_j = (min_x + max_x) / 2
        y_c_j = (min_y + max_y) / 2
        ratio_j = h_j / w_j
        if ratio_i < ratio_j:
            rate = h_i / h_j
        else:
            rate = w_i / w_j
        sketch[:, :2] *= rate
        x_c_j *= rate
        y_c_j *= rate

        sketch[:, 0] += x_c_i - x_c_j
        sketch[:, 1] += y_c_i - y_c_j
        
        xmin, ymin, xmax, ymax = get_absolute_bounds(sketch)
        
        scene_sketch.extend(sketch.tolist())
        num_sk_strokes = int(np.sum(sketch[..., -1] == 1))
        sketch_divisions.append(sketch_divisions[-1] + num_sk_strokes)
        sketch_bboxes.append([xmin, ymin, xmax, ymax])
        
    generated_scene = {"scene_strokes": scene_sketch,
                       "object_divisions": sketch_divisions,
                       "sketch_bboxes": sketch_bboxes}
    
    return generated_scene
