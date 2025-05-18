import os, glob
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import hashlib

class LabelEncoder:
    def __init__(self, classes):
        self.class2id = {cls:id for id, cls in enumerate(classes)}
        self.id2class = {id:cls for cls, id in self.class2id.items()}
    
    def __repr__(self):
        return str(self.class2id)


def resize_and_pad_image_cv2(img, target_size=224):
    try:
        h, w, c = img.shape
        if h == target_size and w == target_size:
            return img
        elif h == 0 or w == 0:
            raise Exception("Image dimensions are zero.")
    except:
        print("WARNING: Image dimensions are zero. Falling back to dummy image.")
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Calculate resizing ratio
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    # Resize the image
    img = cv2.resize(img, (new_w, new_h))

    # Calculate padding and pad image if necessary
    pad_h1 = (target_size - new_h) // 2
    pad_h2 = (target_size - new_h) - pad_h1
    pad_w1 = (target_size - new_w) // 2
    pad_w2 = (target_size - new_w) - pad_w1

    img = cv2.copyMakeBorder(img, pad_h1, pad_h2, pad_w1, pad_w2, cv2.BORDER_CONSTANT)

    return img