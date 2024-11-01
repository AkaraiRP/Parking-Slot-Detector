import pickle

from skimage.transform import resize
import numpy as np
import cv2

MODEL = pickle.load(open("model/model.p", "rb"))

class NoSourceFound(Exception):
    pass


def empty_or_not(spot_bgr):
    # Basic model interaction, crops the spot and the model predicts if empty or not.
    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return True
    else:
        return False
    

def get_spot_from_points(points) -> list:
    # Gets the coords of the 4 mouse clicks and transforms it into a rectangle.
    x = []
    y = []

    for i in range(0, 4):
        x.append(points[i][0])
        y.append(points[i][1])

    x1 = min(x)         # Left
    y1 = min(y)         # Top
    w = max(x) - x1     # Right
    h = max(y) - y1     # Bottom

    return [x1, y1, w, h]


def get_parking_spots_bboxes(connected_components):
    # Uses cv2 connected components to get the spot.
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])
        print([x1, y1, w, h])

    return slots

