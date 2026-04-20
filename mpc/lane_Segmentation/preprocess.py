import torch
import cv2
import numpy as np

from dataset import CaptureDataset


def mask_by_hsv(image, target_hsv, tolerance):
    if isinstance(tolerance, int):
        tol_h, tol_s, tol_v = tolerance, tolerance, tolerance
    else:
        tol_h, tol_s, tol_v = tolerance
    h, s, v = target_hsv
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s_min = max(s - tol_s, 0)
    s_max = min(s + tol_s, 255)
    v_min = max(v - tol_v, 0)
    v_max = min(v + tol_v, 255)

    if h - tol_h < 0:
        lower1 = np.array([0, s_min, v_min])
        upper1 = np.array([h + tol_h, s_max, v_max])

        lower2 = np.array([179 + (h - tol_h), s_min, v_min])
        upper2 = np.array([179, s_max, v_max])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif h + tol_h > 179:
        lower1 = np.array([h - tol_h, s_min, v_min])
        upper1 = np.array([179, s_max, v_max])

        lower2 = np.array([0, s_min, v_min])
        upper2 = np.array([(h + tol_h) - 179, s_max, v_max])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h - tol_h, s_min, v_min])
        upper = np.array([h + tol_h, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
    return mask


if __name__ == "__main__":
    ds = CaptureDataset("data/capture")
    print(len(ds))

    yellow_lane = [30, 255, 255]
    white_lane = [0, 0, 255]

    for i in range(len(ds)):
        image, _ = ds.read(i)
        # image = cv2.imread(os.path.join("data/capture/images", f"{i}.png"))
        # if image is None:
        #     print(f"Skipping {i}: missing image")
        #     continue

        yellow_mask = mask_by_hsv(image, yellow_lane, [10, 100, 150])    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        thresh = cv2.bitwise_or(yellow_mask, white_mask)

        h, w = thresh.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        floodfill = thresh.copy()
        for x in range(w):
            if floodfill[0, x] == 255:
                cv2.floodFill(floodfill, mask, (x, 0), 128)

        top_region_mask = (floodfill == 128)
        thresh[top_region_mask] = 0
    
        ds.write_mask(thresh, i)
