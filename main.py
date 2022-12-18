import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import label, regionprops
from pathlib import Path


def prepare_img(r_img, scale=0.1):
    height = int(r_img.shape[0] * scale)
    width = int(r_img.shape[1] * scale)
    
    # сжимаем, чтобы мой ноутбук не умер окончательно
    c_img = cv2.resize(r_img, (width, height), interpolation=cv2.INTER_AREA)
    
    # 2gray
    g_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    #threshhold
    blur = cv2.GaussianBlur(g_img, (5,53), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return th


def count_pencils(img):
    cnt = 0
    labeled = label(img)
    props = regionprops(labeled)
    for p in props:
        # игнорируем неровный лист >:O
        if ((p.bbox[0] == 0 and p.bbox[1] == 0) or
            (p.bbox[0] == 0 and p.bbox[3] == img.shape[1]) or
            (p.bbox[2] == img.shape[0] and p.bbox[3] == img.shape[1]) or
            (p.bbox[2] == img.shape[0] and p.bbox[1] == 0)):
            continue
        # center = (p.bbox[0] + (p.bbox[2] - p.bbox[0]) / 2, p.bbox[1] + (p.bbox[3] - p.bbox[1]) / 2)
        # dist = np.linalg.norm(np.array(center) - np.array(p.centroid))
        # print(p.label, dist, p.eccentricity)
        
        # у всех карандашей эксцентриситет больше 0.99, но даем себе право на ошибку
        if p.eccentricity > 0.98:
            cnt += 1
    
    return cnt


g_cnt = 0

images = Path('images')
for img in images.iterdir():
    r_img = cv2.imread(str(img))
    img = prepare_img(r_img, 0.1)
    g_cnt += count_pencils(img)

print(g_cnt) # 21