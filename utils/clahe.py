import cv2
import os
import numpy as np
import argparse

def clahe_operate(src_path,trg_path):
    
    folder = os.listdir(src_path)
    
    for i in folder:
        img = cv2.imread(src_path+i)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        clached_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(trg_path+"/"+i, clached_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='the path of original image')
    parser.add_argument('--dst_path', help='the path of result image')
    opt = parser.parse_args()
    clahe_operate(opt.src_path,opt.dst_path)