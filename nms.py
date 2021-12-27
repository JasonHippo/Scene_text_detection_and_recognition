import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.ops import nms
import os

def convert(string,img_heigh,img_width):
    # tmp = [[],[]]
    a = string.split()
    cx = img_width *float(a[1])
    cy = img_heigh*float(a[2])
    w = img_width*float(a[3])
    h = img_heigh*float(a[4])
    lbx = cx-0.5*w
    lby = cy+0.5*h
    return [[lbx,lby,lbx+w,lby+h],float(a[5]),float(a[0])]

def convertb(list,img_heigh,img_width,type):
    cx =  (list[0] +0.5*list[2])/img_width
    cy =  (list[1] -0.5*list[3])/img_heigh
    w = list[2]/img_width
    h = list[3]/img_heigh
    return str(int(type))+ " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h)

if __name__ == "__main__":
    label1_path = "./label1/"
    label2_path = "./label2/"
    img_path = "./img/"
    img_save_path = "./nms_outimg/"
    folder = os.listdir(label1_path)
    for k in folder:
        f1 = open(label1_path+k,'r')
        f2 = open(label2_path+k,'r')
        filename = k.split(".")
        image = plt.imread(img_path+filename[0]+".jpg")
        print(image.shape)    
        all = []
        alls = []
        allc = []
        for line in f1.readlines():
            c = convert(line,image.shape[0],image.shape[1])
            all.append(c[0])
            alls.append(c[1])
            allc.append(c[2])
        for line in f2.readlines():
            c = convert(line,image.shape[0],image.shape[1])
            all.append(c[0])
            alls.append(c[1])
            allc.append(c[2])
        
        boxes = torch.tensor(all,dtype=torch.float32)
        score = torch.as_tensor(alls)
        n = nms(boxes=boxes,scores=score,iou_threshold=0.5)
        select = n.tolist()
        result = []
        for i in range(len(select)):
            s = convertb([all[select[i]][0], all[select[i]][1], all[select[i]][2]-all[select[i]][0], all[select[i]][3]-all[select[i]][1]],image.shape[0],image.shape[1],allc[select[i]])
            result.append(s)
        
        f = open("./nms_output/"+k,'w')
        for i in range(len(result)):    
            f.write(result[i]+'\n')
        f.close()

        