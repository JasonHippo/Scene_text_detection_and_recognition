import cv2
import json
import os

# 0:中文字串 1:中文字元 2:英數字串 3:中英數混和字串 4:中文單字字串 5:其他 255:Don't care

count = 0
count_v = 0; count_h = 0
total = 0
for i in range(1, 15188+1):
    #img = cv2.imread("./train/img/img_"+str(i)+".jpg")
    with open('./train/json/img_'+str(i)+'.json', encoding="utf-8") as json_file:
        data = json.load(json_file)
        for shape in data["shapes"]:
            if shape["group_id"] == 1 or shape["group_id"] == 2 or shape["group_id"] == 4:
                count+=1
                p = shape["points"]
                y1 = min(p[0][1], p[1][1], p[2][1], p[3][1])
                y2 = max(p[0][1], p[1][1], p[2][1], p[3][1])
                x1 = min(p[0][0], p[1][0], p[2][0], p[3][0])
                x2 = max(p[0][0], p[1][0], p[2][0], p[3][0])
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                if x2 > int(data["imageWidth"]): x2 = int(data["imageWidth"])
                if y2 > int(data["imageHeight"]): y2 = int(data["imageHeight"])
                # 2 classes: 0 for ch, 1 for engNum
                if shape["group_id"] == 1 or shape["group_id"] == 4: class_name = 0
                else: class_name = 1
                b_w = (x2 - x1) / int(data["imageWidth"])
                b_h = (y2 - y1) / int(data["imageHeight"])
                x_c = ((x2 - x1) / 2 + x1) / int(data["imageWidth"])
                y_c = ((y2 - y1) / 2 + y1) / int(data["imageHeight"])
                with open("../datasets/high_level/labels/train/img_{}.txt".format(str(i)), 'a') as txt:
                    txt.write("{} {} {} {} {}\n".format(class_name, x_c, y_c, b_w, b_h))
        total+=count
        print("img_"+str(i)+" found:", count)
        count = 0
            

