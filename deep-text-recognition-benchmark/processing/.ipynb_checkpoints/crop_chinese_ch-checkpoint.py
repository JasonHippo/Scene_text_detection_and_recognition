import cv2
import json

# 0:中文字串 1:中文字元 2:英數字串 3:中英數混和字串 4:中文單字字串 5:其他 255:Don't care

count = 0
total = 0
for i in range(1, 15188+1):
    img = cv2.imread("Z:/He-Hao-Liao/aicup-high/train/img/img_"+str(i)+".jpg")
    with open('Z:/He-Hao-Liao/aicup-high/train/json/img_'+str(i)+'.json', encoding="utf-8") as json_file:
        data = json.load(json_file)
        for shape in data["shapes"]:
            if shape["group_id"] == 1 or shape["group_id"] == 4:
                count += 1
                p = shape["points"]
                y1 = min(p[0][1], p[1][1], p[2][1], p[3][1])
                y2 = max(p[0][1], p[1][1], p[2][1], p[3][1])
                x1 = min(p[0][0], p[1][0], p[2][0], p[3][0])
                x2 = max(p[0][0], p[1][0], p[2][0], p[3][0])
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                if x2 > int(data["imageWidth"]): x2 = int(data["imageWidth"])
                if y2 > int(data["imageHeight"]): y2 = int(data["imageHeight"])
                crop_img = img[y1:y2, x1:x2]
                crop_img = cv2.resize(crop_img, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("C:\jasonhippo\deep-text-recognition-benchmark/data_tbrain_and_AE/train/test/"+"word_{}.png".format(count), crop_img)
                with open("C:\jasonhippo\deep-text-recognition-benchmark\data_tbrain_and_AE/train/gt.txt", 'a', encoding="utf-8") as txt:
                    txt.write("test/word_{}.png {}\n".format(count, shape["label"]))
    print("Done img_{}.png".format(i))