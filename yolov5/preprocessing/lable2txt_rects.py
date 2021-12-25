import os
import json
with open('./train_cache_file.json') as f:
    a = json.load(f)
    
from tqdm import tqdm
count = 0
count_v = 0; count_h = 0
total = 0
for i in tqdm(a):
    with open('./train_json/{}'.format(i['annfile'])) as f:
        data = json.load(f)
        if os.path.isfile('./train_label/{}.txt'.format(i['annfile'][:-5])):
            print(i)
            break
        for shape in data['chars']: 
#             count+=1
            p = shape["points"]
            y1 = min(p[1],p[3],p[5],p[7])
            y2 = max(p[1],p[3],p[5],p[7])
            x1 = min(p[0],p[2],p[4],p[6])
            x2 = max(p[0],p[2],p[4],p[6])
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            if x2 > int(i["width"]): x2 = int(i["width"])
            if y2 > int(i["height"]): y2 = int(i["height"])
            # 2 classes: 0 for ch, 1 for engNum
            if '\u4e00' <= shape['transcription'] <= '\u9fff': 
                class_name = 0
            else: 
                class_name = 1   
            b_w = (x2 - x1) / int(i["width"])
            b_h = (y2 - y1) / int(i["height"])
            x_c = ((x2 - x1) / 2 + x1) / int(i["width"])
            y_c = ((y2 - y1) / 2 + y1) / int(i["height"])
            with open("./train_label/{}.txt".format(i['annfile'][:-5]), 'a') as txt:
                txt.write("{} {} {} {} {}\n".format(class_name, x_c, y_c, b_w, b_h))