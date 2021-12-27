import cv2
import json
import glob

count = 188701 # T-brain already crop 188701 words
g = glob.glob("ReCTS_img/*jpg") # ReCTS folder
g.sort()

for path in g:
    print(path)
    img = cv2.imread(path)
    with open(path.replace("img", "gt").replace("jpg", "json"), encoding="utf-8") as json_file:
        data = json.load(json_file)
        check = "chars" in data
        if check == False: continue
        for char in data["chars"]:
            count += 1
            p = char["points"]
            x1 = p[0]; x2 = p[4]; y1 = p[1]; y2 = p[5]
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("data_tbrain_and_AE/train/test/"+"word_{}.png".format(count), crop_img)
            with open("data_tbrain_and_AE/train/gt.txt", 'a', encoding="utf-8") as txt:
                txt.write("test/word_{}.png {}\n".format(count, char["transcription"]))
    print("Done {}, now counts to {}.".format(path, count))
