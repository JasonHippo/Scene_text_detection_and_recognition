# Preporcessing data

## Download the [T-brain](https://tbrain.trendmicro.com.tw/Competitions/Details/19) and [ReCTS](https://rrc.cvc.uab.es/?ch=12) Training datasets

Put the T-brain images and json to data/t-brain/img and data/t-brain/json, Put the ReCTS images and json to data/rects/img and data/rects/json

Run the following commands to create txt for yolo training (The resuls will in the data/t-brain/yolotxt and  data/rects/yolotxt folder)
```
python label2txt_tbrain.py
python label2txt_rects.py
```

Move all images to the yolov5/datasets/high_level/train/images and all txts to the yolov5/datasets/high_level/train/labels.

You should also create val and test images and json . It depends on ur split train/val/test strategy.

Done!
