# Scene_text_detection_and_recognition

## Stage One detection
We use yolov5 to capture the text of the scene.

YoloV5 extracts two types of objects from the scene, the first type is Chinese character and the second type is English\Numeric string or character.

### requirements
```
pip install -r yolov5/requirements.txt
```

### Train
```
python train.py --img 1365 --rect --batch 8 --epochs 300 --data data/high_level.yaml --weights yolov5x6.pt --device 0
```


### Test

Donwload our Training weights to test on private datasets (https://drive.google.com/drive/folders/1NkuSVJcCduJ1YiDAhk2xj4yzkRxn0CWs?usp=sharing)
```
python detect.py --source datasets/high_level/images/private/ --weights runs/train/exp4/weights/best.pt --img 1408 --save-txt --save-conf  --conf-thres 0.7 --iou-thres 0.45 --augment
```
