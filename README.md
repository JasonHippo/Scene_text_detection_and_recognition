# Scene_text_detection_and_recognition
[![PWC](./image/t-brain_icon.png)](https://tbrain.trendmicro.com.tw/Competitions/Details/19)

## Rank 3 on Public and Private
![Public](./image/Public.png)
![Private](./image/Private.png)

## Stage One detection (Remember to modify the path of file to yours. e.g. cd yolov5)
We use [yolov5](https://github.com/ultralytics/yolov5) to capture the text of the scene.

YoloV5 extracts two types of objects from the scene, the first type is Chinese character and the second type is English\Numeric string or character.

### requirements
```
pip install -r yolov5/requirements.txt
```

### Train 
```
python train.py --img 1365 --rect --batch 8 --epochs 300 --data yolov5/data/high_level.yaml --weights yolov5/yolov5x6.pt --device 0
```


### Test

Donwload our Training weights to test on private datasets (https://drive.google.com/drive/folders/1NkuSVJcCduJ1YiDAhk2xj4yzkRxn0CWs?usp=sharing)
```
python detect.py --source yolov5/datasets/high_level/images/private/ --weights yolov5/runs/train/exp4/weights/best.pt --img 1408 --save-txt --save-conf  --conf-thres 0.7 --iou-thres 0.45 --augment
```

## Stage Two recognition (Remember to modify the path of file to yours. e.g. cd deep-text-recognition-benchmark)
We use [deep-text-recognitiom-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) to recognize the text of the detected image.

There are two models for this step, the first model is for Chinese character and the second is for English\Numeric string or character.

For the first model, we train on the training dataset of T-brain and ReCTS.

For the second model, we use the pretrained of deep-text-recognitiom-benchmark.

### Prepare the dataset 
We first crop the chinese character from images. 
```
python processing/crop_chinese_ch.py 
```
Here are the characters we already cropped: (https://drive.google.com/file/d/1flVnxIIRgn2akANQ1Jhix-AbFHrQpYaA/view?usp=sharing). 
Download it and put it on the root of deep-text-recognition-benchmark.

Then we create lmdb datasets, The output will default save at [./result] folder.
```
python create_lmdb_dataset.py --inputPath data_tbrain_and_AE/val --gtFile data_tbrain_and_AE/val/gt.txt --outputPath result/data_tbrain_and_AE/val
```

### Train
```
python train.py --train_data result/data_tbrain_and_AE/train/ --valid_data result/data_tbrain_and_AE/val/ --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --sensitive
```

### Recognize
if there exist old pred0 and pred1 folder, please run this cmd
```
rm -r pred0 && rm -r pred1
```

Every time you recognize the characters from crop images, please follow below:

1. Create folders for cropping images 
```
mkdir pred0 && mkdir pred1
```
2. Download the weight on (https://drive.google.com/file/d/1PIh6JoZ5rlc0_2itRVRgWjFeQUxp2wTr/view?usp=sharing), unzip it and put it on the root of deep-text-recognition-benchmark.
3. modify --out_csv_name and --label_root to where you want to save and where the .txt files you save which detect by yolo, and run cmd.
```
python recognize.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --sensitive --out_csv_name recog_output/out.csv --label_root ../yolov5/runs/detect/exp/ --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111_1130/best_accuracy.pth 
```
Now you can see the results at the path you set.

### Post processing
For the T-brain competition, we need to follow the below steps:

1. edit the --out_csv_name file, and add below header
```
name,x1,y1,x2,y2,x3,y3,x4,y4,pred
```

2. modify --path to path of --out_csv_name and run the cmd
```
python processing/editResult.py --path recog_output/out.csv
```

You can see the post file at the recog_output/out_post.csv


## Cite

```
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019},
  pubstate={published},
  tppubtype={inproceedings}
}
```
