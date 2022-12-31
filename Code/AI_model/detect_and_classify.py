import sys

sys.path.append("../computervision-recipes")
sys.path.append("../computervision-recipes/.")

import glob
import os
from pathlib import Path
from PIL import Image
import torch
import torchvision
from utils_cv.detection.model import (
  DetectionLearner,
  get_pretrained_fasterrcnn,
)
from utils_cv.common.gpu import which_processor, is_windows
import cv2
from tqdm import tqdm
import pandas as pd
import datetime
from datetime import datetime
import torchvision.transforms as T

import fastai
from fastai.vision import (
    models, to_np, load_learner
)
# pip

def getDetectionLabels(detections, image):
  thermal_labels = []
  thermal_scores = []
  color_labels = []
  color_scores = []
  for detection in detections:
    left, top, right, bottom = detection.left, detection.top, detection.right, detection.bottom
    im1 = image.crop((left, top, right, bottom))  # cam6 - 1122, 1192 or cam4,cam8, (972, 1038)
    img_tensor = T.ToTensor()(im1)
    img_fastai = fastai.vision.Image(img_tensor)
    pred_class, pred_idx, outputs = classifier_thermo.predict(img_fastai)
    thermal_labels.append(str(pred_class))
    thermal_scores.append(outputs[pred_idx].item())
    pred_class, pred_idx, outputs = classifier_color.predict(img_fastai)
    color_labels.append(str(pred_class))
    color_scores.append(outputs[pred_idx].item())
  return thermal_labels, thermal_scores, color_labels, color_scores

def saveBoxes(df, detections, thermal_labels, thermal_scores, color_labels, color_scores):
  global txt_file_name

  # print("last modified: %s" % time.ctime(os.path.getmtime(file_name)))
  strt = datetime.fromtimestamp(os.path.getmtime(file_name)).strftime('%Y-%m-%d-%H-%M-%S')
  print("created: %s" % strt)
  folders = file_name.split("/")
  camera = folders[5]

  for i, detection in enumerate(detections):
    thermal_label = thermal_labels[i]
    color_label = color_labels[i]
    xmin, ymin, xmax, ymax = detection.left, detection.top, detection.right, detection.bottom
    df = df.append(
      {'thermo_label': thermal_label, 'thermo_confidence': thermal_scores[i] * 100,
       'color_label': color_label, 'color_confidence': color_scores[i] * 100, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
       'filename': file_name, 'timestamp': strt, 'camera': camera}, ignore_index=True)
  return df

def cvDrawBoxes(detections, img, thermal_labels, thermal_scores, color_labels, color_scores):
  for i, detection in enumerate(detections):
    col = [255, 0, 0]
    xmin, ymin, xmax, ymax = detection.left, detection.top, detection.right, detection.bottom
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)
    cv2.putText(img,
                thermal_labels[i] +
                " [" + str(round(thermal_scores[i] * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2,
                col, 2)
    cv2.putText(img,
                color_labels[i] +
                " [" + str(round(color_scores[i] * 100, 2)) + "]",
                (pt1[0], pt2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2,
                col, 2)
  return img

def remove_strip(frame):
  h, w = frame.shape[:2]
  if (h>3000):
    clean_frame = frame[1:3100, ...]
  else:
    clean_frame = frame[1:2100,... ]
  return(clean_frame)
# Open the image files.

def mytorch():
  global detected_file_name, file_name, IM_SIZE
  df = pd.DataFrame(columns=['thermo_label', 'thermo_confidence', 'color_label', 'color_confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'filename', 'timestamp', 'camera'])
  image_bgr = cv2.imread(file_name)  # use: detect(,,imagePath,)
  if not image_bgr is None:
      image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            #frame_resized = cv2.resize(frame_rgb,
            #                           (IM_SIZE ,
            #                            IM_SIZE ),
            #                           interpolation=cv2.INTER_LINEAR)

      im_pil = Image.fromarray(image)
      detections = detector.predict(im_pil, threshold=0.50)

      #processDetections(detections)
      if len(detections["det_bboxes"]) > 0:
        thermal_labels, thermal_scores, color_labels, color_scores = getDetectionLabels(detections["det_bboxes"], im_pil)
        image = cvDrawBoxes(detections["det_bboxes"], image, thermal_labels, thermal_scores, color_labels, color_scores)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        df = saveBoxes(df, detections["det_bboxes"],thermal_labels, thermal_scores, color_labels, color_scores)
        cv2.imwrite(detected_file_name, image)
  df.to_csv(txt_file_name, mode='w+', header=True)

print(f"TorchVision: {torchvision.__version__}")
which_processor()

#DATA_PATH = "/data3/Ben/Annotations/Empty"  # unzip_url(Urls.fridge_objects_path, exist_ok=True)
#DATA_PATH = "/data3/Ben/data/Images"
DATA_PATH = "/data/lizard_classification_model/model paper - 03.02.2022"
IM_SIZE = 3000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

path = Path(DATA_PATH)

# Pre-trained Faster R-CNN model
detector = DetectionLearner.from_saved_model("lizard_detector_14-08_3_0.71 0.001", path="/data3",)

classification_model_path = '/data/lizard_classification_model'
classification_thermo_model_file = 'lizard_model_ben_5_0.96.pkl'
classifier_thermo = load_learner(classification_model_path, classification_thermo_model_file)
labels_thermo = classifier_thermo.data.classes

classification_color_model_file = 'lizard_model_color_ben__August2_0.97.pkl'
classifier_color = load_learner(classification_model_path, classification_color_model_file)
labels_color = classifier_color.data.classes

print('Finished loading model')

if __name__ == "__main__":
    file_listing = glob.glob(os.path.join(path,'**','IMAG*.JPG'), recursive=True) #os.listdir(path)
    for file_name in tqdm(file_listing, total=len(file_listing), unit="files"):
        if file_name.endswith(".JPG"):
            # file_name = "/data/urban_behavior/videos/test/bdd100k/videos/100k/test/cabe1040-c59cb390.mov"
            print(file_name)
            base_name = os.path.basename(file_name)
            dir_name = os.path.dirname(file_name)
            lock_file_name = os.path.join(dir_name, "lock_" + base_name)
            #rotated_file_name = os.path.join(dir_name, "transposed_" + base_name)
            detected_file_name = os.path.join(dir_name, "detection_" + base_name)
            txt_file_name = os.path.join(dir_name, os.path.basename(file_name) + '.txt')
            if (not os.path.isfile(txt_file_name)) and (not os.path.isfile(detected_file_name)):
                if os.path.exists(lock_file_name):
                    print("file is currently processes by another tracker")
                else:
                    open(lock_file_name, 'a').close()
                    mytorch()
                    os.remove(lock_file_name)

#detector.load(path="lizard_detector_19_0.49 0.001")
#video_file_name = "/data3/Danny/Camera/15.9/3l/ZETTA_2020-09-13_15.25.42.AVI"
#detected_file_name = "/data3/Danny/Camera/17.9/3l/detect_ZETTA_2020-09-16_08.18.56.AVI"
#load a video
print("The End")
