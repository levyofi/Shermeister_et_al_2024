# Code for training an object detection model. In our case, we train it to detect lizards. 
# The code here is an adaptation of the code showen in https://github.com/microsoft/computervision-recipes/blob/staging/scenarios/detection/12_hard_negative_sampling.ipynb
# 
import sys

sys.path.append("../computervision-recipes") # downloaded from 

import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from random import randrange
from typing import Tuple
import torch
import torchvision
from torchvision import transforms
import scrapbook as sb
import numpy as np
from utils_cv.classification.data import Urls as UrlsIC
from utils_cv.common.data import unzip_url, data_path
from utils_cv.detection.data import Urls
from utils_cv.detection.dataset import DetectionDataset, get_transform
from utils_cv.detection.plot import (
  plot_grid,
  plot_boxes,
  plot_pr_curves,
  PlotSettings,
  plot_counts_curves,
  plot_detections
)
from utils_cv.detection.model import (
  DetectionLearner,
  get_pretrained_fasterrcnn,
)
from utils_cv.common.gpu import which_processor, is_windows

# Change matplotlib backend so that plots are shown for windows
if is_windows():
  plt.switch_backend("TkAgg")

print(f"TorchVision: {torchvision.__version__}")
which_processor()

DATA_PATH = "/data3/Ben/Annotations_2022/Lizards"  # unzip_url(Urls.fridge_objects_path, exist_ok=True)
NEG_DATA_PATH = "/data3/Ben/Annotations_2022/Empty/images"  # unzip_url(UrlsIC.fridge_objects_negatives_path, exist_ok=True)
EPOCHS = 3
EPOCHS_HEAD = 4

LEARNING_RATE = 0.001
IM_SIZE = 3000
SAVE_MODEL = True
NEGATIVE_NUM = 3

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

path = Path(DATA_PATH)
os.listdir(path)

#create the detection dataset
data = DetectionDataset(DATA_PATH, train_pct=0.75, batch_size=2,allow_negatives=True )

print(
  f"Training dataset: {len(data.train_ds)} | Training DataLoader: {data.train_dl} \
    \nTesting dataset: {len(data.test_ds)} | Testing DataLoader: {data.test_dl}"
)
matplotlib.use('TkAgg')

#Negative images split into hard-negative mining candidates U, and a negative test set.
#Setting "allow_negatives=True" since the negative images don't have an .xml file with ground truth annotations
neg_data = DetectionDataset(NEG_DATA_PATH, train_pct=0.80,
                            im_dir="", allow_negatives=True,
                            train_transforms=get_transform(train=False))
print(
  f"Negative dataset: {len(neg_data.train_ds)} candidates for hard negative mining and {len(neg_data.test_ds)} test images.")

# Pre-trained Faster R-CNN model
detector = DetectionLearner(data, im_size=IM_SIZE)

#use this line instead of the above if loaded a pre-trained lizard detection model
#detector = DetectionLearner.from_saved_model("lizard_detector_19_0.51 0.001", path="/data3",)

# Record after each mining iteration the validation accuracy and how many objects were found in the negative test set
valid_accs = []
num_neg_detections = []

#initial training
detector.fit(EPOCHS_HEAD, lr=LEARNING_RATE*5, print_freq=30)

# Fine-tune model. After each epoch prints the accuracy on the validation set.
for iteration in range(5):
  detector.fit(EPOCHS, lr=LEARNING_RATE, print_freq=30)
  detector.plot_precision_loss_curves()

  # Get validation accuracy on test set at IOU=0.5:0.95
  acc = float(detector.ap[-1]["bbox"])
  valid_accs.append(acc)

  # Plot validation accuracy versus number of hard-negative mining iterations
  from utils_cv.common.plot import line_graph

  line_graph(
    values=(valid_accs),
    labels=("Validation"),
    x_guides=range(len(valid_accs)),
    x_name="Hard negative mining iteration",
    y_name="mAP@0.5:0.95",
  )

  detections = detector.predict_dl(neg_data.train_dl, threshold=0.2)
  detections[0]

  # Count number of mis-detections on negative test set
  test_detections = detector.predict_dl(neg_data.test_dl, threshold=0.2)
  bbox_scores = [bbox.score for det in test_detections for bbox in det['det_bboxes']]
  num_neg_detections.append(len(bbox_scores))

  # Plot
  from utils_cv.common.plot import line_graph

  line_graph(
    values=(num_neg_detections),
    labels=("Negative test set"),
    x_guides=range(len(num_neg_detections)),
    x_name="Hard negative mining iteration",
    y_name="Number of detections",
  )

  # For each image, get maximum score (i.e. confidence in the detection) over all detected bounding boxes in the image
  max_scores = []
  for idx, detection in enumerate(detections):
    if len(detection['det_bboxes']) > 0:
      max_score = max([d.score for d in detection['det_bboxes']])
    else:
      max_score = float('-inf')
    max_scores.append(max_score)

  # Use the n images with highest maximum score as hard negatives
  hard_im_ids = np.argsort(max_scores)[::-1]
  hard_im_ids = hard_im_ids[:NEGATIVE_NUM]
  hard_im_ids_good = []
  for hard_id in hard_im_ids:
    if max_scores[hard_id] > 0.1:
      hard_im_ids_good.append(hard_id)

  print(hard_im_ids)
  if (len(hard_im_ids_good)>0):
    hard_im_ids = hard_im_ids_good
    hard_im_scores = [max_scores[i] for i in hard_im_ids]
    # hard_im_scores = hard_im_scores[not math.isinf()]

    print(
      f"Indentified {len(hard_im_scores)} hard negative images with detection scores in range {min(hard_im_scores)} to {max(hard_im_scores):4.2f}")

    # Get image paths and ground truth boxes for the hard negative images
    dataset_ids = [detections[i]['idx'] for i in hard_im_ids]
    im_paths = [neg_data.train_ds.dataset.im_paths[i] for i in dataset_ids]
    gt_bboxes = [neg_data.train_ds.dataset.anno_bboxes[i] for i in dataset_ids]

    # # Plot
    # def _grid_helper():
    #     for i in hard_im_ids:
    #         yield detections[i], neg_data, None, None
    # plot_grid(plot_detections, _grid_helper(), rows=1)

    # Add identified hard negatives to training set
    data.add_images(im_paths, gt_bboxes, target="train")
    print(
      f"Added {len(im_paths)} hard negative images. Now: {len(data.train_ds)} training images and {len(data.test_ds)} test images")
    sb.glue("valid_accs", valid_accs)
    sb.glue("hard_im_scores", list(hard_im_scores))
  else:
    print("no negative images were added")
  print(f"Completed {len(valid_accs)} hard negative iterations.")

  # Preserve some of the notebook outputs
  if SAVE_MODEL:
    detector.save("/data3/lizard_detector_" + str(iteration)+"_"+f"{acc:4.2f}"+f"{LEARNING_RATE:6.4}")

print("The End")
