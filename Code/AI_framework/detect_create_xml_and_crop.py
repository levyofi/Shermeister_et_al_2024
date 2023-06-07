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

from pascal_voc_writer import Writer

#This function gets detections from the lizard detection model and return the classifications and their scores for each detection
def getDetectionLabels(detections, image):
  thermal_labels = []
  thermal_scores = []
  color_labels = []
  color_scores = []
  
  #run the classification models for each detection
  for detection in detections:
    #get the detection coordinates
    left, top, right, bottom = detection.left, detection.top, detection.right, detection.bottom
    #crop the lizard
    im1 = image.crop((left, top, right, bottom))  
    #convert to fastai image 
    img_tensor = T.ToTensor()(im1)
    img_fastai = fastai.vision.Image(img_tensor)
    #get thermoregulation prediction and save results
    pred_class, pred_idx, outputs = classifier_thermo.predict(img_fastai)
    thermal_labels.append(str(pred_class))
    thermal_scores.append(outputs[pred_idx].item())
    #get marking-color prediction and save results    
    pred_class, pred_idx, outputs = classifier_color.predict(img_fastai)
    color_labels.append(str(pred_class))
    color_scores.append(outputs[pred_idx].item())
  
  #return results
  return thermal_labels, thermal_scores, color_labels, color_scores

#This function gets detections and their predicted classifications for an image, saves it in a dataframe and write 
def saveBoxes(df, detections, img, thermal_labels, thermal_scores, color_labels, color_scores):
  
  #get the global names of output files
  global txt_file_name, xml_file_name

  #get creation date of the analyzed image
  strt = datetime.fromtimestamp(os.path.getmtime(file_name)).strftime('%Y-%m-%d-%H-%M-%S')
  print("created: %s" % strt)
  
  #get the camera name based on the path 
  folders = file_name.split("/")  
  camera = folders[5]

  #get the image size
  height, width = img.shape[:2]
  
  #define the xml writer
  writer=Writer(file_name, width, height)

  #for each detection - write the information to the xml and the output table
  for i, detection in enumerate(detections):
    thermal_label = thermal_labels[i]
    color_label = color_labels[i]
    xmin, ymin, xmax, ymax = detection.left, detection.top, detection.right, detection.bottom
    df = df.append(
      {'thermo_label': thermal_label, 'thermo_confidence': thermal_scores[i] * 100,
       'color_label': color_label, 'color_confidence': color_scores[i] * 100, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
       'filename': file_name, 'timestamp': strt, 'camera': camera}, ignore_index=True)
    writer.addObject('lizard', xmin, ymin, xmax, ymax)
  
  #loop finished - write the xml file
  writer.save(xml_file_name)
  
  #return the table
  return df


#This function gets an image, lizard detections and their predicted classifications for an image, draw boxes with the classifications and their scores around each detected lizard and returns the modified image 
def cvDrawBoxes(detections, img, thermal_labels, thermal_scores, color_labels, color_scores):
  #run for each detection
  for i, detection in enumerate(detections):
    col = [255, 0, 0] #red
    #get detection coordinates
    xmin, ymin, xmax, ymax = detection.left, detection.top, detection.right, detection.bottom
    # left top coordinate
    pt1 = (xmin, ymin)
    # right bottom coordinate
    pt2 = (xmax, ymax)
    # draw a green box arounf the detection
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)
    #write the thermoregulation classification and confidence score above the box
    cv2.putText(img,
                thermal_labels[i] +
                " [" + str(round(thermal_scores[i] * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2,
                col, 2)
    #write the color classification and confidence score below the box
    cv2.putText(img,
                color_labels[i] +
                " [" + str(round(color_scores[i] * 100, 2)) + "]",
                (pt1[0], pt2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2,
                col, 2)
                
  #return the modified image
  return img


def mytorch():
  global detected_file_name, file_name, IM_SIZE
  df = pd.DataFrame(columns=['thermo_label', 'thermo_confidence', 'color_label', 'color_confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'filename', 'timestamp', 'camera'])
  image_bgr = cv2.imread(file_name)  # use: detect(,,imagePath,)
  if not image_bgr is None:
      image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

      im_pil = Image.fromarray(image)
      detections = detector.predict(im_pil, threshold=0.50)

      #processDetections(detections)
      if len(detections["det_bboxes"]) > 0:
        thermal_labels, thermal_scores, color_labels, color_scores = getDetectionLabels(detections["det_bboxes"], im_pil)

        #create crops
        im_pil = Image.fromarray(image)
        detections = detector.predict(im_pil, threshold=0.50)
        cropped_file_name_end = file_name.replace("/", "_")
        cropped_file_name_end = os.path.splitext(cropped_file_name_end)[0]
        detectio_id = 0
        for detection in detections['det_bboxes']:
          if detection.score > 0.50:
            detectio_id += 1
            # crop and save the cropped part
            left, top, right, bottom = detection.left, detection.top, detection.right, detection.bottom
            im1 = im_pil.crop((left, top, right, bottom))  # cam6 - 1122, 1192 or cam4,cam8, (972, 1038)
            cropped_file_name = "_"+ thermal_labels[detectio_id-1]+"_"+color_labels[detectio_id-1]+cropped_file_name_end + ".jpg"
            im1.save(os.path.join(dir_name, str(detectio_id) + cropped_file_name),
                     "JPEG")  # im1.show()
        #draw boxes and save xml
        image = cvDrawBoxes(detections["det_bboxes"], image, thermal_labels, thermal_scores, color_labels, color_scores)
        df = saveBoxes(df, detections["det_bboxes"], image, thermal_labels, thermal_scores, color_labels, color_scores)
        #show results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        cv2.imwrite(detected_file_name, image)
  df.to_csv(txt_file_name, mode='w+', header=True)

print(f"TorchVision: {torchvision.__version__}")
which_processor()

# set the path to the images to crop and create annotations
DATA_PATH = "path the the folder with the images"
IM_SIZE = 3000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

path = Path(DATA_PATH)

# load the preliminary model
detector = DetectionLearner.from_saved_model("lizard_detector_folder", path="full path to folder",) #e.g. DetectionLearner.from_saved_model("lizard_detector", path="/data",)

print('Finished loading model')


#get the files to scan
file_listing = glob.glob(os.path.join(path,'**','IMAG*.JPG'), recursive=True) #os.listdir(path)
#loop each file
for file_name in tqdm(file_listing, total=len(file_listing), unit="files"):
  #skip files that are not JPG files
  if file_name.endswith(".JPG"):
    print(file_name)
    #get file base name and directory name
    base_name = os.path.basename(file_name)
    dir_name = os.path.dirname(file_name)
    
    #### give names to output files
    # the lock file is used to make sure other processes (if choosin to run in parallel) don't process the same file
    lock_file_name = os.path.join(dir_name, "lock_" + base_name)
    # the detected file will save the image with the marked detections 
    detected_file_name = os.path.join(dir_name, "detection_" + base_name)
    # the text file will save the table with the detections 
    txt_file_name = os.path.join(dir_name, base_name + '.txt')
    # the xml file will save annotations with the detections 
    xml_file_name = os.path.join(dir_name, os.path.splitext(file_name)[0] + '.xml')
    #skip if there is already a table and detection file for that file 
    if (not os.path.isfile(txt_file_name)) and (not os.path.isfile(detected_file_name)):
    	# check if a lock file exist - this means that another process is currently analysing the file
		if os.path.exists(lock_file_name):
			print("file is currently processes by another tracker")
		else: # no lock file
		    # create a lock file
            open(lock_file_name, 'a').close() 
            #analyze image
            mytorch()
            #remove lock file
            if os.path.exists(lock_file_name):
              os.remove(lock_file_name)

print("The End")
