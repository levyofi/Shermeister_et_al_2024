#This code load the trained models and implement the framework - scans a speficic directory, detects lizards and predict their thermoregulation behavior and marking-color

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
def saveBoxes(df, detections, thermal_labels, thermal_scores, color_labels, color_scores):
  
  #get the global name of output file
  global txt_file_name

  
  #get creation date of the analyzed image
  strt = datetime.fromtimestamp(os.path.getmtime(file_name)).strftime('%Y-%m-%d-%H-%M-%S')
  print("created: %s" % strt)
  
  #get the camera name based on the path 
  folders = file_name.split("/")  
  camera = folders[5]

  #get the image size
  height, width = img.shape[:2]

  #for each detection - write the information to the output table
  for i, detection in enumerate(detections):
    thermal_label = thermal_labels[i]
    color_label = color_labels[i]
    xmin, ymin, xmax, ymax = detection.left, detection.top, detection.right, detection.bottom
    df = df.append(
      {'thermo_label': thermal_label, 'thermo_confidence': thermal_scores[i] * 100,
       'color_label': color_label, 'color_confidence': color_scores[i] * 100, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
       'filename': file_name, 'timestamp': strt, 'camera': camera}, ignore_index=True)
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

#This function runs the framework on a camera trap image. The path of the image is in file_name. The function generates an ouput file with a table of the results for each detected lizard (empty table of no lizards were detected) 
def mytorch():
  # get global variables need for the function
  global detected_file_name, file_name, IM_SIZE
  
  #create the output table
  df = pd.DataFrame(columns=['thermo_label', 'thermo_confidence', 'color_label', 'color_confidence', 'xmin', 'ymin', 'xmax', 'ymax', 'filename', 'timestamp', 'camera'])
  
  #read the image
  image_bgr = cv2.imread(file_name)  
  
  #check if the image was succesfully readen
  if not image_bgr is None:
      #convert to RGB
      image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      
      #convert to PIL format and send to the detection model
      im_pil = Image.fromarray(image)
      detections = detector.predict(im_pil, threshold=0.50)

      #If lizards were detected - continue to classification models
      if len(detections["det_bboxes"]) > 0:
        #get classification predictions
        thermal_labels, thermal_scores, color_labels, color_scores = getDetectionLabels(detections["det_bboxes"], im_pil)
        #draw boxes around each detected lizards with the classifications and their confidence scores
        image = cvDrawBoxes(detections["det_bboxes"], image, thermal_labels, thermal_scores, color_labels, color_scores)
        
        #convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #plot the results - comment out if in command line only
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        
        #add the classifications to the dataframe
        df = saveBoxes(df, detections["det_bboxes"],thermal_labels, thermal_scores, color_labels, color_scores)
        
        #write the modified image with the bounding boxes for user evaluation
        cv2.imwrite(detected_file_name, image)
  
  #write the table to a file
  df.to_csv(txt_file_name, mode='w+', header=True)
## main code
print(f"TorchVision: {torchvision.__version__}")
which_processor()

DATA_PATH = "path the the folder with the images"
IM_SIZE = 3000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

path = Path(DATA_PATH)

# load the detector model
detector = DetectionLearner.from_saved_model("lizard_detector_folder", path="path to the folder",)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

#### load the classification models
# set paths to models
classification_model_path = '/data/lizard_classification_model'
classification_thermo_model_file = 'lizard_model_ben_5_0.96.pkl'
# load thermoregulation model and check classes
classifier_thermo = load_learner(classification_model_path, classification_thermo_model_file)
labels_thermo = classifier_thermo.data.classes
# load color model and check classes
classification_color_model_file = 'lizard_model_color_ben__August2_0.97.pkl'
classifier_color = load_learner(classification_model_path, classification_color_model_file)
labels_color = classifier_color.data.classes

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
    # the lock file is used to make sure other processes (if using the "run_detector_classification.sh" script) don't process the same file
    lock_file_name = os.path.join(dir_name, "lock_" + base_name)
    # the detected file will save the image with the marked detections and classifications 
    detected_file_name = os.path.join(dir_name, "detection_" + base_name)
    # the text file will save the table with the detections and classifications
    txt_file_name = os.path.join(dir_name, base_name + '.txt')
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
