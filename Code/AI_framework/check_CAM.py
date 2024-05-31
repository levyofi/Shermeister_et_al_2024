#This code plot gradcam images for a set of images in an input folder

import glob
from functools import partial
import os
from pathlib import Path
import sys
import shutil
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import scrapbook as sb

import fastai
from fastai.metrics import accuracy
from fastai.vision import (
 open_image, load_learner
)
sys.path.append("../computervision-recipes")
from utils_cv.classification.model import (
    IMAGENET_IM_SIZE as IMAGE_SIZE,
    TrainMetricsRecorder,
    get_preds,
)
IMAGE_SIZE = 800
from gradcam import GradCam, plot_combined
from utils_cv.classification.plot import plot_pr_roc_curves
from utils_cv.classification.widget import ResultsWidget
from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url
from utils_cv.common.gpu import db_num_workers, which_processor
from utils_cv.common.misc import copy_files, set_random_seed
from utils_cv.common.plot import line_graph, show_ims

import warnings
warnings.filterwarnings('ignore')

print(f"Fast.ai version = {fastai.__version__}")
which_processor()

DATA_PATH = "path to images folder"

#load thermoregulation model
classification_model_path = '/data/lizard_classification_model'
classification_thermo_model_file = 'lizard_model_ben_5_0.96.pkl'
classifier_thermo = load_learner(classification_model_path, classification_thermo_model_file)
labels_thermo = classifier_thermo.data.classes

#model color model
classification_color_model_file = 'lizard_model_color_ben__August2_0.97.pkl'
classifier_color = load_learner(classification_model_path, classification_color_model_file)
labels_color = classifier_color.data.classes

print('Finished loading model')

#get a list of all the files in the input folder
file_listing = glob.glob(os.path.join(DATA_PATH,'**','*.jpg'), recursive=True) #os.listdir(path)
# get each file and plot the gradcam heatmap for each model
for file_name in file_listing:
    if file_name.endswith(".jpg"):
        print(file_name)
        #get the file name and replace "jpg" with "png"
        base_name = os.path.basename(file_name).split('.')[0]+".png"
        #get the directory name
        dir_name = os.path.dirname(file_name)
        #get the output file name
        output_file_name = os.path.join(dir_name, "gradcam_" + base_name)
        #open image file
        img = open_image(file_name);
        img = img.resize(size=400)
        #get the gradcam for each model
        gcam1 = GradCam.from_one_img(classifier_color,img)
        gcam2 = GradCam.from_one_img(classifier_thermo, img)
        #combine the gradcams and add the original image
        plot_combined(gcam2, gcam1)
        plt.margins(x=0, y=0)
        #save the output
        plt.savefig(output_file_name, format="png", dpi=300)
        plt.show()

print("finished")
