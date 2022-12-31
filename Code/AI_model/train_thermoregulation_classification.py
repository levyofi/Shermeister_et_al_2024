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
    CategoryList, DatasetType, get_image_files, ImageList, imagenet_stats,
    cnn_learner, models, ClassificationInterpretation,
)

sys.path.append("../computervision-recipes")
from utils_cv.classification.model import (
    IMAGENET_IM_SIZE as IMAGE_SIZE,
    TrainMetricsRecorder,
    get_preds,
)
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

DATA_PATH = "/data/lizard_classification_model/training_data/Ben/images_August_2022" #"~/Downloads/lizard_classification_model/training_data/Ben/images"

# Number of negative samples to add for each iteration of negative mining
NEGATIVE_NUM = 10

EPOCHS_HEAD = 4
EPOCHS_BODY = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 16

# Using fast_inference parameters from 02_training_accuracy_vs_speed notebook.
ARCHITECTURE = models.resnet101

IM_SIZE = 400

# Temporary folder to store datasets for hard-negative mining
NEGATIVE_MINING_DATA_DIR = TemporaryDirectory().name

ori_datapath = Path(DATA_PATH)
neg_datapath = Path("/data/lizard_classification_model/training_data/Ben/images_August_2022/negative")
# We split positive samples into 80% training and 20% validation
data_imlist = (
    ImageList.from_folder(ori_datapath)
    .split_by_rand_pct(valid_pct=0.30, seed=10)
    .label_from_folder()
)
# We use 80% of negative images for hard-negative mining (set U) while 20% for validation
neg_data = (
    ImageList.from_folder(neg_datapath)
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_const()  # We don't use labels for negative data
    .transform(size=IMAGE_SIZE)
    .databunch(bs=BATCH_SIZE, num_workers = db_num_workers())
    .normalize(imagenet_stats)
)
# Do not shuffle U when we predict
neg_data.train_dl = neg_data.train_dl.new(shuffle=False)
neg_data

datapath = Path(NEGATIVE_MINING_DATA_DIR)/'data'

# Training set T
copy_files(data_imlist.train.items, datapath/'train', infer_subdir=True)
# We include first NEGATIVE_NUM negative images in U (neg_data.train_ds) to our initial training set T
copy_files(neg_data.train_ds.items[:NEGATIVE_NUM], datapath/'train'/'negative')

# Validation set V
copy_files(data_imlist.valid.items, datapath/'valid', infer_subdir=True)
copy_files(neg_data.valid_ds.items, datapath/'valid'/'negative')

set_random_seed(10)

data = (
    ImageList.from_folder(datapath)
    .split_by_folder()
    .label_from_folder()
    .transform(size=IMAGE_SIZE)
    .databunch(bs=BATCH_SIZE, num_workers = db_num_workers())
    .normalize(imagenet_stats)
)
data.show_batch()

print(f'number of classes: {data.c} = {data.classes}')
print(data.batch_stats)

learn = cnn_learner(data, ARCHITECTURE, metrics=accuracy)

learn.fit_one_cycle(EPOCHS_HEAD, 10* LEARNING_RATE)

# Records train and valid accuracies by using Callback TrainMetricsRecorder
learn.callbacks.append(TrainMetricsRecorder(learn, show_graph=True))
learn.unfreeze()

# We record train and valid accuracies for later analysis
train_acc = []
valid_acc = []
interpretations = []

# Show the number of repetitions you went through the negative mining
print(f"Ran {len(interpretations)} time(s)")

for i in range(0,20):
    learn.fit_one_cycle(EPOCHS_BODY, LEARNING_RATE)
    interpretations.append(ClassificationInterpretation.from_learner(learn))
    # Store train and valid accuracy
    train_acc.extend(np.array(learn.train_metrics_recorder.train_metrics)[:, 0])
    valid_acc.extend(np.array(learn.train_metrics_recorder.valid_metrics)[:, 0])
    acc = valid_acc[-1]

    line_graph(
        values=(train_acc, valid_acc),
        labels=("Train", "Valid"),
        x_guides=[i * EPOCHS_BODY for i in range(1, len(train_acc) // EPOCHS_BODY + 1)],
        x_name="Epoch",
        y_name="Accuracy",
    )
    interpretations[i].plot_confusion_matrix()
   # plt.show()
    pred_outs = np.array(get_preds(learn, neg_data.train_dl)[0].tolist())
    print(f"Prediction results:\n{pred_outs[:10]}\n...")
    # Get top-n false classified images (by confidence)
    preds = np.argmax(pred_outs, axis=1)
    wrong_ids = np.where(preds != data.classes.index('negative'))[0]
    wrong_ids_confs = [(i, pred_outs[i][preds[i]]) for i in wrong_ids]
    wrong_ids_confs = sorted(wrong_ids_confs, key=lambda l: l[1], reverse=True)[:NEGATIVE_NUM]
    negative_sample_ids = [w[0] for w in wrong_ids_confs]
    negative_sample_labels = [f"Pred: {data.classes[preds[w[0]]]}\nConf: {w[1]:.3f}" for w in wrong_ids_confs]
    #show_ims(neg_data.train_ds.items[negative_sample_ids], negative_sample_labels, rows=NEGATIVE_NUM // 5)

    #add the negatives to the training
    copy_files(neg_data.train_ds.items[negative_sample_ids], datapath / 'train' / 'negative')
    # Reload the dataset which includes more negative-samples
    data = (ImageList.from_folder(datapath)
            .split_by_folder()
            .label_from_folder()
            .transform(size=IMAGE_SIZE)
            .databunch(bs=BATCH_SIZE, num_workers=db_num_workers())
            .normalize(imagenet_stats))
    print(data.batch_stats)
    learn.save("/data/lizard_classification_model/lizard_model_ben_"+str(i)+"_"+f"{acc:4.2f}")
    learn.export("/data/lizard_classification_model/lizard_model_ben_"+str(i)+"_"+f"{acc:4.2f}"+".pkl")
    # Set the dataset to the learner
    learn.data = data



