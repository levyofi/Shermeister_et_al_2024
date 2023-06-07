#Fitting the detection and classification models
## Files used in the training stage:
The training process uses following files:
`train_detection.py`: The file is used for training the detection model.
`detect_create_xml_and_crop.py`: The file is used for automatically generate more annotations for training the detector model and cropped lizards images for training the classification models.
`train_thermoregulation_classification.py` and `train_color_classification.py" files: These files are used for training the Behavioral Thermoregulation classification model and Color classification model, respectively.

## Files used in the framework execution and interpretation stage:
`detect_and_classify.py`: The file used to run the framework. 
`check_CAM.py`: The file used to generate heatmaps highlighting the important pixels for classification of our models. 
`gradcam.py`: The file contains functions used by `check_CAM.py`.
`
## Explenation of the training, execution, and interpretation:
Here, we developed the Artificial Intelligence models used in our framework. The training data can be found in `Data/training_data`.
During the training process, we first trained a preleminary lizard detection model using the `train_detection.py` file. Next, we used the `detect_create_xml_and_crop.py` file to create more annotations and crop detected lizards for developing the classification training data. We then trained the classification models using `train_thermoregulation_classification.py` and `train_color_classification.py" files. 
After validating the automatically generated annotations, we repeated the detector training process using the same `train_detection.py` file (but this time with much more annotations). Finally, we used the detection and classification models to execute the framework using the `detect_and_classify.py` file.
To learn what parts of an image were important for the classification process of our models, we used the `check_CAM.py` file on several images shown in Figure 7 in our paper.

## Preparing the python environment:
Our approach use the recommendations and python modules used in Mircosoft's computer vision recipes. Our training scripts are an adaptation of their training scripts for detection and classification scenarios. 
We prepared the conda environment as described in their github repository:
```
git clone https://github.com/Microsoft/computervision-recipes
cd computervision-recipes
conda env create -f environment.yml
```
We also installed other packages needed for our project.

## Executing models in parallel:
The framework can be run in parallel using the `run_detector_classification.sh` bash script. The script opens several terminal tabs and execute the `detect_and_classify.py` file in each tab. Each execution will analyze different images since once a process scans a file it "locks" it using a lock file, so the other parallel process will skip it. The processes will also skip any image for which output files were already generated.
The script takes two arguments. The first is the number of processes to run in parallel and the second is the id of the GPU that we want to use. For example:
running
```
./run_detector_classification.sh 5 0
```
will open 4 tabs that will execute `detect_and_classify.py` in parellel on GPU number 0. 
