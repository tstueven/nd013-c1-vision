# Object Detection in an Urban Environment

## Project overview
This section contains a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

As a driver of a car (or any other vehicle) one needs to be on constant alert, watch and evaluate the surroundings in order to avoid accidents. Different agents show different behaviours, i.e. a car is faster as a cyclist as a pedestrian, whereas the latter might change direction quickly or appear in between cars. All this and more needs to be taken into account when "calculating" a path to ones destination (or at least to the next street corner). The first step in any of this is to recognize what agents at which direction and distance need to be taken into account.

All this is true as well for a self driving car which is in a sense its own driver. It must be able to understand its surroundings, spot other vehicles, cyclists, or pedetrians and act accordingly. The first step -- which we are trying to archive here -- is to localize other objects on the street and classify whether they are a vehicle, cyclist, or pedestrian. This image segmentation and classification task is nowadays done best by a neural network.

To train a complex model such as one to segment an image and classify objects, one luckily does not have to start from scratch or with lower level library APIs such as Keras, but can instead use the [TensorFlow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) which comes with a whole [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) of prebuilt and pretrained models. What is also needed is a lot of training data which luckily gets provided for free by Waymo in the form of the [Waymo Open Dataset](https://waymo.com/open/).

In the following we are going to describe how to set up an environment in which the code can be run. Then there is a brief part about why and how the dataset is split for cross validation. We then take a look at the data that is given to see different examples of inputs and also how the classes are distributed. In the end, of course, we look at training and results.

## Set up
_(This section is heavily based on the Udacity setup instructions.)_

### Environment
The code in this project is run best in a docker container whose image can be build from the `Dockerfile` in the `build` directory. A detailed instruction is given in `build/README.md`.

#### nvtop
To track the GPU utilization the tool [nvtop](https://github.com/Syllo/nvtop) comes in handy. This can easily be installed from the Ubuntu package manager:
```bash
sudo apt install nvtop
```

### Data
#### Acquisition
The data can be downloaded from a Google Cloud Storage bucket. They are quite large and contain a lot of data (such as LIDAR) which are not relevant for the task here. Also, the format of the data is not exactly matching the requirements for inputs to the Object Detection API. The `download_process` script takes care of all this. It downloads and transforms the data for our purposes here. It can be run as follows:
```bash
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```
If you have limited space on your machine, comment in line 123 and only every tenth image gets saved.

#### Splitting
The data need to be split in training, validation, and testing data (see below for why). To do this and avoid copies, symlinks are a good way. The script `create_splits.py` sets up three new folder, `train`, `val`, and `test` and symlinks 80% of the files in the first, 15% in the second, and 5% in the last folder.

```bash
python create_splits.py --source {processed_file_location} --destination {path_for_splits}
```

### Model Pipeline Config
#### Model
To start we first need a model and as a first model we choose the [SSD ResNet50 V1 FPN 640x640 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz). Download (e.g. via `wget`) and unpack this in a `pretrained_models` folder. Details about this model can be found in this [paper](https://arxiv.org/pdf/1708.02002.pdf).

### Pipeline Config
To train a model the config files that come with it when downloaded from the zoo needs to be edited to work with our data. These edits range from the number of result classes to different types of augmentatoins which we gonna discuss later. The command to prepare the reference model is the following:

```bash
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```

### Training and Validation
To start the first training process, move the generated `pipeline_new.config` to the `experiments/reference` folder in your workspace. The run a training process via
```bash
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

When training starts, also already get the evaluation going as not all checkpoints will be kept. At least on my system, due to memory limitations the evaluations cannot be run on the GPU at the same time as the training. As a workaround, the evalution (which does not happen as often and is not as calculation heavy) can be run on the CPU. To do this, use the following command:
```bash
 CUDA_VISIBLE_DEVICES="" python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```