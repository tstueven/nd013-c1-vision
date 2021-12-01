# Object Detection in an Urban Environment

## Project overview
_This section contains a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?_

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
The data can be downloaded from a [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files). They are quite large and contain a lot of data (such as LIDAR) which are not relevant for the task here. Also, the format of the data is not exactly matching the requirements for inputs to the Object Detection API. The `download_process` script takes care of all this. It downloads and transforms the data for our purposes here. It can be run as follows:
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

## Dataset
### Cross validation
In order to validate a model and evaluate training progress, data unseen in the training process are needed. For this reason one has to split the dataset in training and validation as well as test data. Validation data serve e.g. for the purpose of registering overfitting and can be used as a benchmark in preventing it as well as training evaluation in general whereas testdata are not to be used at all until everything is done and ready for a final test.

The `create_splits.py` script splits the data such that 80% are used for training, 15% for validation, and 5% for testing which seems to be pretty common. One thing to remark about the data here is that each of the tfrecord files, along which the splits run, contains around 200 images of a similar scene, ususally the same street. I thought about if those should be mixed and distributed among training, validation, and test sets, but decided against since then the training data would contain images too similar to teh testing and validation images which is not wanted.


### Dataset analysis
[explore_display]: ./writeup_images/explore_display.png "Exploration"
[training_data_dist]: ./writeup_images/training_data_dist.png "Training Data Distribution"

_This section contains a quantitative and qualitative description of the dataset._

There are 15862 images in the training dataset and 2948 in the validation dataset, each with a resolution of 640 x 640. A random sample of ten is dispalyed below.

There are images of cities (1, 0) and suburbs (0,1), at daytime, nighttime (1, 1), and dusk / dawn (2, 1). There are images with many parked or driving cars, a few with pedestrians, and very few with cyclists (0, 1), (1, 1).

We also see taht partly and also many almost completely occluded objects are marked, e.g. pedestrains in (0, 1) or cars in (1, 0). Many of these objects are hardly spotable or distinguishable for the human eye and thus we do not expect a too perfect natwork performance here.

![alt text][explore_display]

The plot below shows the distribution of the different object classes. As can be seen, the majority (282049, 73.69%) of the objects are vehicles and there is also a still a considerable amount of pedestrains (97990, 25.60%). Cyclists, though, there are only few (2725, 0.71%). This is not so surpring as these data were taken mostly in large American cities, but might be a big problem for using the data in an area where cyclists are more common. Luckily, there are also some cyclists (744, 0.90%) in the validation data. 

![alt text][training_data_dist]


## Training
### Reference experiment
[reference_training]: ./writeup_images/reference_training.png "Reference Training"
[batch_10_training]: ./writeup_images/batch10_training.png "Batchsize 10"


_This section details the results of the reference experiment._

As a reference the original pipeline file was used as the starting point. This then was modified via the script described in section [Pipeline Config](#pipeline-config). The pipline can be found in `pipeline_files/pipeline_reference.config`. The result was awful.

![alt text][reference_training]

The localization loss did on average not change too much, but the classification loss, after staying relatively low for a while, exploded and never recovered.

### Improve on the reference
_This section should highlights the different strategies adopted to improve your model._

#### Batch size
The first thing I did to improve the results was to choose a larger batch size as 2 seems really small to me. My first intuition was to use the model's default which is 64. Unfortunately, this was too much for my GPU memory. What was still within limits is a natch size of 10 and that's what I usen then. The pipline can be found in` pipeline_files/pipeline_batch10.config`.

![alt text][batch_10_training]

As can be seen from teh plots above, the results improved tremendously.

#### Model choice
The next thing I tried was a different model. The [model zoo website](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) provides helpful information about model performance. From these data i decided the EfficientDet D1 640x640 looks promising as it is build for the same input size and leads to better results without a big loss in performance.

Unfortunately, despite some effort, I did not manage to get it running. Training would just stop before it really started and I could not find the reason for this.

#### Droput
What I also tested was if the performance could be improved by dropout. The loss in the plot above seemed to have reached a plateau and validation loss is a lot worse then training loss. At least the former might hint to some overfitting. The [box predictor has an option for dropout](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/box_predictor.proto) which was used.

The expted improvement did not happen. Setting this option did not have any effect at all on training.

### Augmentations
[none]: ./writeup_images/augmentations/no_augmentation.png "None"
[gray]: ./writeup_images/augmentations/gray.png "Grayscale"
[brightness_bright]: ./writeup_images/augmentations/brightness_bright.png "Brightness Bright"
[brightness_dark]: ./writeup_images/augmentations/brightness_dark.png "Brightness Dark"
[contrast_high]: ./writeup_images/augmentations/contrast_high.png "Hight Contrast"
[saturation_little]: ./writeup_images/augmentations/saturation_little.png "Little Saturation"
[saturation_lot]: ./writeup_images/augmentations/saturation_lot.png "Lot Saturation"
[hue]: ./writeup_images/augmentations/hue.png "Hue"
[black_patches]: ./writeup_images/augmentations/black_patches.png "Black Patches"

One more method to help improve reults are augmentations. Here the object detection API provides a lot some of which are going to be discussed below.

Two, a horizontal flip and a random crop were already used in the reference model  and kept in everything after that since they seem to amke a lot of sense.

A vertical flip was not included since there is no normal road scenario in which the car would be upside down. Scaling was not used either as this, in one direction, is already a subset of cropping, in he other padding would be needed which brings its own complications.

#### No augmentation
Below one representative image without any augmentations to compare against.

![alt text][none]

#### Gray
Grayscale could simulate rain or fog.

![alt text][gray]

#### Brightness
Changing the brightness could simulate very sunny days (probably with the sun coming in from the front) as well as darker times. i.e. morning or evening.

![alt text][brightness_bright]
![alt text][brightness_dark]

#### Contrast
A change in contrast could supposedly make edge detection in the network more flexible. Below is an example with a higher contrast.

![alt text][contrast_high]

#### Saturation
Going down with the saturation leads towards grayscale but is not as extreme. Going up leads to stronger colours, sometimes too strong as teh example below. Staying in the default limits seems to do just tine though.

![alt text][saturation_little]
![alt text][saturation_lot]

#### Hue
Changes in hue quickly lead to unnatural transformations (below). But when kept small, i.e. within defaults, the variations are modest and could be useful.

![alt text][hue]

### Black patches
Random black patches were used as well as a means to simulate object occlusions. The default seemed to etreme though and the the overall size, number, and probability was lowered.

Below is one typical example containing all augmentations.

![alt text][all]

### Final results
[final_training_results]: ./writeup_images/augmented_training.png "Final training results"
[precision]: ./writeup_images/precision.png "Precision"
[recall]: ./writeup_images/recall.png "recall"



The final pipeline for training with augmentations can be found in `pipeline_files/pipeline_augmented.config`. The trainign results are below.

![alt text][final_training_results]

The results are only marhinally better than without the additional augmentations. The main difference is that saturation does not seem to set in as early and training over more epochs leads to some benefits.

Data for precision and recall of the validation data is dispalyed below.

![alt text][precision]
![alt text][recall]

We find that from both metrics that the detection of large objects works quite well whereas small objects are problematic. One thing to try here migt be to work higher resolution images. But as the larger onjects are the closer ones, this is also the more important metric for us and all together it seems to be solid overall result.