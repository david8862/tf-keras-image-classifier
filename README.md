# TF Keras CNN image classification Modelset

## Introduction

An end-to-end CNN image classification model training framework. Implement with tf.keras, including model training/tuning, model evaluation, trained model export (PB/ONNX/TFLITE) and a simple on device inference sample (MNN).

#### Model Type
- [x] MobileNet
- [x] MobileNetV2
- [x] MobilenetV3(Large/Small)
- [x] MobileViT(S/XS/XXS) ([paper](https://arxiv.org/abs/2110.02178))
- [x] PeleeNet ([paper](https://arxiv.org/abs/1804.06882))
- [x] GhostNet ([paper](https://arxiv.org/abs/1911.11907))
- [x] SqueezeNet ([paper](https://arxiv.org/abs/1602.07360))
- [x] ResNet50


## Guide of train/evaluate/demo

### Train

1. Install requirements on Ubuntu 16.04/18.04:

```
# pip install -r requirements.txt
```

2. Prepare dataset

    * Collect classification train/val/test images and place at `<dataset path>/` with 1 folder for a class. The folder name should be using the class name, like:

    ```
    <dataset path>/
    ├── class1
    │   ├── class1_1.jpg
    │   ├── class1_2.jpg
    │   ├── class1_3.jpg
    │   └── ...
    ├── class2
    │   ├── class2_1.jpg
    │   ├── class2_2.jpg
    │   ├── class2_3.jpg
    │   └── ...
    ├── class3
    │   ├── class3_1.jpg
    │   └── ...
    │
    └──...
    ```

    The train/val/test dataset path should follow the same structure


3. [train.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/train.py)

```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE]
                [--model_input_shape MODEL_INPUT_SHAPE]
                [--head_conv_channel HEAD_CONV_CHANNEL]
                [--weights_path WEIGHTS_PATH] --train_data_path
                TRAIN_DATA_PATH --val_data_path VAL_DATA_PATH
                [--classes_path CLASSES_PATH] [--batch_size BATCH_SIZE]
                [--optimizer {adam,rmsprop,sgd}]
                [--learning_rate LEARNING_RATE]
                [--decay_type {None,cosine,exponential,polynomial,piecewise_constant}]
                [--init_epoch INIT_EPOCH] [--transfer_epoch TRANSFER_EPOCH]
                [--total_epoch TOTAL_EPOCH] [--gpu_num GPU_NUM]

train a simple CNN classifier

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        backbone model type: mobilenetv3/v2/simple_cnn,
                        default=mobilenetv2
  --model_input_shape MODEL_INPUT_SHAPE
                        model image input shape as <height>x<width>,
                        default=224x224
  --head_conv_channel HEAD_CONV_CHANNEL
                        channel number for head part convolution, default=128
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --train_data_path TRAIN_DATA_PATH
                        path to train image dataset
  --val_data_path VAL_DATA_PATH
                        path to validation image dataset
  --classes_path CLASSES_PATH
                        path to classes definition
  --batch_size BATCH_SIZE
                        batch size for train, default=32
  --optimizer {adam,rmsprop,sgd}
                        optimizer for training (adam/rmsprop/sgd),
                        default=adam
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --decay_type {None,cosine,exponential,polynomial,piecewise_constant}
                        Learning rate decay type, default=None
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training,
                        default=0
  --transfer_epoch TRANSFER_EPOCH
                        Transfer training (from Imagenet) stage epochs,
                        default=5
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=100
  --gpu_num GPU_NUM     Number of GPU to use, default=1
```

Following is reference config cmd for training mobilenetv2 model:
```
# python train.py --model_type=mobilenetv2 --model_input_shape=112x112 --train_data_path=./train_data/ --val_data_path=./val_data/ --decay_type=cosine --transfer_epoch=1
```

Checkpoints during training could be found at `logs/000/`. Choose a best one as result

For MobileNetV1/V2/V3 family, you can manually change its alpha param in [model.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/classifier/model.py)


### Evaluation
Use [eval.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/eval.py) to do evaluation on the trained model with test dataset:

```
# python eval.py -h
usage: eval.py [-h] --model_path MODEL_PATH --dataset_path DATASET_PATH
               [--classes_path CLASSES_PATH]
               [--model_input_shape MODEL_INPUT_SHAPE]

evaluate CNN classifer model (h5/pb/onnx/tflite/mnn) with test dataset

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model file
  --dataset_path DATASET_PATH
                        path to evaluation image dataset
  --classes_path CLASSES_PATH
                        path to class definitions
  --model_input_shape MODEL_INPUT_SHAPE
                        model image input size as <height>x<width>,
                        default=224x224
```

Reference cmd:

```
# python eval.py --model_path=model.h5 --dataset_path=./test_data/ --model_input_shape=112x112
```

You can also use [heatmap_check.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/tools/evaluation/heatmap_check.py) to check the model heatmap activation with test images:

```
# cd tools/evaluation/ && python heatmap_check.py -h
usage: heatmap_check.py [-h] --model_path MODEL_PATH --image_path IMAGE_PATH
                        [--classes_path CLASSES_PATH] --heatmap_path
                        HEATMAP_PATH

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        model file to predict
  --image_path IMAGE_PATH
                        Image file or directory to predict
  --classes_path CLASSES_PATH
                        path to class definition, optional
  --heatmap_path HEATMAP_PATH
                        output heatmap file or directory
```


### Demo
1. [validate_classifier.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/tools/evaluation/validate_classifier.py)

```
# cd tools/evaluation/ && python validate_classifier.py -h
usage: validate_classifier.py [-h] --model_path MODEL_PATH --image_file
                              IMAGE_FILE [--model_image_size MODEL_IMAGE_SIZE]
                              [--classes_path CLASSES_PATH]
                              [--loop_count LOOP_COUNT]

validate CNN classifier model (h5/pb/onnx/tflite/mnn) with image

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        model file to predict
  --image_file IMAGE_FILE
                        image file to predict
  --model_image_size MODEL_IMAGE_SIZE
                        model image input size as <height>x<width>,
                        default=224x224
  --classes_path CLASSES_PATH
                        path to class name definitions
  --loop_count LOOP_COUNT
                        loop inference for certain times
```

### Tensorflow model convert
Using [keras_to_tensorflow.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/tools/model_converter/keras_to_tensorflow.py) to convert the tf.keras .h5 model to tensorflow frozen pb model:
```
# python keras_to_tensorflow.py
    --input_model="path/to/keras/model.h5"
    --output_model="path/to/save/model.pb"
```

### ONNX model convert
Using [keras_to_onnx.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/tools/model_converter/keras_to_onnx.py) to convert the tf.keras .h5 model to ONNX model:
```
### need to set environment TF_KERAS=1 for tf.keras model
# export TF_KERAS=1
# python keras_to_onnx.py
    --keras_model_file="path/to/keras/model.h5"
    --output_file="path/to/save/model.onnx"
    --op_set=11
```

You can also use [eval.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/eval.py) to do evaluation on the pb & onnx inference model

### Inference model deployment
See [on-device inference](https://github.com/david8862/tf-keras-image-classifier/tree/master/inference) for MNN model deployment

