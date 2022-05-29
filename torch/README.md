# PyTorch CNN image classification Modelset

## Introduction

An end-to-end CNN image classification model training framework. Implement with PyTorch, including model training/tuning, model evaluation, trained model export (ONNX/TorchScript) and a simple on device inference sample (MNN).

#### Model Type
- [x] MobileNetV2
- [x] MobilenetV3(Large/Small)
- [x] PeleeNet ([paper](https://arxiv.org/abs/1804.06882))
- [x] CSPPeleeNet ([link](https://github.com/WongKinYiu/CrossStagePartialNetworks))
- [x] GhostNet ([link](https://arxiv.org/abs/1911.11907))
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


3. [train.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/train.py)

```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE] [--model_input_shape MODEL_INPUT_SHAPE] [--head_conv_channel HEAD_CONV_CHANNEL]
                [--weights_path WEIGHTS_PATH] --train_data_path TRAIN_DATA_PATH --val_data_path VAL_DATA_PATH
                [--batch_size BATCH_SIZE] [--optimizer {adam,rmsprop,sgd}] [--learning_rate LEARNING_RATE]
                [--decay_type {None,cosine,plateau,exponential,step}] [--weight_decay WEIGHT_DECAY] [--init_epoch INIT_EPOCH]
                [--transfer_epoch TRANSFER_EPOCH] [--total_epoch TOTAL_EPOCH] [--no_cuda]

train a simple CNN classifier with PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        backbone model type: mobilenetv3/v2/simple_cnn, default=mobilenetv2
  --model_input_shape MODEL_INPUT_SHAPE
                        model image input shape as <height>x<width>, default=224x224
  --head_conv_channel HEAD_CONV_CHANNEL
                        channel number for head part convolution, default=128
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --train_data_path TRAIN_DATA_PATH
                        path to train image dataset
  --val_data_path VAL_DATA_PATH
                        path to validation image dataset
  --batch_size BATCH_SIZE
                        batch size for train, default=64
  --optimizer {adam,rmsprop,sgd}
                        optimizer for training (adam/rmsprop/sgd), default=adam
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --decay_type {None,cosine,plateau,exponential,step}
                        Learning rate decay type, default=None
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer, default=0.0005
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training, default=0
  --transfer_epoch TRANSFER_EPOCH
                        Transfer training (from Imagenet) stage epochs, default=5
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=100
  --no_cuda             disables CUDA training
```

Following is reference config cmd for training mobilenetv2 model:
```
# python train.py --model_type=mobilenetv2 --model_input_shape=112x112 --train_data_path=./train_data/ --val_data_path=./val_data/ --decay_type=cosine --transfer_epoch=1
```

Checkpoints during training could be found at `logs/000/`. Choose a best one as result

For MobileNetV2/V3 family, you can manually change its width multiply param in [model.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/classifier/model.py)


### Evaluation
Use [eval.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/eval.py) to do evaluation on the trained model with test dataset:

```
# python eval.py -h
usage: eval.py [-h] --model_path MODEL_PATH --dataset_path DATASET_PATH [--model_input_shape MODEL_INPUT_SHAPE]
               [--class_index CLASS_INDEX]

evaluate CNN classifer model (pth/onnx/mnn) with test dataset

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model file
  --dataset_path DATASET_PATH
                        path to evaluation image dataset
  --model_input_shape MODEL_INPUT_SHAPE
                        model image input size as <height>x<width>,
                        default=224x224
  --class_index CLASS_INDEX
                        class index to check the best threshold, default=0
```

Reference cmd:

```
# python eval.py --model_path=model.pth --dataset_path=./test_data/ --model_input_shape=112x112
```

You can also use [heatmap_check.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/tools/evaluation/heatmap_check.py) to check the model heatmap activation with test images:

```
# cd tools/evaluation/ && python heatmap_check.py -h
usage: heatmap_check.py [-h] --image_path IMAGE_PATH --model_path MODEL_PATH
                        [--model_input_shape MODEL_INPUT_SHAPE] --heatmap_path
                        HEATMAP_PATH [--classes_path CLASSES_PATH]

check heatmap activation for CNN classifer model (pth) with test images

optional arguments:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        Image file or directory to predict
  --model_path MODEL_PATH
                        model file to predict
  --model_input_shape MODEL_INPUT_SHAPE
                        model input image shape as <height>x<width>,
                        default=224x224
  --heatmap_path HEATMAP_PATH
                        output heatmap file or directory
  --classes_path CLASSES_PATH
                        path to class definition, optional
```


### Demo
1. [validate_classifier.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/tools/evaluation/validate_classifier.py)

```
# cd tools/evaluation/ && python validate_classifier.py -h
usage: validate_classifier.py [-h] --model_path MODEL_PATH --image_file
                              IMAGE_FILE
                              [--model_input_shape MODEL_INPUT_SHAPE]
                              [--classes_path CLASSES_PATH]
                              [--loop_count LOOP_COUNT]

validate CNN classifier model (pth/onnx/mnn) with image

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        model file to predict
  --image_file IMAGE_FILE
                        image file to predict
  --model_input_shape MODEL_INPUT_SHAPE
                        model input image shape as <height>x<width>,
                        default=224x224
  --classes_path CLASSES_PATH
                        path to class name definitions
  --loop_count LOOP_COUNT
                        loop inference for certain times
```

### ONNX model convert
Using [export.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/tools/model_converter/export.py) to convert PyTorch .pth model to ONNX model:

```
# cd tools/model_converter/ && python export.py
    --model_path="path/to/model.pth"
    --output_path="path/to/save/model.onnx"
    --model_input_shape=112x112
```

You can also use [eval.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/eval.py) to do evaluation on the onnx inference model

### Inference model deployment
See [on-device inference](https://github.com/david8862/tf-keras-image-classifier/tree/master/torch/inference) for MNN model deployment

