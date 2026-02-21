## Deploy triton model inference service

1. Refer to [server_docker_install.sh](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/triton_deployment/server_docker_install.sh) to install triton server docker environment.

2. Run [imagenet_model_dump.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/triton_deployment/imagenet_model_dump.py) to dump out onnx model for triton inference:

```
# python imagenet_model_dump.py -h
usage: imagenet_model_dump.py [-h] [--model_type MODEL_TYPE]
                                   [--model_input_shape MODEL_INPUT_SHAPE]
                                   [--batch_size BATCH_SIZE]
                                   [--output_path OUTPUT_PATH]

Dump imagenet pretrained CNN classifier model and convert to onnx/torchscript

options:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type: mobilenetv3/v2/csppeleenet, default=csppeleenet
  --model_input_shape MODEL_INPUT_SHAPE
                        model image input shape as <height>x<width>, default=224x224
  --batch_size BATCH_SIZE
                        batch size for inference, default=-1
  --output_path OUTPUT_PATH
                        output path to save dumped model, default=model_repository/classifier_onnx/1

# python imagenet_model_dump.py --model_type=csppeleenet
...
Done. Dumped model has been saved to model_repository/classifier_onnx/1
```

3. launch interactive triton server docker image to enter CLI, then install needed package & start triton server:

```
# docker run --gpus=1 -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v./model_repository:/models nvcr.io/nvidia/tritonserver:23.12-py3 bash
# pip install --upgrade pip
# pip install numpy pillow
# tritonserver --model-repository=/models &> /tmp/server.log &
```

4. now the triton service should be ready to use. you can run [classifier_client.py](https://github.com/david8862/tf-keras-image-classifier/blob/master/torch/triton_deployment/python_client/classifier_client.py) or other client script on the client host (need to install packages in requirements.txt) to send inference request to triton server:

```
# cd python_client
# python classifier_client.py -h
usage: classifier_client.py [-h] [--server_addr SERVER_ADDR]
                                 [--server_port SERVER_PORT]
                                 [--model_name MODEL_NAME]
                                 --image_path IMAGE_PATH
                                 [--classes_path CLASSES_PATH]
                                 [--output_path OUTPUT_PATH]
                                 [--protocol {http,grpc}]

classifier http/grpc client for triton inference server

options:
  -h, --help            show this help message and exit
  --server_addr SERVER_ADDR
                        triton server address, default=localhost
  --server_port SERVER_PORT
                        triton server port (8000 for http & 8001 for grpc), default=8000
  --model_name MODEL_NAME
                        model name for inference, default=classifier_onnx
  --image_path IMAGE_PATH
                        image file or directory to inference
  --classes_path CLASSES_PATH
                        path to class name definitions
  --output_path OUTPUT_PATH
                        output path to save dumped model, default=None
  --protocol {http,grpc}
                        comm protocol between triton server & client (http/grpc), default=http

# python classifier_client.py --server_port=8001 --image_path=../test_image/bird.jpg --classes_path=../../../configs/imagenet_2012_classes.txt --protocol=grpc
NCHW input layout
Inference time: 18.05281639ms
Class result
 indigo_bunting:0.939
```

