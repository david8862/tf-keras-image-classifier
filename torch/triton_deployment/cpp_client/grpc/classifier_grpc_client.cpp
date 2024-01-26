//
//  classifier_grpc_client.cpp
//  Triton gRPC client
//
//  Created by david8862 on 2024/01/23.
//
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>

#include "grpc_client.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_warmup_runs = 2;
    int top_k = 1;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::string server_addr = "localhost";
    std::string server_port = "8001";
    std::string model_name = "classifier_onnx";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    //bool verbose = false;
};


struct ModelInfo {
    std::string output_name_;
    std::string input_name_;
    std::string input_datatype_;
    // The shape of the input
    int input_c_;
    int input_h_;
    int input_w_;
    // The format of the input
    std::string input_format_;
    int type1_;
    int type3_;
    int max_batch_size_;
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


std::string grpc_data_type_str(int data_type)
{
    switch(data_type)
    {
        case 0:
            return "TYPE_INVALID";
        case 1:
            return "TYPE_BOOL";
        case 2:
            return "TYPE_UINT8";
        case 3:
            return "TYPE_UINT16";
        case 4:
            return "TYPE_UINT32";
        case 5:
            return "TYPE_UINT64";
        case 6:
            return "TYPE_INT8";
        case 7:
            return "TYPE_INT16";
        case 8:
            return "TYPE_INT32";
        case 9:
            return "TYPE_INT64";
        case 10:
            return "TYPE_FP16";
        case 11:
            return "TYPE_FP32";
        case 12:
            return "TYPE_FP64";
        case 13:
            return "TYPE_STRING";
        case 14:
            return "TYPE_BF16";
        default:
            return "TYPE_INVALID";
    }
}


//Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    assert(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        std::cerr << "Can't alloc memory" << std::endl;
        exit(-1);
    }
    stbir_resize_uint8(inputImage, image_width, image_height, 0,
                     input_image, input_width, input_height, 0, image_channel);

    return input_image;
}



//Center crop image to model input shape
uint8_t* image_crop(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    assert(image_channel == input_channel);

    int x_offset = int((image_width - input_width) / 2);
    int y_offset = int((image_height - input_height) / 2);

    if (image_height < input_height || image_width < input_width) {
        std::cerr << "fail to crop due to small input image" << std::endl;
        exit(-1);
    }

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        std::cerr << "Can't alloc memory" << std::endl;
        exit(-1);
    }

    // Crop out src image into input image
    for (int h = 0; h < input_height; h++) {
        for (int w = 0; w < input_width; w++) {
            for (int c = 0; c < input_channel; c++) {
                input_image[h*input_width*input_channel + w*input_channel + c] = inputImage[(h+y_offset)*image_width*image_channel + (w+x_offset)*image_channel + c];
            }
        }
    }

    return input_image;
}


void fill_data(std::vector<float>& out, uint8_t* in, int input_width, int input_height,
            int input_channels, Settings* s) {
  auto output_number_of_pixels = input_height * input_width * input_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
      out[i] = (in[i] - s->input_mean) / s->input_std;

    //if (s->input_floating)
      //out[i] = (in[i] - s->input_mean) / s->input_std;
    //else
      //out[i] = (uint8_t)in[i];
  }

  return;
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;
    bool verbose = false;
    std::string url = s->server_addr + ":" + s->server_port;
    std::string model_version = "";
    tc::Headers http_headers;

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    std::cout << "num_classes: " << num_classes << "\n";

    // create InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::Error err;

    err = tc::InferenceServerGrpcClient::Create(&client, url, verbose);
    if (!err.IsOk()) {
        std::cerr << "unable to create grpc client: " << err << std::endl;
        exit(1);
    };

    // confirm server & model is available
    bool live;
    err = client->IsServerLive(&live);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get server liveness: " << err << std::endl;
        exit(1);
    }
    if (!live) {
        std::cerr << "error: server is not live" << std::endl;
        exit(1);
    }

    bool ready;
    err = client->IsServerReady(&ready);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get server readiness: " << err << std::endl;
        exit(1);
    }
    if (!ready) {
        std::cerr << "error: server is not ready" << std::endl;
        exit(1);
    }

    bool model_ready;
    err = client->IsModelReady(&model_ready, s->model_name);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get model readiness: " << err << std::endl;
        exit(1);
    }
    if (!model_ready) {
        std::cerr << "error: model " << s->model_name << " is not ready" << std::endl;
        exit(1);
    }

    // get model metadata
    inference::ModelMetadataResponse model_metadata;
    err = client->ModelMetadata(&model_metadata, s->model_name, model_version, http_headers);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
        exit(1);
    }

    // get model config
    inference::ModelConfigResponse model_config;
    err = client->ModelConfig(&model_config, s->model_name, model_version, http_headers);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }

    // check input/output num
    if (model_config.config().input().size() != 1) {
        std::cerr << "expecting 1 input in model configuration, got "
            << model_config.config().input().size() << std::endl;
        exit(1);
    }
    if (model_config.config().output().size() != 1) {
        std::cerr << "expecting 1 output in model configuration, got "
            << model_config.config().output().size() << std::endl;
        exit(1);
    }

    // parse input metadata & config
    auto input_metadata = model_metadata.inputs(0);
    auto input_config = model_config.config().input(0);

    std::string input_name = input_metadata.name();
    std::string input_type = input_metadata.datatype();
    //std::string input_name = input_config.name();
    //std::string input_type = grpc_data_type_str(input_config.data_type());


    // assume NCHW layout for input
    auto input_dims = input_config.dims().size();
    assert(input_dims == 4);
    int input_batch = input_config.dims(0);
    int input_channel = input_config.dims(1);
    int input_height = input_config.dims(2);
    int input_width = input_config.dims(3);

    std::cout << "input tensor info: "
              << "name " << input_name << ", "
              << "type " << input_type << ", "
              << "dim_size " << input_dims << ", "
              << "batch " << input_batch << ", "
              << "height " << input_height << ", "
              << "width " << input_width << ", "
              << "channels " << input_channel << "\n";

    // assume input tensor type is fp32
    assert(input_type == "FP32");
    assert(input_batch == 1);
    std::vector<int64_t> input_shape{input_batch, input_channel, input_height, input_width};

    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        std::cerr << "can't open " << inputPath << std::endl;
        return;
    }
    std::cout << "origin image size: width:" << image_width
              << ", height:" << image_height
              << ", channel:" << image_channel
              << "\n";

    // crop input image
    uint8_t* cropedImage = image_crop(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    uint8_t* targetImage = cropedImage;

    // free input image
    int data_num = input_batch * input_channel * input_height * input_width;
    std::cout << "data_num: " << data_num << "\n";

    std::vector<float> input_data(data_num);

    fill_data(input_data, targetImage,
              input_width, input_height, input_channel, s);

    // Initialize the inputs with the data.
    tc::InferInput* image_input;
    err = tc::InferInput::Create(&image_input, input_name, input_shape, input_type);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get input: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferInput> image_input_ptr(image_input);

    err = image_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input_data[0]), input_data.size()*sizeof(float));
    if (!err.IsOk()) {
        std::cerr << "error: unable to set data for " + input_name + ": " << err << std::endl;
        exit(1);
    }


    // parse output metadata & config
    auto output_metadata = model_metadata.outputs(0);
    auto output_config = model_config.config().output(0);

    std::string output_name = output_metadata.name();
    std::string output_type = output_metadata.datatype();
    //std::string output_name = output_config.name();
    //std::string output_type = grpc_data_type_str(output_config.data_type());


    // get output tensor info, assume only 1 output tensor (scores)
    // image_input: 1 x 3 x 224 x 224
    // "scores": 1 x num_classes
    auto output_dims = output_config.dims().size();
    assert(output_dims == 2);

    int output_batch = output_config.dims(0);
    int output_classes = output_config.dims(1);

    std::cout << "output tensor info: "
              << "name " << output_name << ", "
              << "type " << output_type << ", "
              << "dim_size " << output_dims << ", "
              << "batch " << output_batch << ", "
              << "classes " << output_classes << "\n";

    // check if predict class number matches label file
    assert(num_classes == output_classes);

    // assume output tensor type is fp32
    assert(output_type == "FP32");
    assert(output_batch == 1);
    std::vector<int64_t> output_shape{output_batch, output_classes};


    // Generate the outputs to be requested.
    tc::InferRequestedOutput* score_output;
    err = tc::InferRequestedOutput::Create(&score_output, output_name);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get output: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferRequestedOutput> score_output_ptr(score_output);


    // inference settings
    tc::InferOptions options(s->model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = 0;

    // inference inputs/outputs/results
    std::vector<tc::InferInput*> inputs = {image_input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {score_output_ptr.get()};
    tc::InferResult* results;

    // do inference to get result
    err = client->Infer(&results, options, inputs, outputs);
    //err = client->Infer(&results, options, inputs, outputs, http_headers, compression_algorithm);
    if (!err.IsOk()) {
        std::cerr << "error: unable to run model: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferResult> results_ptr(results);


    // validate result shape and data type


#if 0
    // Create the data for the two input tensors. Initialize the first
    // to unique integers and the second to all ones.
    std::vector<int32_t> input0_data(16);
    std::vector<int32_t> input1_data(16);
    for (size_t i = 0; i < 16; ++i) {
      input0_data[i] = i;
      input1_data[i] = 1;
    }


    // Initialize the inputs with the data.
    tc::InferInput* input0;
#endif



    //std::cout << "input_name: " << input_name << "\n";
    //std::cout << "output_name: " << output_name << "\n";
    //std::cout << "input_type: " << input_type << "\n";
    //std::cout << "output_type: " << output_type << "\n";





    // Run with the same name to ensure cached channel is not used
    for (int i = 0; i < s->loop_count; i++) {
        //tc::Error err = tc::InferenceServerGrpcClient::Create(
            //&client, url, verbose, use_ssl, ssl_options, tc::KeepAliveOptions(),
            //use_cached_channel);

        //if (!err.IsOk()) {
            //std::cerr << "unable to create grpc client: " << err << std::endl;
            //exit(1);
        //};

        //err);


    }



    return;
}


void display_usage() {
    std::cout
        << "Usage: classifier_grpc_client\n"
        << "--server_addr, -a: localhost\n"
        << "--server_port, -p: 8001\n"
        << "--model_name, -m: classifier_onnx\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--top_k, -k: show top k classes result\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"server_addr", required_argument, nullptr, 'a'},
        {"server_port", required_argument, nullptr, 'p'},
        {"model_name", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"top_k", required_argument, nullptr, 'k'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:hi:k:l:m:p:s:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.server_addr = optarg;
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_img_name = optarg;
        break;
      case 'k':
        s.top_k=
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.server_port = optarg;
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}
