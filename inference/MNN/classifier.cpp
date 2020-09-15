//
//  classifier.cpp
//  MNN
//
//  Created by david8862 on 2020/08/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
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
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_threads = 4;
    int number_of_warmup_runs = 2;
    float input_mean = 0.0f;
    float input_std = 255.0f;
    std::string model_name = "./model.mnn";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    bool input_floating = false;
    //bool verbose = false;
    //string input_layer_type = "uint8_t";
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


void display_usage() {
    std::cout
        << "Usage: deeplabSegment\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


// CNN Classifier postprocess
void classifier_postprocess(const Tensor* score_tensor, std::vector<uint8_t> &class_indexes)
{
    // 1. do following transform to get the top-1 class index:
    //
    //    class = np.argmax(pred, axis=-1)
    //
    const float* data = score_tensor->host<float>();
    auto unit = sizeof(float);
    auto dimType = score_tensor->getDimensionType();

    auto batch   = score_tensor->batch();
    auto channel = score_tensor->channel();
    auto height  = score_tensor->height();
    auto width   = score_tensor->width();


    int class_size;
    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        // output is on height dim, so width & channel should be 0
        class_size = height;
        MNN_ASSERT(width == 0);
        MNN_ASSERT(channel == 0);

        //bytesPerRow   = channel * unit;
        //bytesPerImage = width * bytesPerRow;
        //bytesPerBatch = height * bytesPerImage;
        bytesPerBatch   = height * unit;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");

        // output is on channel dim, so width & height should be 0
        class_size = channel;
        MNN_ASSERT(width == 0);
        MNN_ASSERT(height == 0);

        //bytesPerRow   = width * unit;
        //bytesPerImage = height * bytesPerRow;
        //bytesPerBatch = channel * bytesPerImage;
        bytesPerBatch   = channel * unit;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    for (int b = 0; b < batch; b++) {
        const float* bytes = data + b * bytesPerBatch / unit;
        MNN_PRINT("batch %d:\n", b);

        // Get class index with max score,
        // just as Python postprocess:
        //
        // class = np.argmax(pred, axis=-1)
        //
        uint8_t class_index = 0;
        float max_score = 0.0;
        for (int i = 0; i < class_size; i++) {
            if (bytes[i] > max_score) {
                class_index = i;
                max_score = bytes[i];
            }
        }
        class_indexes.emplace_back(class_index);
    }
    return;
}


//Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    MNN_ASSERT(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
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
    MNN_ASSERT(image_channel == input_channel);

    int x_offset = int((image_width - input_width) / 2);
    int y_offset = int((image_height - input_height) / 2);

    if (image_height < input_height || image_width < input_width) {
        MNN_PRINT("fail to crop due to small input image\n");
        exit(-1);
    }

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
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


//Reorder image from channel last to channel first
uint8_t* image_reorder(uint8_t* inputImage, int image_width, int image_height, int image_channel)
{
    uint8_t* reorder_image = (uint8_t*)malloc(image_height * image_width * image_channel * sizeof(uint8_t));
    if (reorder_image == nullptr) {
        MNN_PRINT("Can't alloc memory\n");
        exit(-1);
    }

    // Reorder src image (channel last) into channel first image
    for (int h = 0; h < image_height; h++) {
        for (int w = 0; w < image_width; w++) {
            for (int c = 0; c < image_channel; c++) {
                int image_offset = h * image_width * image_channel + w * image_channel + c;
                int reorder_offset = c * image_width * image_height + h * image_width + w;

                reorder_image[reorder_offset] = inputImage[image_offset];
            }
        }
    }

    return reorder_image;
}


template <class T>
void fill_data(T* out, uint8_t* in, int input_width, int input_height,
            int input_channels, Settings* s) {
  auto output_number_of_pixels = input_height * input_width * input_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (in[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)in[i];
  }

  return;
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_CPU, MNN_FORWARD_OPENCL
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = s->number_of_threads;

    BackendConfig bnconfig;
    bnconfig.memory = BackendConfig::Memory_Normal; //Memory_High, Memory_Low
    bnconfig.power = BackendConfig::Power_Normal; //Power_High, Power_Low
    bnconfig.precision = BackendConfig::Precision_Normal; //Precision_High, Precision_Low
    config.backendConfig = &bnconfig;

    auto session = net->createSession(config);
    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // get classes labels and add background label
    std::vector<std::string> classes;
    //classes.emplace_back("background");
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    MNN_PRINT("num_classes: %d\n", num_classes);

    // get input tensor info, assume only 1 input tensor (image_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto image_input = inputs.begin()->second;
    int input_width = image_input->width();
    int input_height = image_input->height();
    int input_channel = image_input->channel();
    int input_dim_type = image_input->getDimensionType();

    std::vector<std::string> dim_type_string = {"TENSORFLOW", "CAFFE", "CAFFE_C4"};

    MNN_PRINT("image_input: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", inputs.begin()->first.c_str(), input_width, input_height, input_channel, dim_type_string[input_dim_type].c_str());


    //auto shape = image_input->shape();
    //shape[0] = 1;
    //net->resizeTensor(image_input, shape);
    //net->resizeSession(session);

    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPath);
        return;
    }
    MNN_PRINT("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);

    // crop input image
    uint8_t* cropedImage = image_crop(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    uint8_t* targetImage = cropedImage;
    if (input_dim_type == Tensor::CAFFE) {
        uint8_t* reorderImage = image_reorder(cropedImage, input_width, input_height, input_channel);

        // free croped image
        stbi_image_free(cropedImage);
        cropedImage = nullptr;
        targetImage = reorderImage;
    }

    // assume input tensor type is float
    MNN_ASSERT(image_input->getType().code == halide_type_float);
    s->input_floating = true;

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            fill_data<float>(image_input->host<float>(), targetImage,
                input_width, input_height, input_channel, s);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        fill_data<float>(image_input->host<float>(), targetImage,
            input_width, input_height, input_channel, s);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));

    // get output tensor info, assume only 1 output tensor (fc)
    // image_input: 1 x 3 x 112 x 112
    // "fc": 1 x num_classes
    auto outputs = net->getSessionOutputAll(session);
    MNN_ASSERT(outputs.size() == 1);

    auto class_output = outputs.begin()->second;
    int class_width = class_output->width();
    int class_height = class_output->height();
    int class_channel = class_output->channel();
    auto class_dim_type = class_output->getDimensionType();
    MNN_PRINT("output tensor: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", outputs.begin()->first.c_str(), class_width, class_height, class_channel, dim_type_string[class_dim_type].c_str());

    // get class dimension according to different tensor format
    int class_size;
    if (class_dim_type == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        class_size = class_height;
    } else if (class_dim_type == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        class_size = class_channel;
    } else if (class_dim_type == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", class_dim_type);
        exit(-1);
    }

    // check if predict class number matches label file
    MNN_ASSERT(num_classes == class_size);

    // Copy output tensors to host, for further postprocess
    std::shared_ptr<Tensor> output_tensor(new Tensor(class_output, class_dim_type));
    class_output->copyToHostTensor(output_tensor.get());

    // Now we only support float32 type output tensor
    MNN_ASSERT(output_tensor->getType().code == halide_type_float);
    MNN_ASSERT(output_tensor->getType().bits == 32);


    std::vector<uint8_t> class_indexes;
    // Do classifier_postprocess to get top-1 class index
    gettimeofday(&start_time, nullptr);
    classifier_postprocess(output_tensor.get(), class_indexes);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("classifier_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);


    // Show classification result
    MNN_PRINT("Inferenced class:\n");
    for(auto class_index : class_indexes) {
        MNN_PRINT("%s\n", classes[class_index].c_str());
    }

    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();
    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"mnn_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "b:c:hi:l:m:s:t:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
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
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
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

