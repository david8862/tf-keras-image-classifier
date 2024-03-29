//
//  classifier.cpp
//  Tengine version
//
//  Created by david8862 on 2021/01/11.
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

#include "tengine_c_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_threads = 4;
    int number_of_warmup_runs = 2;
    int top_k = 1;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::string model_name = "./model.tmfile";
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
        << "--tengine_model, -m: model_name.tmfile\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--top_k, -k: show top k classes result\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        //<< "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


//descend order sort for class prediction records
bool compare_conf(std::pair<uint8_t, float> lpred, std::pair<uint8_t, float> rpred)
{
    if (lpred.second < rpred.second)
        return false;
    else
        return true;
}


// CNN Classifier postprocess
void classifier_postprocess(const tensor_t score_tensor, std::vector<std::pair<uint8_t, float>> &class_results)
{
    // 1. do following transform to get sorted class index & score:
    //
    //    class = np.argsort(pred, axis=-1)
    //    class = class[::-1]
    //

    const float* data = ( float* )get_tensor_buffer(score_tensor);
    auto unit = sizeof(float);

    int score_dims[4] = {0};    // nchw
    int score_dim_number = get_tensor_shape(score_tensor, score_dims, 4);

    if ( score_dim_number != 4 ) {
        fprintf(stderr, "output tensor dim %d incorrect.\n", score_dim_number);
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
    }

    int batch = score_dims[0];
    int channel = score_dims[1];
    int height = score_dims[2];
    int width = score_dims[3];
    int layout = get_tensor_layout(score_tensor); //0: NCHW, 1: NHWC
    int data_type = get_tensor_data_type(score_tensor); //0: TENGINE_DT_FP32, 1: TENGINE_DT_FP16, 2: TENGINE_DT_INT8, 3: TENGINE_DT_UINT8, 4: TENGINE_DT_INT32, 5: TENGINE_DT_INT16

    //printf("output tensor: name:%s, width:%d, height:%d, channel:%d, layout:%d, data_type:%d\n", get_tensor_name(output_tensor), output_width, output_height, output_channel, output_layout, output_data_type);

    // batch size should be always 1
    assert(batch == 1);

    int class_size;
    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (layout == 1) {
        // Tensorflow format tensor, NHWC
        printf("Tensorflow format: NHWC\n");

        // output is on height dim, so width & channel should be 0
        class_size = channel;
        //assert(width == 0);
        //assert(channel == 0);

        bytesPerBatch   = class_size * unit;

    } else if (layout == 0) {
        // Caffe format tensor, NCHW
        printf("Caffe format: NCHW\n");

        // output is on channel dim, so width & height should be 0
        class_size = channel;
        //assert(width == 0);
        //assert(height == 0);

        bytesPerBatch   = class_size * unit;

    } else {
        printf("Invalid layout: %d\n", layout);
        exit(-1);
    }

#if 1
    for (int b = 0; b < batch; b++) {
        const float* bytes = data + b * bytesPerBatch / unit;
        printf("batch %d:\n", b);

        // Get sorted class index & score,
        // just as Python postprocess:
        //
        // class = np.argsort(pred, axis=-1)
        // class = class[::-1]
        //
        uint8_t class_index = 0;
        float max_score = 0.0;
        for (int i = 0; i < class_size; i++) {
            class_results.emplace_back(std::make_pair(i, bytes[i]));
            if (bytes[i] > max_score) {
                class_index = i;
                max_score = bytes[i];
            }
        }
        // descend sort the class prediction list
        std::sort(class_results.begin(), class_results.end(), compare_conf);
    }
#endif
    return;
}


//Resize image to model input shape
uint8_t* image_resize(uint8_t* inputImage, int image_width, int image_height, int image_channel, int input_width, int input_height, int input_channel)
{
    // assume the data channel match
    assert(image_channel == input_channel);

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        printf("Can't alloc memory\n");
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
        printf("fail to crop due to small input image\n");
        exit(-1);
    }

    uint8_t* input_image = (uint8_t*)malloc(input_height * input_width * input_channel * sizeof(uint8_t));
    if (input_image == nullptr) {
        printf("Can't alloc memory\n");
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
        printf("Can't alloc memory\n");
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

    // set runtime options
    struct options opt;
    opt.num_thread = s->number_of_threads;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 255;

    // inital tengine
    if (init_tengine() != 0) {
        fprintf(stderr, "Initial tengine failed.\n");
        return;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    // create graph, load tengine model xxx.tmfile
    graph_t graph = create_graph(NULL, "tengine", s->model_name.c_str());
    if (NULL == graph) {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return;
    }

    // get classes labels and add background label
    std::vector<std::string> classes;
    //classes.emplace_back("background");
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    printf("num_classes: %d\n", num_classes);


    // get input tensor number, assume only 1 input tensor (image_input)
    int input_number = get_graph_input_node_number(graph);
    assert(input_number == 1);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL) {
        fprintf(stderr, "Get input tensor failed\n");
        return;
    }

    int temp_dims[4] = {0};    // nchw
    int temp_dim_number = get_tensor_shape(input_tensor, temp_dims, 4);

    if ( temp_dim_number != 4 ) {
        fprintf(stderr, "input tensor dim %d incorrect.\n", temp_dim_number);
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
    }

    // set the shape, data buffer of input_tensor of the graph
    int input_width = 112;
    int input_height = 112;
    int input_channel = 3;
    int input_layout = get_tensor_layout(input_tensor); //0: NCHW, 1: NHWC
    int input_data_type = get_tensor_data_type(input_tensor); //0: TENGINE_DT_FP32, 1: TENGINE_DT_FP16, 2: TENGINE_DT_INT8, 3: TENGINE_DT_UINT8, 4: TENGINE_DT_INT32, 5: TENGINE_DT_INT16

    int img_size = input_height * input_width * input_channel;
    int dims[] = {1, input_channel, input_height, input_width};    // nchw
    float* input_data = (float*)malloc(img_size * sizeof(float));

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
        fprintf(stderr, "Set input tensor shape failed\n");
        return;
    }

    printf("image_input: name:%s, width:%d, height:%d, channel:%d, layout:%d, data_type:%d\n", get_tensor_name(input_tensor), input_width, input_height, input_channel, input_layout, input_data_type);

    // prerun graph, set work options(num_thread, cluster, precision)
    if (prerun_graph_multithread(graph, opt) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return;
    }


    // load input image
    auto inputPath = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    uint8_t* inputImage = (uint8_t*)stbi_load(inputPath, &image_width, &image_height, &image_channel, input_channel);
    if (nullptr == inputImage) {
        fprintf(stderr, "Can't open %s\n", inputPath);
        return;
    }
    printf("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);

    // crop input image
    uint8_t* cropedImage = image_crop(inputImage, image_width, image_height, image_channel, input_width, input_height, input_channel);

    // free input image
    stbi_image_free(inputImage);
    inputImage = nullptr;

    uint8_t* targetImage = cropedImage;
    if (input_layout == 0) {
        uint8_t* reorderImage = image_reorder(cropedImage, input_width, input_height, input_channel);

        // free croped image
        stbi_image_free(cropedImage);
        cropedImage = nullptr;
        targetImage = reorderImage;
    }

    // assume input tensor type is float
    //MNN_ASSERT(image_input->getType().code == halide_type_float);
    s->input_floating = true;

    if (set_tensor_buffer(input_tensor, input_data, img_size*sizeof(float)) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return;
    }

    // create a host tensor for input data
    fill_data<float>(input_data, targetImage,
                input_width, input_height, input_channel, s);

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            if (run_graph(graph, 1) < 0) {
                fprintf(stderr, "Run graph failed\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        if (run_graph(graph, 1) < 0) {
            fprintf(stderr, "Run graph failed\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    printf("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));

    // get output tensor number, assume only 1 output tensor (fc)
    // image_input: 1 x 3 x 112 x 112
    // "fc": 1 x num_classes
    int output_number = get_graph_output_node_number(graph);
    assert(output_number == 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    if (output_tensor == NULL) {
        fprintf(stderr, "Get output tensor failed\n");
        return;
    }

    //int temp_dims[4] = {0};    // nchw
    temp_dim_number = get_tensor_shape(output_tensor, temp_dims, 4);

    if ( temp_dim_number != 4 ) {
        fprintf(stderr, "output tensor dim %d incorrect.\n", temp_dim_number);
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
    }

    int output_width = temp_dims[3];
    int output_height = temp_dims[2];
    int output_channel = temp_dims[1];
    int output_layout = get_tensor_layout(output_tensor); //0: NCHW, 1: NHWC
    int output_data_type = get_tensor_data_type(output_tensor); //0: TENGINE_DT_FP32, 1: TENGINE_DT_FP16, 2: TENGINE_DT_INT8, 3: TENGINE_DT_UINT8, 4: TENGINE_DT_INT32, 5: TENGINE_DT_INT16

    printf("output tensor: name:%s, width:%d, height:%d, channel:%d, layout:%d, data_type:%d\n", get_tensor_name(output_tensor), output_width, output_height, output_channel, output_layout, output_data_type);

    // get class dimension according to different tensor format
    //int class_size;
    //if (class_dim_type == Tensor::TENSORFLOW) {
        //// Tensorflow format tensor, NHWC
        //class_size = class_height;
    //} else if (class_dim_type == Tensor::CAFFE) {
        //// Caffe format tensor, NCHW
        //class_size = class_channel;
    //} else if (class_dim_type == Tensor::CAFFE_C4) {
        //MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        //exit(-1);
    //} else {
        //MNN_PRINT("Invalid tensor dim type: %d\n", class_dim_type);
        //exit(-1);
    //}

    // check if predict class number matches label file
    //assert(num_classes == class_size);




    // Copy output tensors to host, for further postprocess
    //std::shared_ptr<Tensor> output_tensor(new Tensor(class_output, class_dim_type));
    //class_output->copyToHostTensor(output_tensor.get());

    // Now we only support float32 type output tensor
    //MNN_ASSERT(output_tensor->getType().code == halide_type_float);
    //MNN_ASSERT(output_tensor->getType().bits == 32);


    std::vector<std::pair<uint8_t, float>> class_results;
    // Do classifier_postprocess to get sorted class index & scores
    gettimeofday(&start_time, nullptr);
    classifier_postprocess(output_tensor, class_results);
    gettimeofday(&stop_time, nullptr);
    printf("classifier_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // check class size and top_k
    //assert(num_classes == class_results.size());
    assert(s->top_k <= num_classes);

    // Show classification result
    printf("Inferenced class:\n");
    for(int i = 0; i < s->top_k; i++) {
        auto class_result = class_results[i];
        printf("%s: %f\n", classes[class_result.first].c_str(), class_result.second);
    }

    // release tengine
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"tengine_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"top_k", required_argument, nullptr, 'k'},
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
                    "b:c:hi:k:l:m:s:t:w:", long_options,
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

