//
//  pipeline_grpc_client.cpp
//  Triton gRPC client for classifier pipeline
//
//  Created by david8862 on 2024/02/06.
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

namespace tc = triton::client;


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_warmup_runs = 2;
    int top_k = 1;
    std::string server_addr = "localhost";
    std::string server_port = "8001";
    std::string model_name = "classifier_pipeline";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    bool verbose = false;
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


// descend order sort for class prediction records
bool compare_conf(std::pair<int, float> lpred, std::pair<int, float> rpred)
{
    if (lpred.second < rpred.second)
        return false;
    else
        return true;
}

// CNN Classifier postprocess
void classifier_postprocess(const float* score_data, std::vector<std::pair<int, float>> &class_results, std::vector<int64_t> shape)
{
    // 1. do following transform to get sorted class index & score:
    //
    //    class = np.argsort(pred, axis=-1)
    //    class = class[::-1]
    //
    int batch = shape[0];
    int class_size = shape[1];

    // Get sorted class index & score,
    // just as Python postprocess:
    //
    // class = np.argsort(pred, axis=-1)
    // class = class[::-1]
    //
    int class_index = 0;
    float max_score = 0.0;
    for (int i = 0; i < class_size; i++) {
        class_results.emplace_back(std::make_pair(i, score_data[i]));
        if (score_data[i] > max_score) {
            class_index = i;
            max_score = score_data[i];
        }
    }
    // descend sort the class prediction list
    std::sort(class_results.begin(), class_results.end(), compare_conf);

    return;
}


int get_file_size(const char* filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
        return -1;

    fseek(fp, 0, SEEK_END);

    int size = ftell(fp);
    fclose(fp);

    return size;
}


void show_image_info(const char* image_file)
{
    int image_width;
    int image_height;
    int image_channel = 3;
    uint8_t* image_data = (uint8_t*)stbi_load(image_file, &image_width, &image_height, &image_channel, image_channel);
    if (nullptr == image_data) {
        std::cerr << "can't open " << image_file << std::endl;
        return;
    }
    std::cout << "origin image size: width:" << image_width
              << ", height:" << image_height
              << ", channel:" << image_channel
              << "\n";

    // free input image
    stbi_image_free(image_data);
    image_data = nullptr;

    return;
}


void RunInference(Settings* s)
{
    // record run time for every stage
    struct timeval start_time, stop_time;
    std::string server_url = s->server_addr + ":" + s->server_port;
    std::string model_version = "";

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    std::cout << "num_classes: " << num_classes << "\n";

    // create InferenceServerGrpcClient instance to communicate
    // with triton server using gRPC protocol
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::Error err;

    err = tc::InferenceServerGrpcClient::Create(&client, server_url, s->verbose);
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

    // get model config
    inference::ModelConfigResponse model_config;
    err = client->ModelConfig(&model_config, s->model_name, model_version);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }

    // get model metadata, here
    // we will use metadata to parse model info
    inference::ModelMetadataResponse model_metadata;
    err = client->ModelMetadata(&model_metadata, s->model_name, model_version);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
        exit(1);
    }

    // check input/output num
    if (model_metadata.inputs().size() != 1) {
        std::cerr << "expecting 1 input in model metadata, got "
            << model_metadata.inputs().size() << std::endl;
        exit(1);
    }
    if (model_metadata.outputs().size() != 1) {
        std::cerr << "expecting 1 output in model metadata, got "
            << model_metadata.outputs().size() << std::endl;
        exit(1);
    }

    // parse input metadata & config
    auto input_metadata = model_metadata.inputs(0);
    //auto input_config = model_config.config().input(0);

    std::string input_name = input_metadata.name();
    std::string input_type = input_metadata.datatype();
    //std::string input_name = input_config.name();
    //std::string input_type = grpc_data_type_str(input_config.data_type());

    auto shape_str_fn = [](const std::vector<int64_t> &sizes, const uint32_t n_dims) {
        std::stringstream ss;
        ss << "(";

        for (int i = 0; i < n_dims; i++) {
            ss << sizes[i] << ",";
        }
        ss << ")";
        return ss.str();
    };

    // check input shape
    auto input_shape_size = input_metadata.shape().size();
    assert(input_shape_size == 2);
    std::vector<int64_t> input_shape{input_metadata.shape(0),
                                     input_metadata.shape(1)};

    std::cout << "input tensor info: "
              << "name " << input_name << ", "
              << "type " << input_type << ", "
              << "shape " << shape_str_fn(input_shape, input_shape_size) << "\n";
    // assume input tensor type is UINT8
    assert(input_type == "UINT8");

    // show input image info
    show_image_info(s->input_img_name.c_str());

    // read image data in raw mode
    int file_size = get_file_size(s->input_img_name.c_str());
    if (file_size < 0) {
        perror("failed to get file size of input image");
        exit(1);
    }

    uint8_t* input_image = (uint8_t*)malloc(file_size);
    FILE* image_file = fopen(s->input_img_name.c_str(), "rb");
    int ret = fread(input_image, 1, file_size, image_file);
    fclose(image_file);

    // update input_shape with image data size
    input_shape[0] = 1;
    input_shape[1] = file_size;


    // Initialize the inputs with image data.
    tc::InferInput* image_input;
    err = tc::InferInput::Create(&image_input, input_name, input_shape, input_type);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get input: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferInput> image_input_ptr(image_input);

    err = image_input_ptr->AppendRaw(input_image, file_size);
    if (!err.IsOk()) {
        std::cerr << "error: unable to set data for " + input_name + ": " << err << std::endl;
        exit(1);
    }


    // parse output metadata & config
    auto output_metadata = model_metadata.outputs(0);
    //auto output_config = model_config.config().output(0);

    std::string output_name = output_metadata.name();
    std::string output_type = output_metadata.datatype();
    //std::string output_name = output_config.name();
    //std::string output_type = grpc_data_type_str(output_config.data_type());

    // get output tensor info, assume only 1 output tensor
    // "input": batch_size x -1
    // "output": batch_size x num_classes
    auto output_shape_size = output_metadata.shape().size();
    assert(output_shape_size == 2);
    std::vector<int64_t> output_shape{output_metadata.shape(0),
                                      output_metadata.shape(1)};
    int output_batch = output_metadata.shape(0);
    int output_classes = output_metadata.shape(1);

    std::cout << "output tensor info: "
              << "name " << output_name << ", "
              << "type " << output_type << ", "
              << "shape " << shape_str_fn(output_shape, output_shape_size) << ", "
              << "batch " << output_batch << ", "
              << "classes " << output_classes << "\n";

    // check if predict class number matches label file
    assert(num_classes == output_classes);

    // assume output tensor type is fp32
    assert(output_type == "FP32");
    assert(output_batch == -1);

    // generate the outputs to be requested.
    tc::InferRequestedOutput* score_output;
    err = tc::InferRequestedOutput::Create(&score_output, output_name);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get output: " << err << std::endl;
        exit(1);
    }
    std::shared_ptr<tc::InferRequestedOutput> score_output_ptr(score_output);

    // inference settings involving more inference
    // control for service, like sequence info, etc.
    tc::InferOptions options(s->model_name);
    options.model_version_ = model_version;
    options.request_id_ = "1";
    options.sequence_id_ = 0;
    options.sequence_id_str_ = "";
    options.sequence_start_ = false;
    options.sequence_end_ = false;
    options.priority_ = false;
    options.server_timeout_ = 0;
    options.client_timeout_ = 0;

    // prepare inference inputs/outputs/results
    std::vector<tc::InferInput*> inputs = {image_input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {score_output_ptr.get()};
    tc::InferResult* results;

    // do warm up inference
    if (s->loop_count > 1) {
        for (int i = 0; i < s->number_of_warmup_runs; i++) {

            err = client->Infer(&results, options, inputs, outputs);
            if (!err.IsOk()) {
                std::cerr << "error: unable to run model: " << err << std::endl;
                exit(1);
            }
        }
    }

    // do inference to get result
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        err = client->Infer(&results, options, inputs, outputs);
        if (!err.IsOk()) {
            std::cerr << "error: unable to run model: " << err << std::endl;
            exit(1);
        }
    }
    gettimeofday(&stop_time, nullptr);
    std::shared_ptr<tc::InferResult> results_ptr(results);
    std::cout << "model invoke average time: " << (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count) << "ms\n";

    // validate result shape
    std::vector<int64_t> result_shape;
    err = results_ptr->Shape(output_name, &result_shape);
    if (!err.IsOk()) {
        std::cerr << "unable to get shape for '" + output_name + "'" << std::endl;
        exit(1);
    }
    if ((result_shape.size() != 2) || (result_shape[0] != 1) || (result_shape[1] != num_classes)) {
        std::cerr << "error: received incorrect shapes for '" << output_name << "'"
                  << std::endl;
        exit(1);
    }

    // validate result datatype
    std::string result_datatype;
    err = results_ptr->Datatype(output_name, &result_datatype);
    if (!err.IsOk()) {
        std::cerr << "unable to get datatype for '" + output_name + "'" << std::endl;
        exit(1);
    }
    if (result_datatype.compare("FP32") != 0) {
        std::cerr << "error: received incorrect datatype for '" << output_name
                  << "': " << result_datatype << std::endl;
        exit(1);
    }

    // get pointer to result data
    float* result_data;
    size_t result_byte_size;
    err = results_ptr->RawData(output_name, (const uint8_t**)&result_data, &result_byte_size);
    if (!err.IsOk()) {
        std::cerr << "unable to get result data for '" + output_name + "'" << std::endl;
        exit(1);
    }
    if (result_byte_size != sizeof(float)*num_classes) {
        std::cerr << "error: received incorrect byte size for '" << output_name << "'"
                  << result_byte_size << std::endl;
        exit(1);
    }

    std::vector<std::pair<int, float>> class_results;
    gettimeofday(&start_time, nullptr);
    classifier_postprocess(result_data, class_results, result_shape);
    gettimeofday(&stop_time, nullptr);
    std::cout << "classifier_postprocess time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << "ms\n";

    // check class size and top_k
    assert(num_classes == class_results.size());
    assert(s->top_k <= num_classes);

    // Show classification result
    std::cout << "Inferenced class:\n";
    for(int i = 0; i < s->top_k; i++) {
        auto class_result = class_results[i];
        std::cout << classes[class_result.first].c_str() << ": " << class_result.second << std::endl;
    }

    // Release buffer memory
    if (input_image) {
        free(input_image);
        input_image = nullptr;
    }

    if (s->verbose) {
        // show some statistic info
        std::cout << "======Inference Statistics======" << std::endl;
        std::cout << results_ptr->DebugString() << std::endl;

        tc::InferStat infer_stat;
        client->ClientInferStat(&infer_stat);
        std::cout << "======Client Statistics======" << std::endl;
        std::cout << "completed_request_count "
            << infer_stat.completed_request_count << std::endl;
        std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
        std::cout << "cumulative_send_time_ns "
            << infer_stat.cumulative_send_time_ns << std::endl;
        std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

        inference::ModelStatisticsResponse model_stat;
        client->ModelInferenceStatistics(&model_stat, s->model_name);
        std::cout << "======Model Statistics======" << std::endl;
        std::cout << model_stat.DebugString() << std::endl;
    }

    return;
}


void display_usage() {
    std::cout
        << "Usage: pipeline_grpc_client\n"
        << "--server_addr, -a: localhost\n"
        << "--server_port, -p: 8001\n"
        << "--model_name, -m: classifier_pipeline\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--top_k, -k: show top k classes result\n"
        << "--count, -c: loop model run for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        << "--verbose, -v: [0|1] print more information\n"
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
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:c:hi:k:l:m:p:v:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.server_addr = optarg;
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
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'v':
        s.verbose =
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

