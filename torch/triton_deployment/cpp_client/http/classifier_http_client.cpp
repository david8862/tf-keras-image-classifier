//
//  classifier_http_client.cpp
//  Triton HTTP client
//
//  Created by david8862 on 2024/01/28.
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

#include "http_client.h"
#include "json_utils.h"
#include <rapidjson/error/en.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace tc = triton::client;


// model inference settings
struct Settings {
    int loop_count = 1;
    int number_of_warmup_runs = 2;
    int top_k = 1;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::string server_addr = "localhost";
    std::string server_port = "8000";
    std::string model_name = "classifier_onnx";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    bool verbose = false;
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


// Re-implement JSON parse with rapidjson, to avoid link error
tc::Error
ParseJson(rapidjson::Document* document, const std::string& json_str)
{
  const unsigned int parseFlags = rapidjson::kParseNanAndInfFlag;
  document->Parse<parseFlags>(json_str.c_str(), json_str.size());
  if (document->HasParseError()) {
    return tc::Error(
        "failed to parse JSON at" + std::to_string(document->GetErrorOffset()) +
        ": " + std::string(GetParseError_En(document->GetParseError())));
  }

  return tc::Error::Success;
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


// Resize image to model input shape
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


// Center crop image to model input shape
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


// Reorder image from channel last to channel first
uint8_t* image_reorder(uint8_t* inputImage, int image_width, int image_height, int image_channel)
{
    uint8_t* reorder_image = (uint8_t*)malloc(image_height * image_width * image_channel * sizeof(uint8_t));
    if (reorder_image == nullptr) {
        std::cerr << "Can't alloc memory" << std::endl;
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

    // create InferenceServerHttpClient instance to communicate
    // with triton server using HTTP protocol
    std::unique_ptr<tc::InferenceServerHttpClient> client;
    tc::Error err;

    err = tc::InferenceServerHttpClient::Create(&client, server_url, s->verbose);
    if (!err.IsOk()) {
        std::cerr << "unable to create http client: " << err << std::endl;
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

    // get server metadata
    std::string server_metadata;
    err = client->ServerMetadata(&server_metadata);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get server metadata: " << err << std::endl;
        exit(1);
    }
    rapidjson::Document server_metadata_json;
    //err = tc::ParseJson(&server_metadata_json, server_metadata);
    err = ParseJson(&server_metadata_json, server_metadata);
    if (!err.IsOk()) {
        std::cerr << "error: failed to parse server metadata: " << err << std::endl;
        exit(1);
    }
    if (s->verbose) {
        std::cout << "\nServer Metadata:\n" << server_metadata << std::endl;
    }

    // get model metadata, here
    // we will use metadata to parse model info
    std::string model_metadata;
    err = client->ModelMetadata(&model_metadata, s->model_name, model_version);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get model metadata: " << err << std::endl;
        exit(1);
    }
    rapidjson::Document model_metadata_json;
    //err = tc::ParseJson(&model_metadata_json, model_metadata);
    err = ParseJson(&model_metadata_json, model_metadata);
    if (!err.IsOk()) {
        std::cerr << "error: failed to parse model metadata: " << err << std::endl;
        exit(1);
    }
    if (s->verbose) {
        std::cout << "\nModel Metadata:\n" << model_metadata << std::endl;
    }

    // get model config
    std::string model_config;
    err = client->ModelConfig(&model_config, s->model_name, model_version);
    if (!err.IsOk()) {
        std::cerr << "error: unable to get model config: " << err << std::endl;
        exit(1);
    }
    rapidjson::Document model_config_json;
    //err = tc::ParseJson(&model_config_json, model_config);
    err = ParseJson(&model_config_json, model_config);
    if (!err.IsOk()) {
        std::cerr << "error: unable to parse model config: " << err << std::endl;
        exit(1);
    }
    if (s->verbose) {
        std::cout << "\nModel Config:\n" << model_config << std::endl;
    }

    // check input/output num in metadata
    const auto& input_itr = model_metadata_json.FindMember("inputs");
    size_t input_count = 0;
    if (input_itr != model_metadata_json.MemberEnd()) {
        input_count = input_itr->value.Size();
    }
    if (input_count != 1) {
        std::cerr << "expecting 1 input in model, got " << input_count
                  << std::endl;
        exit(1);
    }

    const auto& output_itr = model_metadata_json.FindMember("outputs");
    size_t output_count = 0;
    if (output_itr != model_metadata_json.MemberEnd()) {
        output_count = output_itr->value.Size();
    }
    if (output_count != 1) {
        std::cerr << "expecting 1 output in model, got " << output_count
                  << std::endl;
        exit(1);
    }

#if 0
    // check input/output num in config
    const auto& input_config_itr = model_config_json.FindMember("input");
    input_count = 0;
    if (input_config_itr != model_config_json.MemberEnd()) {
        input_count = input_config_itr->value.Size();
    }
    if (input_count != 1) {
        std::cerr << "expecting 1 input in model configuration, got " << input_count
                  << std::endl;
        exit(1);
    }

    const auto& output_config_itr = model_config_json.FindMember("output");
    output_count = 0;
    if (output_config_itr != model_config_json.MemberEnd()) {
        output_count = output_config_itr->value.Size();
    }
    if (output_count != 1) {
        std::cerr << "expecting 1 output in model configuration, got " << output_count
                  << std::endl;
        exit(1);
    }
#endif

    // parse input metadata & config
    const auto& input_metadata = *input_itr->value.Begin();
    //const auto& input_config = *input_config_itr->value.Begin();

    const auto& input_name_itr = input_metadata.FindMember("name");
    if (input_name_itr == input_metadata.MemberEnd()) {
        std::cerr << "input missing name in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    std::string input_name = std::string(input_name_itr->value.GetString(), input_name_itr->value.GetStringLength());

    const auto& input_dtype_itr = input_metadata.FindMember("datatype");
    if (input_dtype_itr == input_metadata.MemberEnd()) {
        std::cerr << "input missing datatype in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    std::string input_type = std::string(input_dtype_itr->value.GetString(), input_dtype_itr->value.GetStringLength());

    // check input layout (NCHW/NHWC) and get shape
    const auto& input_shape_itr = input_metadata.FindMember("shape");
    if (input_shape_itr == input_metadata.MemberEnd()) {
        std::cerr << "input missing shape in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    size_t input_shape_size = input_shape_itr->value.Size();
    assert(input_shape_size == 4);

    std::string input_layout;
    int input_batch, input_height, input_width, input_channel;
    if (input_shape_itr->value[1].GetInt() == 3) {
        // NCHW
        input_layout = "NCHW";
        input_batch = input_shape_itr->value[0].GetInt();
        input_channel = input_shape_itr->value[1].GetInt();
        input_height = input_shape_itr->value[2].GetInt();
        input_width = input_shape_itr->value[3].GetInt();
    } else {
        // NHWC
        input_layout = "NHWC";
        input_batch = input_shape_itr->value[0].GetInt();
        input_height = input_shape_itr->value[1].GetInt();
        input_width = input_shape_itr->value[2].GetInt();
        input_channel = input_shape_itr->value[3].GetInt();
    }

    std::cout << "input tensor info: "
              << "name " << input_name << ", "
              << "type " << input_type << ", "
              << "shape_size " << input_shape_size << ", "
              << "layout " << input_layout << ", "
              << "batch " << input_batch << ", "
              << "height " << input_height << ", "
              << "width " << input_width << ", "
              << "channels " << input_channel << "\n";

    // assume input tensor type is fp32
    assert(input_type == "FP32");
    assert(input_batch == -1);
    input_batch = 1;
    std::vector<int64_t> input_shape{input_batch,
                                     input_shape_itr->value[1].GetInt(),
                                     input_shape_itr->value[2].GetInt(),
                                     input_shape_itr->value[3].GetInt()};

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

    if (input_layout == "NCHW") {
        // convert image data from NHWC to NCHW
        uint8_t* reorderImage = image_reorder(cropedImage, input_width, input_height, input_channel);
        // free croped image
        free(cropedImage);
        cropedImage = nullptr;
        targetImage = reorderImage;
    }

    // fill input data
    int data_num = input_batch * input_channel * input_height * input_width;
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
    const auto& output_metadata = *output_itr->value.Begin();
    //const auto& output_config = *output_config_itr->value.Begin();

    const auto& output_name_itr = output_metadata.FindMember("name");
    if (output_name_itr == output_metadata.MemberEnd()) {
        std::cerr << "output missing name in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    std::string output_name = std::string(output_name_itr->value.GetString(), output_name_itr->value.GetStringLength());

    const auto& output_dtype_itr = output_metadata.FindMember("datatype");
    if (output_dtype_itr == output_metadata.MemberEnd()) {
        std::cerr << "output missing datatype in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    std::string output_type = std::string(output_dtype_itr->value.GetString(), output_dtype_itr->value.GetStringLength());

    // get output tensor info, assume only 1 output tensor (scores)
    // "image_input": batch_size x 3 x 224 x 224
    // "scores": batch_size x num_classes
    const auto& output_shape_itr = output_metadata.FindMember("shape");
    if (output_shape_itr == output_metadata.MemberEnd()) {
        std::cerr << "output missing shape in the metadata for model'"
                  << model_metadata_json["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    size_t output_shape_size = output_shape_itr->value.Size();
    assert(output_shape_size == 2);

    int output_batch = output_shape_itr->value[0].GetInt();
    int output_classes = output_shape_itr->value[1].GetInt();

    std::cout << "output tensor info: "
              << "name " << output_name << ", "
              << "type " << output_type << ", "
              << "shape_size " << output_shape_size << ", "
              << "batch " << output_batch << ", "
              << "classes " << output_classes << "\n";

    // check if predict class number matches label file
    assert(num_classes == output_classes);

    // assume output tensor type is fp32
    assert(output_type == "FP32");
    assert(output_batch == -1);

    // set batch size to 1 for output shape
    output_batch = 1;
    std::vector<int64_t> output_shape{output_batch, output_classes};

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
    if (targetImage) {
        free(targetImage);
        targetImage = nullptr;
    }

    if (s->verbose) {
        // show some statistic info
        std::cout << "======Inference Statistics======" << std::endl;
        std::cout << results_ptr->DebugString() << std::endl;

        tc::InferStat infer_stat;
        client->ClientInferStat(&infer_stat);
        std::cout << "======Client Statistics======" << std::endl;
        std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
        std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
        std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
        std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

        std::string model_stat;
        client->ModelInferenceStatistics(&model_stat, s->model_name);
        std::cout << "======Model Statistics======" << std::endl;
        std::cout << model_stat << std::endl;
    }

    return;
}


void display_usage() {
    std::cout
        << "Usage: classifier_http_client\n"
        << "--server_addr, -a: localhost\n"
        << "--server_port, -p: 8000\n"
        << "--model_name, -m: classifier_onnx\n"
        << "--image, -i: image_name.jpg\n"
        << "--classes, -l: classes labels for the model\n"
        << "--top_k, -k: show top k classes result\n"
        << "--input_mean, -b: input mean\n"
        << "--input_std, -s: input standard deviation\n"
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
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:hi:k:l:m:p:s:v:w:", long_options,
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

