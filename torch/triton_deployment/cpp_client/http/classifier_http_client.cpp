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
    std::string server_port = "8001";
    std::string model_name = "classifier_onnx";
    std::string input_img_name = "./dog.jpg";
    std::string classes_file_name = "./classes.txt";
    bool verbose = false;
};


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


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

    // get model metadata
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

#if 0
    if ((std::string(model_metadata_json["name"].GetString()))
            .compare(s->model_name) != 0) {
      std::cerr << "error: unexpected model metadata: " << model_metadata
                << std::endl;
      exit(1);
    }
#endif


    if (s->verbose) {
        // show some statistic info
        //std::cout << "======Inference Statistics======" << std::endl;
        //std::cout << results_ptr->DebugString() << std::endl;

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


#if 0

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


namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 1) || (shape[1] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("INT32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t-i <none|gzip|deflate>" << std::endl;
  std::cerr << "\t-o <none|gzip|deflate>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "\t--verify-peer" << std::endl;
  std::cerr << "\t--verify-host" << std::endl;
  std::cerr << "\t--ca-certs" << std::endl;
  std::cerr << "\t--cert-file" << std::endl;
  std::cerr << "\t--key-file" << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl
      << "For -i, it sets the compression algorithm used for sending request "
         "body."
      << "For -o, it sets the compression algorithm used for receiving "
         "response body."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  auto request_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  auto response_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  long verify_peer = 1;
  long verify_host = 2;
  std::string cacerts;
  std::string certfile;
  std::string keyfile;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"verify-peer", 1, 0, 0}, {"verify-host", 1, 0, 1}, {"ca-certs", 1, 0, 2},
      {"cert-file", 1, 0, 3},   {"key-file", 1, 0, 4},    {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:H:i:o:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        verify_peer = std::atoi(optarg);
        break;
      case 1:
        verify_host = std::atoi(optarg);
        break;
      case 2:
        cacerts = optarg;
        break;
      case 3:
        certfile = optarg;
        break;
      case 4:
        keyfile = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'i': {
        std::string arg = optarg;
        if (arg == "gzip") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case 'o': {
        std::string arg = optarg;
        if (arg == "gzip") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  tc::HttpSslOptions ssl_options;
  ssl_options.verify_peer = verify_peer;
  ssl_options.verify_host = verify_host;
  ssl_options.ca_info = cacerts;
  ssl_options.cert = certfile;
  ssl_options.key = keyfile;
  // Create a InferenceServerHttpClient instance to communicate with the
  // server using HTTP protocol.
  std::unique_ptr<tc::InferenceServerHttpClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerHttpClient::Create(&client, url, verbose, ssl_options),
      "unable to create http client");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  std::vector<int32_t> input0_data(16);
  std::vector<int32_t> input1_data(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = i;
    input1_data[i] = 1;
  }

  std::vector<int64_t> shape{1, 16};

  // Initialize the inputs with the data.
  tc::InferInput* input0;
  tc::InferInput* input1;

  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"),
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
  FAIL_IF_ERR(
      tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"),
      "unable to get INPUT1");
  std::shared_ptr<tc::InferInput> input1_ptr;
  input1_ptr.reset(input1);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT0");
  FAIL_IF_ERR(
      input1_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input1_data[0]),
          input1_data.size() * sizeof(int32_t)),
      "unable to set data for INPUT1");

  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
  // Empty output vector will request data for all the output tensors from
  // the server.
  std::vector<const tc::InferRequestedOutput*> outputs = {};

  tc::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(
          &results, options, inputs, outputs, http_headers, tc::Parameters(),
          request_compression_algorithm, response_compression_algorithm),
      "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT0", results_ptr);
  ValidateShapeAndDatatype("OUTPUT1", results_ptr);

  // Get pointers to the result returned...
  int32_t* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  int32_t* output1_data;
  size_t output1_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT1", (const uint8_t**)&output1_data, &output1_byte_size),
      "unable to get result data for 'OUTPUT1'");
  if (output0_byte_size != 64) {
    std::cerr << "error: received incorrect byte size for 'OUTPUT1': "
              << output0_byte_size << std::endl;
    exit(1);
  }

  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input1_data[i] << " = "
              << *(output0_data + i) << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << *(output1_data + i) << std::endl;

    if ((input0_data[i] + input1_data[i]) != *(output0_data + i)) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if ((input0_data[i] - input1_data[i]) != *(output1_data + i)) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }

  // Get full response
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
  FAIL_IF_ERR(
      client->ModelInferenceStatistics(&model_stat, model_name),
      "unable to get model statistics");
  std::cout << "======Model Statistics======" << std::endl;
  std::cout << model_stat << std::endl;

  std::cout << "PASS : Infer" << std::endl;

  return 0;
}

#endif

