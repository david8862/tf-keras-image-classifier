//
//  classifier_grpc_client.cpp
//  Triton gRPC client
//
//  Created by david8862 on 2024/01/23.
//
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
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


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;
    bool verbose = false;
    std::string url = s->server_addr + ":" + s->server_port;
    std::string model_version = "";
    tc::Headers http_headers;

    // create InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    tc::Error err;

    err = tc::InferenceServerGrpcClient::Create(&client, url, verbose);
    if (!err.IsOk()) {
        std::cerr << "unable to create grpc client: " << err << std::endl;
        exit(1);
    };

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


    inference::ModelConfigResponse model_config;
    err = client->ModelConfig(&model_config, s->model_name, model_version, http_headers);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }

    std::cout << "model input size: " << model_config.config().input().size() << std::endl;
    std::cout << "model output size: " << model_config.config().output().size() << std::endl;


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

    auto input_config = model_config.config().input(0);
    auto output_config = model_config.config().output(0);

    std::string input_name = input_config.name();
    std::string input_type = grpc_data_type_str(input_config.data_type());
    std::string output_name = output_config.name();
    std::string output_type = grpc_data_type_str(output_config.data_type());

    std::cout << "input_name: " << input_name << "\n";
    std::cout << "output_name: " << output_name << "\n";
    std::cout << "input_type: " << input_type << "\n";
    std::cout << "output_type: " << output_type << "\n";

    auto input_dims = input_config.dims().size();
    std::cout << "input_dims: " << input_dims << "\n";





    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    std::cout << "num_classes: " << num_classes << "\n";


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
