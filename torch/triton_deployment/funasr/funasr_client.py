#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton grpc client for FunASR model inference service
"""
import os, sys, argparse
import glob
import numpy as np
import math
from tqdm import tqdm
import soundfile as sf
import librosa

import tritonclient.grpc as grpcclient
#import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype



class OfflineSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, sample_rate, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name
        self.sample_rate = sample_rate

    def recognize(self, wav_file, idx=0):
        #waveform, sample_rate = sf.read(wav_file)
        waveform, sample_rate = librosa.load(wav_file, sr=self.sample_rate)
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        # better pad waveform to nearest length here
        # target_seconds = math.cel(len(waveform) / sample_rate)
        # target_samples = np.zeros([1, target_seconds  * sample_rate])
        # target_samples[0][0: len(waveform)] = waveform
        # samples = target_samples
        sequence_id = 10086 + idx
        result = ""
        inputs = [
            self.protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            self.protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype)),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        response = self.triton_client.infer(
            self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=str(sequence_id),
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=None
        )
        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            result = b" ".join(decoding_results).decode("utf-8")
        else:
            result = decoding_results.decode("utf-8")
        return [result]


class StreamingSpeechClient(object):
    def __init__(self, triton_client, model_name, protocol_client, sample_rate, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name
        self.sample_rate = sample_rate
        chunk_size = args.chunk_size
        subsampling = args.subsampling
        context = args.context
        frame_shift_ms = args.frame_shift_ms
        frame_length_ms = args.frame_length_ms
        # for the first chunk
        # we need additional frames to generate
        # the exact first chunk length frames
        # since the subsampling will look ahead several frames
        first_chunk_length = (chunk_size - 1) * subsampling + context
        add_frames = math.ceil(
            (frame_length_ms - frame_shift_ms) / frame_shift_ms)
        first_chunk_ms = (first_chunk_length + add_frames) * frame_shift_ms
        other_chunk_ms = chunk_size * subsampling * frame_shift_ms
        self.first_chunk_in_secs = first_chunk_ms / 1000
        self.other_chunk_in_secs = other_chunk_ms / 1000

    def recognize(self, wav_file, idx=0):
        #waveform, sample_rate = sf.read(wav_file)
        waveform, sample_rate = librosa.load(wav_file, sr=self.sample_rate)
        wav_segs = []
        i = 0
        while i < len(waveform):
            if i == 0:
                stride = int(self.first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i:i + stride])
            else:
                stride = int(self.other_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i:i + stride])
            i += len(wav_segs[-1])

        results = ""
        sequence_id = idx + 10086
        #pbar = tqdm(total=len(wav_segs), desc='Stream Inference')
        # simulate streaming
        for idx, seg in enumerate(wav_segs):
            chunk_len = len(seg)
            if idx == 0:
                chunk_samples = int(self.first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(self.other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            expect_input[0][0:chunk_len] = seg
            input0_data = expect_input
            input1_data = np.array([[chunk_len]], dtype=np.int32)

            inputs = [
                self.protocol_client.InferInput(
                    "WAV",
                    input0_data.shape,
                    np_to_triton_dtype(input0_data.dtype),
                ),
                self.protocol_client.InferInput(
                    "WAV_LENS",
                    input1_data.shape,
                    np_to_triton_dtype(input1_data.dtype),
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            outputs = [
                self.protocol_client.InferRequestedOutput("TRANSCRIPTS")
            ]
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            response = self.triton_client.infer(
                self.model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=str(1),
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
                priority=0,
                timeout=None
            )
            idx += 1
            result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
            results += result
            print("Get response from {}th chunk: {}".format(idx, results))
            #pbar.update(1)
        #pbar.close()
        return results


def funasr_inference(server_addr, server_port, model_name, audio_files, sample_rate, streaming, args):
    # init triton grpc client
    server_url = server_addr + ':' + server_port
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False, ssl=False)

    # check if triton server & target model is ready
    assert triton_client.is_server_live() & triton_client.is_server_ready(), 'Triton server is not ready'
    assert triton_client.is_model_ready(model_name), 'model ' + model_name + ' is not ready'

    # get input/output config & metadata, here
    # we use metadata to parse model info
    model_metadata = triton_client.get_model_metadata(model_name)
    model_config = triton_client.get_model_config(model_name)

    # input:
    # name:"WAV", datatype:"FP32", shape: (-1, -1)
    # name:"WAV_LENS", datatype: "INT32", shape: (-1, 1)
    #
    # output:
    # name: "TRANSCRIPTS", datatype: "BYTES", shape: (-1, 1)
    #
    inputs_metadata = model_metadata.inputs
    outputs_metadata = model_metadata.outputs

    assert len(inputs_metadata) == 2, 'invalid input number.'
    assert len(outputs_metadata) == 1, 'invalid output number.'

    input_0_name = inputs_metadata[0].name
    input_0_type = inputs_metadata[0].datatype
    input_1_name = inputs_metadata[1].name
    input_1_type = inputs_metadata[1].datatype
    output_name = outputs_metadata[0].name
    output_type = outputs_metadata[0].datatype

    # check input & output metadata
    assert input_0_type == 'FP32', 'invalid input_0 type.'
    assert input_1_type == 'INT32', 'invalid input_1 type.'
    assert output_type == 'BYTES', 'invalid output type.'

    assert len(outputs_metadata[0].shape) == 2, 'invalid output shape.' # (-1, 1)

    if streaming:
        speech_client = StreamingSpeechClient(triton_client, model_name, grpcclient, sample_rate, args)
    else:
        speech_client = OfflineSpeechClient(triton_client, model_name, grpcclient, sample_rate, args)

    # loop the sample list to predict on each image
    for audio_file in audio_files:
        result = speech_client.recognize(audio_file)
        print("ASR result: {}".format(result))


def main():
    parser = argparse.ArgumentParser(description='Triton grpc client for FunASR model inference server')
    parser.add_argument('--server_addr', type=str, required=False, default='localhost',
        help='triton server address, default=%(default)s')
    parser.add_argument('--server_port', type=str, required=False, default='8001',
        help='triton server port (8000 for http & 8001 for grpc), default=%(default)s')
    parser.add_argument('--model_name', type=str, required=False, choices=["attention_rescoring", "streaming_paraformer"], default='attention_rescoring',
        help='model name for inference, default=%(default)s')
    parser.add_argument('--audio_path', type=str, required=True,
        help="audio file or directory to inference")
    parser.add_argument('--sample_rate', type=int, required=False, default=16000,
        help="sample rate used by model, default=%(default)s")
    parser.add_argument('--streaming', default=False, action="store_true",
        help='Whether to run streaming inference model')

    # below arguments are for streaming inference
    # Please check onnx_config.yaml and train.yaml
    parser.add_argument('--frame_length_ms', type=int, required=False, default=25,
        help="audio frame length in ms, default=%(default)s")
    parser.add_argument('--frame_shift_ms', type=int, required=False, default=10,
        help="audio frame shift length in ms, default=%(default)s")
    parser.add_argument('--chunk_size', type=int, required=False, default=16,
        help="audio chunk size, default=%(default)s")
    parser.add_argument('--context', type=int, required=False, default=7,
        help="subsampling context, default=%(default)s")
    parser.add_argument("--subsampling", type=int, required=False, default=4,
        help="subsampling rate, default=%(default)s")

    args = parser.parse_args()

    # get audio file list or single audio
    if os.path.isdir(args.audio_path):
        audio_files = glob.glob(os.path.join(args.audio_path, '*.wav'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        audio_files = [args.audio_path]

    funasr_inference(args.server_addr, args.server_port, args.model_name, audio_files, args.sample_rate, args.streaming, args)



if __name__ == "__main__":
    main()
