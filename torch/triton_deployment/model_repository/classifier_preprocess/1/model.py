#!/usr/bin/env python3
# -*- coding=utf-8 -*-
#
# Reference from:
# https://github.com/triton-inference-server/python_backend/blob/main/examples/preprocessing/model.py
#
import io
import json
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # get input/output config
        self.input_config = model_config["input"]
        self.output_config = model_config["output"]

        assert len(self.input_config) == 1, 'invalid input number.'
        assert len(self.output_config) == 1, 'invalid output number.'

        # check if output layout is NHWC or NCHW
        assert len(self.output_config[0]["dims"]) == 3, 'invalid output dims.'

        if self.output_config[0]["dims"][0] == 3:
            self.output_layout = "NCHW"
            channel, height, width = self.output_config[0]["dims"] #NCHW
        else:
            self.output_layout = "NHWC"
            height, width, channel = self.output_config[0]["dims"] #NHWC
        self.output_shape = (height, width)


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output_dtype = pb_utils.triton_string_to_numpy(self.output_config[0]["data_type"])
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, self.input_config[0]["name"])

            # process input image
            img = input_tensor.as_numpy()
            image = Image.open(io.BytesIO(img.tobytes()))
            resized_image = image.resize(self.output_shape[::-1], Image.BICUBIC)
            image_data = np.asarray(resized_image).astype(np.float32)
            image_data = image_data / 127.5 - 1

            if self.output_layout == "NCHW":
                # convert image data to channel first
                image_data = np.transpose(image_data, (2, 0, 1))
            image_out = np.expand_dims(image_data, axis=0)

            # create input tensor
            output_tensor = pb_utils.Tensor(self.output_config[0]["name"], image_out.astype(output_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")

