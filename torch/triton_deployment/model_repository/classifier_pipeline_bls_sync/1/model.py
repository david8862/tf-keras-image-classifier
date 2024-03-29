#!/usr/bin/env python3
# -*- coding=utf-8 -*-
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
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # get input/output config
        self.input_config = model_config["input"]
        self.output_config = model_config["output"]

        # 4 input tensors: 1. target model name (model_name)
        #                  2. target model input name (model_input_name)
        #                  3. target model output name (model_output_name)
        #                  4. real input data (input)
        assert len(self.input_config) == 4, 'invalid input number.'
        # 1 output tensor for model output
        assert len(self.output_config) == 1, 'invalid output number.'


    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
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
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            if request.is_cancelled():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Req Cancelled", pb_utils.TritonError.CANCELLED)))
            else:
                # get target model name
                model_name = pb_utils.get_input_tensor_by_name(request, "model_name")
                # target model name string
                model_name_string = model_name.as_numpy()[0].decode('utf-8')

                # get target model input name
                model_input_name = pb_utils.get_input_tensor_by_name(request, "model_input_name")
                # target model input name string
                model_input_name_string = model_input_name.as_numpy()[0].decode('utf-8')

                # get target model output name
                model_output_name = pb_utils.get_input_tensor_by_name(request, "model_output_name")
                # target model output name string
                model_output_name_string = model_output_name.as_numpy()[0].decode('utf-8')

                # get input data tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, "input")

                # Create inference request object
                infer_request = pb_utils.InferenceRequest(
                    model_name=model_name_string,
                    requested_output_names=[model_output_name_string],
                    inputs=[input_tensor],
                )

                # Perform synchronous blocking inference request
                infer_response = infer_request.exec()

                # Make sure that the inference response doesn't have an error. If
                # it has an error and you can't proceed with your model execution
                # you can raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(infer_response.error().message())

                # Create InferenceResponse. You can set an error here in case
                # there was a problem with handling this inference request.
                # Below is an example of how you can set errors in inference
                # response:
                #
                # pb_utils.InferenceResponse(
                #    output_tensors=..., TritonError("An error occurred"))
                #
                # Because the infer_response of the models contains the final
                # outputs with correct output names, we can just pass the list
                # of outputs to the InferenceResponse object.
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=infer_response.output_tensors()
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
