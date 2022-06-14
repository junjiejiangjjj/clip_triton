import json
from towhee.models import clip

import triton_python_backend_utils as pb_utils



class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.tokenize = clip.tokenize
        
    def execute(self, requests):
        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in0 = in0.as_numpy()
            text = in0[0].decode('utf-8')
            output = self.tokenize(text)
            output = output.numpy()
            out = pb_utils.Tensor("OUTPUT0", output.astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
