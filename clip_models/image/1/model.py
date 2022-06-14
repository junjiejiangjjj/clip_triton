import json
import torch
from PIL import Image
from my_lib import MyLib
import triton_python_backend_utils as pb_utils

from torchvision import transforms


class TritonPythonModel:

    def initialize(self, args):
        self._my_lit = MyLib()
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.tfms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def execute(self, requests):
        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in0 = in0.as_numpy()
            image_path = in0[0].decode('utf-8')
            with Image.open(image_path) as img:
                output = self.tfms(img).unsqueeze(0)

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
