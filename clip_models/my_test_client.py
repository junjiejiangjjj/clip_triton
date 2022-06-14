from tritonclient.utils import *
import tritonclient.http as httpclient
import threading

import numpy as np

model_name = "clip_image"

def run():

    with httpclient.InferenceServerClient("localhost:9091") as client:
        inputs = [
            httpclient.InferInput("INPUT", [1, 1], "BYTES")
        ]
    
        image_path = '/home/junjie.jiangjjj/images/3.jpg'
        input_data = np.array([[image_path.encode('utf-8')]], dtype=np.object_)

        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT"),
        ]

        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

        # result = response.get_response()
        return response.as_numpy("OUTPUT")

ret = run()
print(ret.shape)

# ts = []
# for _ in range(100):
#     ts.append(threading.Thread(target=run))

# for t in ts:
#     t.start()

# for t in ts:
#     t.join()
    
