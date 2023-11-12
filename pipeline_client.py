import tritonclient.http as tritonhttpclient
import numpy as np
from tritonclient.utils import *

input_name = ["input_text"]
output_name = ["intent_slot"]
client = tritonhttpclient.InferenceServerClient('0.0.0.0:8050')
sentence = "customer service is not good"

input_feature = np.array([bytes(sentence, 'utf8')], dtype=np.bytes_).reshape(1, -1)
print(input_feature.shape)

input0 = tritonhttpclient.InferInput(input_name[0], input_feature.shape, np_to_triton_dtype(input_feature.dtype))
input0.set_data_from_numpy(input_feature)
out = tritonhttpclient.InferRequestedOutput(output_name[0], binary_data=False)

response = client.infer(model_name='pipeline', inputs=[input0], outputs=[out])

res = response.as_numpy(output_name[0])
print(res[0])
# final_res = [i.decode("utf-8") for i in res]
# print(final_res)

