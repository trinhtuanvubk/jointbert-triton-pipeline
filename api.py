
import os
import numpy as np
import traceback
import sys
import re
import time
import gc


from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile

import tritonhttpclient
from tritonclient.utils import *


app = FastAPI()

class TextInput(BaseModel):
    data: str


input_name = ["input_text"]
output_name = ["intent_slot"]

client = tritonhttpclient.InferenceServerClient('0.0.0.0:8050')  


@app.get('/')
async def health_check():
    data = {"Status":200}
    return data


@app.post('/api/triton/jointbert')
async def triton_infer(text: TextInput): 

    api_res = {}
    
    input_feature = np.array([bytes(text.data, 'utf8')], dtype=np.bytes_).reshape(1, -1)

    input0 = tritonhttpclient.InferInput(input_name[0], input_feature.shape, np_to_triton_dtype(input_feature.dtype))
    input0.set_data_from_numpy(input_feature)
    out = tritonhttpclient.InferRequestedOutput(output_name[0], binary_data=False)

    response = client.infer(model_name='pipeline', inputs=[input0], outputs=[out])

    res = response.as_numpy(output_name[0])
    print(type(res))
    api_res['output'] = res[0]

    gc.collect()
    return api_res