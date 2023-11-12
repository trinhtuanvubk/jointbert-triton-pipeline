# Jointbert Triton


### Installing

- Create a wrap triton image (if needed):
```
docker build -t wrap_triton .
```

- Create conda 
```
conda create -n venv
conda activate venv
pip install -r requirements.txt
```

### Model preparation
- Make sure move `model.onnx` like this figure:

![model tree](figure/jb_fig.png)
### Run Triton server
- To run triton server
```
docker run --gpus=1 -itd --add-host=host.docker.internal:host-gateway -p 8050-8052:8000-8002 -v ${PWD}/model_repository:/models --shm-size 2gb --name jointbert_triton wrap_triton:latest tritonserver --model-repository=/models
```
- To test triton client
```
python3 pipeline_client.py
```

### FastAPI
- To run api:
```
uvicorn api:app --host 0.0.0.0 --port 1211
```

- To test client:
```
python3 test_api.py
```

or go to docs `http://0.0.0.0:1211/docs`