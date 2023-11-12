import requests
import json


# url
url = "http://0.0.0.0:1211/api/triton/jointbert/"

# sentence
sentence = "customer service"

# payload
payload = {'data': sentence}

# response
res = requests.post(url, json=payload)
res = json.loads(res.text)
print(res['output'])