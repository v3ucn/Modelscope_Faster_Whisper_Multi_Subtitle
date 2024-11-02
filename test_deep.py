import httpx, json

deeplx_api = "http://127.0.0.1:1188/translate"

data = {
	"text": "Hello World",
	"source_lang": "EN",
	"target_lang": "ZH"
}

# JA KO

post_data = json.dumps(data)
r = httpx.post(url = deeplx_api, data = post_data).json()
print(r["data"])