import requests

url = 'http://127.0.0.1:5000/predict'

data = {
    'num_links': 1,
    'num_words': 10,
    'has_offer': 1,
    'sender_score': 0,
    'all_caps': 0
}

repsonse = requests.post(url, json=data)
print(repsonse.json())