import requests

try:
    response = requests.get("http://localhost:1234/v1/models")
    if response.status_code == 200:
        data = response.json()
        print("Available models:")
        for model in data['data']:
            print(f" - {model['id']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error: {e}")
