# gender_detection
Docker application for making gender detection

Using steps:

1. Define enviroment variables (file .env)

2. Run application

```bash
 docker-compose up --build -d
```

3. For using API you need make payload for POST request:

```bash
names = ['Brian Oliver V', 'Wilburn Romero', 'Bridget Conti']
payload = {'data': names, 'date_request':datetime.now().isoformat()}

url = 'http://{host}:{ACCESS_PORT}/{ACCESS_NAME}/v1/predict_data'

response = requests.post(url, json = payload, timeout=6)

response.json() #contains values with next atributes: "gender", "name", "probability_F", "probability_M",  "probability_unknown"
```

4. For monitoring logs you can use next url:

url = 'http://{host}:{ACCESS_PORT}