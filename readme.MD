# Steps
### Start scored api
```
sh start_api.sh docker
```

### sample company id
```
5fc9e4427874e77cc00c97c1
```

### run rasa engine
```
RASA_ENV=dev ./start_rasa.sh
RASA_ENV=prod ./start_rasa.sh
```

### try out apis
```
0.0.0.0:8000/docs
```