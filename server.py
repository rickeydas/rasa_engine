import os
import yaml
import json
import requests
import shutil

from google.auth import jwt

from typing import Optional
from fastapi import FastAPI, status, BackgroundTasks
from pydantic import BaseModel

from rasa.nlu.train import load_data
from rasa.nlu import train, config
from rasa.nlu.model import Trainer, Interpreter

from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

from data_parser import get_data
from context_parser import get_filtered_result

pubsub_id = "rasa-pubsub"
subscription_id = "rasa_training"
topic_id = "rasa_training"

global nlu_models
nlu_models = {}
root_dir = "./bots/"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

service_account_info = json.load(open("rasa-pubsub-99cc440554fe.json"))

subscriber_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
credentials_sub = jwt.Credentials.from_service_account_info(service_account_info, audience=subscriber_audience)
subscriber = pubsub_v1.SubscriberClient(credentials=credentials_sub)

publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
credentials_pub = credentials_sub.with_claims(audience=publisher_audience)
publisher = pubsub_v1.PublisherClient(credentials=credentials_pub)

topic_path = publisher.topic_path(pubsub_id, topic_id)
subscription_path = subscriber.subscription_path(pubsub_id, subscription_id)

app = FastAPI(title="Rasa Apis", description="rasa apis built in python for Napses.")

class TrainingData(BaseModel):
    company_id: str

class LoadModel(BaseModel):
    company_id: str
        
class ParseMessage(BaseModel):
    company_id: str
    message_id: Optional[str]
    message: str
    contexts: Optional[list]

def load_all_models():
    for company_id in os.listdir(root_dir):
        nlu_models[company_id] = Interpreter.load(root_dir+company_id+"/model")

def train_nlu_model(data, company_id):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(root_dir + company_id):
        os.mkdir(root_dir + company_id)
    with open(root_dir + company_id + "/nlu.yml", 'w') as file:
        file.write(data['nlu'])
    with open(root_dir + company_id + "/config.yml", 'w') as file:
        yaml.dump(data["config_data"], file)
    training_data = load_data(root_dir + company_id + "/nlu.yml")
    trainer = Trainer(config.load(root_dir + company_id + "/config.yml"))
    trainer.train(training_data)
    model_dir = trainer.persist(root_dir + company_id)
    try:
        os.rename(model_dir, root_dir + company_id + "/model")
    except Exception:
        shutil.rmtree(root_dir + company_id + "/model")
        os.rename(model_dir, root_dir + company_id + "/model")

# def load_nlu_model(items : LoadModel):
#     # get nlu model from google storage
#     if items.company_id == "all":
#         load_all_models()
#     else:
#         nlu_models[items.company_id] = Interpreter.load(root_dir + items.company_id + "/model")

def load_nlu_model(company_id):
    if company_id == "all":
        load_all_models()
    else:
        nlu_models[company_id] = Interpreter.load(root_dir + company_id + "/model")

def parse_pubsub_message(message):
    nlu_models[message.company_id].parse(message.message)
    message.ack()

def callback_train(message):
    print(f"Received message :: ", message.data.decode("utf-8"))
    if "train the model" in message.data.decode("utf-8"):
        company_id = message.attributes.get("company_id")
        data = get_data(company_id)
        train_nlu_model(data, company_id)
        load_nlu_model(company_id)
        publisher.publish(topic_path, f"Model Loaded :: {company_id}".encode("utf-8"), company_id=company_id.encode("utf-8"))
        message.ack()
    else:
        pass

load_all_models()
subscriber.subscribe(subscription_path, callback=callback_train)

@app.post("/api/v1/napses/train", status_code=status.HTTP_202_ACCEPTED)
async def start_train_nlu_model(items: TrainingData):
    publisher.publish(topic_path, f"train the model {items.company_id}".encode("utf-8"), company_id = str(items.company_id))
    return {"message":"sucessful"}

# @app.post("/api/v1/napses/load", status_code=status.HTTP_202_ACCEPTED)
# async def start_load_nlu_model(items : LoadModel, background_tasks: BackgroundTasks):
#     background_tasks.add_task(load_nlu_model, items)
#     return {"message":"Loading model in background."}

@app.post("/api/v1/napses/parse", status_code=status.HTTP_200_OK)
async def start_parse_message(items : ParseMessage):
    try:
        result = nlu_models[items.company_id].parse(items.message)
        # requests.post("localhost:8000/results", data=result, verify=False)
        return get_filtered_result(result, items.contexts)
    except KeyError:
        return {"error": f"Model {items.company_id} not loaded. Please load the model first"}
