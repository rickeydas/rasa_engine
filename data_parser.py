import os
import json
import itertools
import operator
import pymongo

import pandas as pd

from bson.objectid import ObjectId
from flashtext import KeywordProcessor

rasa_env = os.getenv("RASA_ENV")
if rasa_env == None or rasa_env == "dev":
    with open("dev_config.json", "r") as f:
        configs = json.load(f)
        config_data = configs["nlu_config"]
        db_user = configs["db_user"]
        db_port = configs["db_port"]
        db_pass = configs["db_pass"]
        db_host = configs["db_host"]
        db_name = configs["db_name"]
elif rasa_env == "prod":
    with open("prod_config.json", "r") as f:
        configs = json.load(f)
        config_data = configs["nlu_config"]
        db_user = configs["db_user"]
        db_port = configs["db_port"]
        db_pass = configs["db_pass"]
        db_host = configs["db_host"]
        db_name = configs["db_name"]

myclient = pymongo.MongoClient(f"mongodb://{db_user}:{db_pass}@{db_host}:{db_port}/")
mydb = myclient[db_name]

def get_intents(company_id):
    mycol = mydb["intents"]
    myquery = { "companyId": company_id }
    mydoc = mycol.find(myquery)
    return list(mydoc)

def get_entities(company_id):
    mycol = mydb["entities"]
    myquery = { "company": ObjectId(company_id) }
    mydoc = mycol.find(myquery)
    return list(mydoc)

def get_triggers(company_id):
    mycol = mydb["campaigns"]
    myquery = { "company": ObjectId(company_id) }
    mydoc = mycol.find(myquery)
    return list(mydoc)

def get_data_from_db(company_id):
    entity_data, intent_data, trigger_data = get_entities(company_id), get_intents(company_id), get_triggers(company_id)
    return entity_data, config_data, intent_data, trigger_data

def create_dictionary(data):
    value_dict = KeywordProcessor()
    entity_dict = KeywordProcessor()
    
    for entities in data:
        for synonym in entities['entityValues']:
            value_dict.add_keywords_from_dict({synonym['value']:list(set(synonym['synonyms']+[synonym['value']]))})
            entity_dict.add_keywords_from_dict({entities['name']:list(set(synonym['synonyms']+[synonym['value']]))})
    return value_dict, entity_dict

def format_training_phrase(txt, value_dict, entity_dict):
    replace_kp = KeywordProcessor()
    values = value_dict.extract_keywords(txt, span_info=True)
    entities = entity_dict.extract_keywords(txt)
    list(map(lambda x : replace_kp.add_keyword(txt[x[0][1]:x[0][2]], f"[{txt[x[0][1]:x[0][2]]}] {{'value':'{x[0][0]}', 'entity':'{x[1]}'}}"), list(zip(values, entities))))
    return replace_kp.replace_keywords(txt)

def rasa_format(entity_data, intent_data, trigger_data):
    value_dict, entity_dict = create_dictionary(entity_data)
    trigger_phrases = list(map(lambda x: (x["triggerMessages"], x["name"]), trigger_data))
    trigger_phrases = list(map(lambda x: list(zip(x[0], len(x[0])*[x[1]])), trigger_phrases))
    trigger_phrases = [item for sublist in trigger_phrases for item in sublist]
    trigger_phrases = list(map(lambda x: (x[0]['message'], x[1]), trigger_phrases))
    training_phrases = list(map(lambda x: (x['trainingPhrasesParts'], x['displayName']), intent_data))
    training_phrases = list(map(lambda x: list(zip(x[0], len(x[0])*[x[1]])), training_phrases))
    training_phrases = [item for sublist in training_phrases for item in sublist]
    training_phrases = list(map(lambda x: (x[0]['text'], x[1]), training_phrases))
    training_phrases = training_phrases + trigger_phrases
    return list(map(lambda x: (x[1], format_training_phrase(x[0], value_dict, entity_dict)), training_phrases)), value_dict.get_all_keywords()

def write_rasa_yml(training_phrases, synonyms):
    out = []
    synonyms = list(zip(synonyms.values(), synonyms.keys()))
    it = [tuple(x) for x in pd.DataFrame(training_phrases, columns=['value', 'synonym']).groupby('value')['synonym'].apply(list).reset_index().to_numpy()]
    for key, subiter in it:
        formatted = '\n   - '.join(subiter)               
        out.append(f"\n- intent: {key}\n  examples: |\n   - {formatted}")
    it = [tuple(x) for x in pd.DataFrame(synonyms, columns=['value', 'synonym']).groupby('value')['synonym'].apply(list).reset_index().to_numpy()]
    for key, subiter in it:
        formatted = '\n   - '.join(subiter)               
        out.append(f"\n- synonym: {key}\n  examples: |\n   - {formatted}")
    return "version: '2.0'\n\nnlu:" + ''.join(out)

def get_data(company_id):
    entity_data, config_data, intent_data, trigger_data = get_data_from_db(company_id)
    training_phrases, synonyms = rasa_format(entity_data, intent_data, trigger_data)
    data = write_rasa_yml(training_phrases, synonyms)
    return {"nlu":data, "config_data":config_data}
