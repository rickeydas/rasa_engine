import os
import json

rasa_env = os.getenv("RASA_ENV")
if rasa_env == "" or rasa_env == "dev":
    with open("dev_config.json", "r") as f:
        configs = json.load(f)
        lower_threshold = configs["lower_threshold"]
        upper_threshold = configs["upper_threshold"]
elif rasa_env == "prod":
    with open("dev_config.json", "r") as f:
        configs = json.load(f)
        lower_threshold = configs["lower_threshold"]
        upper_threshold = configs["upper_threshold"]

def remove_lower_scores(results):
    updated_results = {}
    results = list(map(lambda x: updated_results.update({x['name']: x['confidence']}) if x['confidence']>(lower_threshold/100) else None, results['intent_ranking']))
    results = list(filter(None, results))
    return results

def get_filtered_result(results, contexts):
    if results['intent']['confidence'] > (upper_threshold/100):
        return {"intent":results['intent']['name'], "entity":results['entities']}
    else:
        contexts = list(set(contexts).intersection(set(list(map(lambda x: x['name'], results['intent_ranking'])))))
        results = remove_lower_scores(results)
        results = {context: results[context] for context in contexts}
        results = dict(sorted(results.items(), key=lambda item: item[1]))
        return {"intent":list(results.keys())[0], "entity":results['entities']}
