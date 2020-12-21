import json
import requests
import labelconfig
import pandas as pd
from nltk.corpus import stopwords

global ner_texts
global ner_labels
ner_texts = []
ner_labels = []

# stanford api
url = 'http://0.0.0.0:9000/?properties=%7B%22annotators%22%3A%22tokenize%2Cssplit%2Cpos%2Cner%22%2C%22outputFormat%22%3A%22json%22%7D'
headers = {"Content-Type": "application/json"}
LABELS = labelconfig.label_config["LABELS"]

def normalize_entity():
    pass

# generate POS tags and entities using stanford corenlp
def stanford_pos_tags(val):
    val = val.encode("utf-8").decode("ascii", "ignore")
    tokens = requests.post(url = url, headers = headers, data = val).content
    tokens = tokens.decode('utf8')
    tokens = json.loads(tokens)
    #print(tokens)
    temp_data = []

    for index in tokens['sentences']:
        tokens, tags = [], []
        for token in index['tokens']:
            tokens.append(token['originalText'])
            tags.append(token['pos'])
        temp_data.append([' '.join(tokens), ' '.join(tags)])
        for token in index['entitymentions']:
            #print(token['text'], token['ner'])
            if token['ner'] in ['PERSON', 'MONEY', 'DATE', 'TIME', 'LOCATION', 'ORGANIZATION', 'EMAIL', 'URL', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'NATIONALITY', 'RELIGION', 'CAUSE_OF_DEATH', 'PERCENT', 'TITLE', 'CRIMINAL_CHARGE', 'IDEOLOGY', 'DURATION']:
                if token['text'].lower() not in stopwords.words('english') and token['text'] not in modifiers:
                    ner_texts.append(token['text'])
                    ner_labels.append((token['text'], token['ner']))
    #print(ner_texts)
    return(pd.DataFrame(temp_data, columns=['Sentence', 'Tags']))
