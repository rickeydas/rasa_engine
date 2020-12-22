import json
import requests
import labelconfig

# stanford api
# url = 'http://0.0.0.0:9000/?properties=%7B%22annotators%22%3A%22tokenize%2Cssplit%2Cpos%2Cner%22%2C%22outputFormat%22%3A%22json%22%7D'
# headers = {"Content-Type": "application/json"}
LABELS = labelconfig.label_config["LABELS"]

def normalize_entity(text):
    url = "http://corenlp.run/?properties=%7B%22annotators%22%3A%20%22tokenize%2Cssplit%2Cpos%2Cner%2Cdepparse%2Copenie%22%2C%20%22date%22%3A%20%222020-12-21T11%3A18%3A47%22%7D&pipelineLanguage=en"
    myobj = {'text': text}

    try:
        output = requests.post(url, data = myobj)
        sentences = output.json()["sentences"]
        finalOutput = []
        for sentence in sentences:
            for entity in sentence["entitymentions"]:
                outputStructure = {
                        "start": entity["characterOffsetBegin"] - 5,
                        "end": entity["characterOffsetEnd"] - 5,
                        "entity": entity["text"],
                        "type": entity["ner"],
                        "normalizedNER": ""
                    }
                if 'normalizedNER' in entity:
                    outputStructure["normalizedNER"] = "-".join(entity["normalizedNER"].split("T")[0].split("-")[::-1])
                finalOutput.append(outputStructure)
        
        for i, item in enumerate(finalOutput):
            for label in LABELS:
                if item["type"] in label["name"]:
                    finalOutput[i]["type"] = label["key"]
        
        return json.dumps(finalOutput)
    except requests.exceptions.HTTPError as errh:
        return "An Http Error occurred:" + repr(errh)
    except requests.exceptions.ConnectionError as errc:
        return "An Error Connecting to the API occurred:" + repr(errc)
    except requests.exceptions.Timeout as errt:
        return "A Timeout Error occurred:" + repr(errt)
    except requests.exceptions.RequestException as err:
        return "An Unknown Error occurred" + repr(err)

# print(normalize_entity('my job at California will be done by next monday. I will buy mercedez car next year april. I will go to London by 20 july next year. He might leave today'))