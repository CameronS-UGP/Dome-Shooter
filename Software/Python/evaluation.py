import json


class Annotations:

    def __init__(self, jsonFilePath):
        self.jsonFile = jsonFilePath
        self.json = None

        self.openJSONFile()

    # Opens JSON file and converts to python dict
    # Dump: Dict -> JSON
    # Load: JSON -> Dict
    def openJSONFile(self):
        with open(self.jsonFile, "r") as json_file:
            self.json = json.load(json_file)

            print("json_file:",json_file)
            print("JSON Conv:",self.json)



if __name__ == "__main__":
    annotations = Annotations("data/eval/default.json")
