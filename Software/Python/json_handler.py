from eval_frame import EvalFrame
import json


class JSONFileHandler:

    def __init__(self, jsonFilePath):
        self.jsonPath = jsonFilePath    # File directory path
        self.json = None                # converted json as a dict
        self.usefulKeys = ["items", "id", "annotations", "label_id", "points"]

        self.openJSONFile()

    # Opens JSON file and converts to python dict
    # Dumps: Dict -> JSON (Dump for file writing)
    # Load: JSON -> Dict
    def openJSONFile(self):
        with open(self.jsonPath, "r") as json_file:
            self.json = json.load(json_file)

            print("JSON Conv:",self.json)

            json_file.close()

    # Writes new data to JSON file, taking a Python dict
    def writeJSONFile(self,newData):
        with open(self.jsonPath, "w") as json_file:
            json.dump(newData, json_file)

            json_file.close()

    def compileAnnotatedFrames(self):
        evalFrames = []  # list of objects containing hand-annotated frames

        items = self.json.get('items')
        for item in items:
            frame_number, annotations = dictIterator(item, self.usefulKeys)
            newFrame = EvalFrame(frame_number, annotations)
            evalFrames.append(newFrame)
            print(f"Frame Number: {frame_number},  Annotations: {annotations}")

        return evalFrames

# Dict iterator
def dictIterator(passedDict, usefulKeys):
    newFrameNumber = None
    newAnnotations = []

    for key in passedDict:
        if key in usefulKeys:  # First checks if value should be read
            value = passedDict.get(key)
            if isinstance(value, dict):  # If found nested dict, run through it
                dictIterator(value, usefulKeys)

            elif isinstance(value, list):  # If found nested list, "
                if key == 'annotations' and value:  # Check if current key value has annotation
                    for annotation in value: # Ripping necessary information from CVAT Garbage
                        annotationDict = {}

                        annotationDict["label_id"] = annotation['label_id']
                        annotationDict["points"] = annotation['points']

                        newAnnotations.append(annotationDict)
                else:
                    arrayIterator(value, usefulKeys)  # Passes list to iterator

            else:
                if key == 'id' and isinstance(value, str):  # Check for correct id, and then get frame number
                    image_file = value.split("_")
                    if image_file[2].isnumeric():  # Validating split file name number
                        frame_number = int(image_file[2])
                        newFrameNumber = frame_number
                    else:
                        raise ValueError("Fatal Error: Failed to find frame number.")

    return newFrameNumber, newAnnotations


# Array iterator
def arrayIterator(passedArray, usefulKeys):

    for item in passedArray:
        if isinstance(item, dict):
            dictIterator(item, usefulKeys)
        elif isinstance(item, list):
            arrayIterator(item, usefulKeys)
        elif isinstance(item, int):
            print("Found uncaught int")
            pass
        else:
            raise TypeError("Unexpected item found in array")



if __name__ == "__main__":
    myJsonFile = JSONFileHandler("data/eval/default.json")

    evalFrames = myJsonFile.compileAnnotatedFrames()

    print("--------------------------------------------------------------------------------------"
          "\n\n\n"
          "--------------------------------------------------------------------------------------")
    for index, frame in enumerate(evalFrames):
        print(f"{index}: Frame Number {frame.frame_number}, Annotations {frame.annotations}")
    # dictIterator()

