from eval_frame import EvalFrame
import json

# script written to take a

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
            json_file.close()

    # Writes new data to JSON file, taking a Python dict
    def writeJSONFile(self,newData):
        with open(self.jsonPath, "w") as json_file:
            json.dump(newData, json_file)

            json_file.close()

    # Function for taking an annotation (datumaru) JSON file and converting to annotation objects
    def compileAnnotatedFrames(self):
        evalFrames = []  # list of objects containing hand-annotated frames
        evalFrameIndex = [] # list of corresponding frame number for searching

        items = self.json.get('items')  # isolate the items key which contains all frames and annotation values
        for item in items:  # iterating through each frame in dataset
            frame_number, annotations = dictIterator(item, self.usefulKeys)  # outputs extracted frame number and annotations (if any) per frame in json file
            evalFrames.append(EvalFrame(frame_number, annotations))  # creates a new eval object for evalFrames[]
            evalFrameIndex.append(frame_number)
            # print(f"Frame Number: {frame_number},  Annotations: {annotations}")

        return evalFrames, evalFrameIndex

# Dict iterator
def dictIterator(passedDict, usefulKeys):
    newFrameNumber = None
    newAnnotations = []

    for key in passedDict:
        if key in usefulKeys:  # First checks if value should be read
            value = passedDict.get(key)
            if isinstance(value, dict):  # If found nested dict, run through it
                dictIterator(value, usefulKeys)

            elif isinstance(value, list):  # If found nested list
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
        else:
            raise TypeError("Unexpected item found in array")



if __name__ == "__main__":
    myJsonFile = JSONFileHandler("data/eval/default.json")

    evalFrames, indexList = myJsonFile.compileAnnotatedFrames()

    print("--------------------------------------------------------------------------------------"
          "\n\n\n"
          "--------------------------------------------------------------------------------------")
    for index, frame in enumerate(evalFrames):
        print(f"{index}: Frame Number {frame.frame_number}, Annotations {frame.annotations}")
    # dictIterator()

