from imageai.Detection import ObjectDetection
import sys

detector = ObjectDetection()

model_path = "media/yolo-tiny.h5"
input_path = sys.argv[1]
output_path = "media/output.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])