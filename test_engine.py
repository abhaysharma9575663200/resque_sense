import cv2

from resq_sense.engine import ResQSenseEngine

engine = ResQSenseEngine("yolov8n.pt")

image = cv2.imread("test_image.jpg")

detections, latency = engine.detect(image)

print("Detections:", detections)
print("Inference time:", latency, "ms")