from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8s.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8s.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/home/user/Code/ai-agent-icon/dataset/data.yaml", epochs=100, imgsz=1280, batch=4)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("/home/user/Code/ai-agent-icon/dataset/test/images")

# Export the model to ONNX format
success = model.export(format="onnx")