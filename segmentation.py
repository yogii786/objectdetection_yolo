import torch
from PIL import Image
import io
import numpy as np
def get_yolov5():
    model = torch.hub.load('/home/yogeshbhati/Yogesh_Bhati/objectdetection_yolo/yolov5', 'custom', '/home/yogeshbhati/Yogesh_Bhati/objectdetection_yolo/model/best.pt', source = "local")
    model.conf = 0.5
    return model
def get_image_from_bytes(binary_image):
    input_image =Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    # resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((224, 224))
    resized_image = np.array(resized_image)

    return resized_image