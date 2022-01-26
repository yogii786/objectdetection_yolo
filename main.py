from fastapi import FastAPI, File, UploadFile
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import pandas
model = get_yolov5()
app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
    and return image and json result""",
    version="0.0.1",
)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]
app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')

@app.post("/object-to-json")
async def detect_racoon_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    # print("file: \n\n")
    # print(contents)
    results = model(input_image)
    results.render()

    labels , cord = results.xyxyn [ 0 ][ : , -1 ], results.xyxyn [ 0 ] [ : , :-1 ]

    return {"class": labels.tolist(), "cord": cord.tolist()}


@app.post("/object-to-img")
async def detect_racoon_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(),
media_type="image/jpeg")