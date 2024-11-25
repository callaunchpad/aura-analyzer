from typing import Union
from typing import Annotated
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
import subprocess

app = FastAPI()

class AuraRequest(BaseModel):
    input_image: Annotated[UploadFile, File(description="A file read as UploadFile")]
    outfit_style: str
    gender_expression: str
    current_weather: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

def run_demo(filename: str):
    result = subprocess.run(["./run_demo.sh", filename], capture_output=True)
    output = result.stdout
    str_output = str(output)
    index = str_output.index("your color season is")
    color_season = str_output[index+len("your color season is "):-3]
    print("Success")
    print(color_season)
    return color_season
    
# Upload file endpoint
@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
):
    name = "../input-imgs/input.jpg"
    contents = file.file.read()
    im = Image.open(BytesIO(contents))
    im.save(name)
    color_season = run_demo(name)


    # When returning an image do the following commented return
    redbox_file = "../output-imgs/redbox.jpg"
    cropped_file = "../output-imgs/cropped.jpg"
    palette = "../output-imgs/your-palette.jpg"
     # return FileResponse(redbox_file)

    return color_season
