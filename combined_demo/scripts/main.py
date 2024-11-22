from typing import Union
from typing import Annotated
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os

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
    cmd = "./run_demo.sh " + filename
    os.system(cmd)
    print("Success")
    
# Upload file endpoint
@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
):
    name = "../input-imgs/input.jpg"
    contents = file.file.read()
    im = Image.open(BytesIO(contents))
    im.save(name)
    run_demo(name)


    # output_filename = "combined_demo/output-imgs/" + file.file
    redbox_file = "../output-imgs/redbox.jpg"
    cropped_file = "../output-imgs/cropped.jpg"
    palette = "../output-imgs/your-palette.jpg"
    return FileResponse(redbox_file)
    # return {"redbox_image": FileResponse(redbox_file), "cropped_image": cropped_file, "color_analysis": FileResponse(palette)}
    # return {"filename": file.filename, "content": contents}
