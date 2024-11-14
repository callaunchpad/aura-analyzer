from typing import Union
from typing import Annotated
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI()

class Item(BaseModel):
    input_image: UploadFile	
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/files/")
async def create_file(file: Annotated[bytes | None, File()] = None):
    
    if not file:
        return {"message": "No file sent"}
    else:
        return FileResponse("combined-demo/input-imgs/sza.jpg")

# Upload file endpoint
@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
):
    name = file.filename
    contents = file.file.read()
    im = Image.open(BytesIO(contents))
    im.save(name)

    return FileResponse(name)
    # return {"filename": file.filename, "content": contents}
