from typing import Annotated, Literal
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import func
import subprocess

class AuraRequest(BaseModel):
    outfit_style: Literal['Casual', 'Ethnic', 'Formal', 'Home', 'Party', 'Smart Casual', 'Sports', 'Travel'] | None
    gender_expression: Literal['Boys', 'Girls', 'Men', 'Unisex', 'Women']
    colorSeason: Literal['autumn', 'winter', 'spring', 'summer']
    num_outfits: int = 1


class Fashion(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    gender: str = Field(index=True)
    masterCategory: str = Field(index=True)
    subCategory: str = Field(index=True)
    articleType: str = Field(index=True)
    baseColour: str
    season: str
    year: str
    usage: str
    productDisplayName: str
    colorSeason: str = Field(default=None, index=True)

# where images are located
data_path = "../../color_analysis/fashion-dataset-small/images"

# start database
sqlite_file_name = "../../color_analysis/small-fashion-dataset.db"
# sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

def run_demo(filename: str):
    result = subprocess.run(["./run_demo.sh", filename], capture_output=True)
    output = result.stdout
    str_output = str(output)
    index = str_output.index("your color season is")
    color_season = str_output[index+len("your color season is "):-3]
    print("Success")
    print(color_season)
    return color_season

# Get color season
@app.post("/aura_analyze/")
async def aura_analyze(
    file: Annotated[UploadFile, File()]
):  
    if not os.path.exists("../input-imgs"):
        os.makedirs("../input-imgs")
    if not os.path.exists("../intermediate-imgs"):
        os.makedirs("../intermediate-imgs")
    if not os.path.exists("../output-imgs"):
        os.makedirs("../output-imgs")

    name = "../input-imgs/input.jpg"
    contents = file.file.read()
    im = Image.open(BytesIO(contents)).convert("RGB")
    im.save(name)
    color_season = run_demo(name)

    return color_season

@app.get("/aura_analyze/redbox")
async def get_redbox():
    redbox_path = "../output-imgs/redbox.jpg"
    if not os.path.exists(redbox_path):
        raise HTTPException(status_code=404, detail=f"Redbox not found")

    return FileResponse(redbox_path, media_type="image/jpg")

@app.get("/aura_analyze/cropped")
async def get_cropped():
    cropped_path = "../output-imgs/cropped.jpg"
    if not os.path.exists(cropped_path):
        raise HTTPException(status_code=404, detail=f"Cropped image not found")
    
    return FileResponse(cropped_path, media_type="image/jpg")

@app.get("/aura_analyze/palette")
async def get_palette():
    palette_path = "../output-imgs/your-palette.jpg"
    if not os.path.exists(palette_path):
        raise HTTPException(status_code=404, detail=f"Palette not found")
    
    return FileResponse(palette_path, media_type="image/jpg")

# Generate outfits
@app.post("/generate-outfit")
async def generate_outfit(
    session: SessionDep,
    aura_request: AuraRequest
):
    tops = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.usage == aura_request.outfit_style).where(Fashion.masterCategory == "Apparel").where(Fashion.subCategory == "Topwear").order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not tops:
        tops = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.usage == "Casual").where(Fashion.masterCategory == "Apparel").where(Fashion.subCategory == "Topwear").order_by(func.random()).limit(aura_request.num_outfits)).all()
        if not tops:
            raise HTTPException(status_code=404, detail="No matching tops found")
    
    bottoms = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.usage == aura_request.outfit_style).where(Fashion.masterCategory == "Apparel").where(Fashion.subCategory == "Bottomwear").order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not bottoms:
        bottoms = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.masterCategory == "Apparel").where(Fashion.subCategory == "Bottomwear").order_by(func.random()).limit(aura_request.num_outfits)).all()
        if not bottoms:
            raise HTTPException(status_code=404, detail="No matching bottoms found")
    
    accessories = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.masterCategory == "Accessories").order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not accessories:
        accessories = session.exec(select(Fashion).where(Fashion.colorSeason == aura_request.colorSeason).where(Fashion.gender == aura_request.gender_expression).where(Fashion.masterCategory == "Accessories").order_by(func.random()).limit(aura_request.num_outfits)).all()
        if not accessories:
            raise HTTPException(status_code=404, detail="No matching accessories found")

    return [tops, bottoms, accessories]

# return image of item that matches id
@app.get("/items/{image_id}")
async def get_item_image(
    session: SessionDep,
    image_id: int
):
    if session.exec(select(Fashion).where(Fashion.id==image_id)).first() is None:
        raise HTTPException(status_code=404, detail=f"Item {image_id} does not exist")
    if not os.path.exists(f"{data_path}/{image_id}.jpg"):
        raise HTTPException(status_code=404, detail=f"Image for item {image_id} not found")
    
    return FileResponse(f"{data_path}/{image_id}.jpg", media_type="image/jpg")
