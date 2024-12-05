from typing import Annotated, Literal
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
from sqlmodel import Field, Session, SQLModel, or_, create_engine, select
from sqlalchemy import func
import subprocess

class AuraRequest(BaseModel):
    Department: Literal['menswear', 'womenswear']
    ColorSeason: Literal['autumn', 'winter', 'spring', 'summer']
    # MasterCategory: Literal['Tops', 'Bottoms', 'Outerwear', 'Footwear', 'Tailoring', 'Accessories']
    num_outfits: int = 1


class Items(SQLModel, table=True):
    Id: int = Field(default=None, primary_key=True)
    Department: str = Field(index=True)
    MasterCategory: str = Field(index=True)
    SubCategory: str = Field(index=True)
    Size: str
    Color: str
    # Designers:list[str]
    # Hashtags: list[str]
    ProductDisplayName: str
    ItemUrl: str
    ColorSeason: str = Field(default=None, index=True)

    class Config:
        arbitrary_types_allowed=True

# where images are located
# data_path = "color_analysis/fashion-dataset-small/images"
data_path = "color_analysis/grailed-dataset/images"

# start database
# sqlite_file_name = "color_analysis/small-fashion-dataset.db"
sqlite_file_name = "color_analysis/grailed.db"

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

def run_demo(filename: str):
    script_path = os.path.join(os.path.dirname(__file__),'run_demo.sh')
    result = subprocess.run([script_path, filename], capture_output=True)
    output = result.stdout
    str_output = str(output)
    index = str_output.index("your color season is")
    color_season = str_output[index+len("your color season is "):-3]
    return color_season

# Get color season
@app.post("/aura_analyze/")
async def aura_analyze(
    file: Annotated[UploadFile, File()]
):  
    if not os.path.exists("combined_demo/input-imgs"):
        os.makedirs("combined_demo/input-imgs")
    if not os.path.exists("combined_demo/intermediate-imgs"):
        os.makedirs("combined_demo/intermediate-imgs")
    if not os.path.exists("combined_demo/output-imgs"):
        os.makedirs("combined_demo/output-imgs")

    name = "input.jpg"
    contents = file.file.read()
    im = Image.open(BytesIO(contents)).convert("RGB")
    im.save(f"combined_demo/input-imgs/{name}")
    color_season = run_demo(name)

    return color_season

@app.get("/aura_analyze/redbox")
async def get_redbox():
    redbox_path = "combined_demo/output-imgs/redbox.jpg"
    if not os.path.exists(redbox_path):
        raise HTTPException(status_code=404, detail=f"Redbox not found")

    return FileResponse(redbox_path, media_type="image/jpg")

@app.get("/aura_analyze/cropped")
async def get_cropped():
    cropped_path = "combined_demo/output-imgs/cropped.jpg"
    if not os.path.exists(cropped_path):
        raise HTTPException(status_code=404, detail=f"Cropped image not found")
    
    return FileResponse(cropped_path, media_type="image/jpg")

@app.get("/aura_analyze/palette")
async def get_palette():
    palette_path = "combined_demo/output-imgs/your-palette.jpg"
    if not os.path.exists(palette_path):
        raise HTTPException(status_code=404, detail=f"Palette not found")
    
    return FileResponse(palette_path, media_type="image/jpg")

# Generate outfits
@app.post("/generate_outfit")
async def generate_outfit(
    session: SessionDep,
    aura_request: AuraRequest
):
    # Get tops
    tops = session.exec(select(Items).where(Items.ColorSeason == aura_request.ColorSeason).where(Items.Department == aura_request.Department).where(or_(Items.MasterCategory == "tops", Items.MasterCategory == "womens_tops")).order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not tops:
        raise HTTPException(status_code=404, detail="No matching tops found")
    
    # Get bottoms
    bottoms = session.exec(select(Items).where(Items.ColorSeason == aura_request.ColorSeason).where(Items.Department == aura_request.Department).where(or_(Items.MasterCategory == "bottoms", Items.MasterCategory == "womens_bottoms")).order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not bottoms:
        raise HTTPException(status_code=404, detail="No matching bottoms found")
    
    # Get shoes
    shoes = session.exec(select(Items).where(Items.ColorSeason == aura_request.ColorSeason).where(Items.Department == aura_request.Department).where(or_(Items.MasterCategory == "footwear", Items.MasterCategory == "womens_footwear")).order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not bottoms:
        raise HTTPException(status_code=404, detail="No matching shoes found")
    
    # Get outerwear
    outerwear = session.exec(select(Items).where(Items.ColorSeason == aura_request.ColorSeason).where(Items.Department == aura_request.Department).where(or_(Items.MasterCategory == "outerwear", Items.MasterCategory == "outerwear")).order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not outerwear:
        print("No matching outerwear found")
        # raise HTTPException(status_code=404, detail="No matching outerwear found")
    
    # Get accessories
    accessories = session.exec(select(Items).where(Items.ColorSeason == aura_request.ColorSeason).where(Items.Department == aura_request.Department).where(or_(Items.MasterCategory == "accessories", Items.MasterCategory == "womens_accessories")).order_by(func.random()).limit(aura_request.num_outfits)).all()
    if not bottoms:
        print("No matching accessories found")
        # raise HTTPException(status_code=404, detail="No matching outerwear found")

    return [tops, bottoms, shoes, outerwear, accessories]

# return image of item that matches id
@app.get("/items/{image_id}")
async def get_item_image(
    session: SessionDep,
    image_id: int
):
    if session.exec(select(Items).where(Items.Id==image_id)).first() is None:
        raise HTTPException(status_code=404, detail=f"Item {image_id} does not exist")
    if not os.path.exists(f"{data_path}/{image_id}.jpg"):
        raise HTTPException(status_code=404, detail=f"Image for item {image_id} not found")
    
    return FileResponse(f"{data_path}/{image_id}.jpg", media_type="image/jpg")
