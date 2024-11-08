from fastapi import FastAPI
from pydantic import BaseModel


class AuraRequest(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/items/")
async def create_item(item: AuraRequest):
    return item