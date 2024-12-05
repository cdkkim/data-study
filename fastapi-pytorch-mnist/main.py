from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from mnist import Net
import pdb


app = FastAPI()
model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt', weights_only=True))
model.eval()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class Item(BaseModel):
    # 28x28=784 integers
    image: list[int]


@app.post("/mnist")
def infer_mnist(request_body: Item):
    data = torch.tensor(request_body.image).view(1, 1, 28, 28)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    print(pred)
    return { 'data': pred.item() }

