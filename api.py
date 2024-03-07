from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Response

from infer import VTONInference


app = FastAPI(docs_url=None, redoc_url=None)
model = VTONInference()


@app.post("/")
async def generate_try_on(person: Annotated[bytes, File()], cloth: Annotated[bytes, File()]):
    person_img = bytes_to_img(person)
    cloth_img = bytes_to_img(cloth)
    img = model(person_img, cloth_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return Response(cv2.imencode('.jpg', img)[1].tobytes(),
                    media_type='image/jpeg')



def bytes_to_img(x: bytes) -> np.ndarray:
    x = np.frombuffer(x, dtype = np.uint8)
    x = cv2.imdecode(x, cv2.IMREAD_COLOR)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
