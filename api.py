from fastapi import FastAPI


app = FastAPI()


@app.post("/")
async def generate_try_on():
    return {'hello': 'world'}
