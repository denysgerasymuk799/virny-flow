from fastapi.responses import JSONResponse

from init_config import app, db_client


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    db_client.connect()


@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
    await db_client.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# For local development
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
