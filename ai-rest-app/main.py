from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette import status
from starlette.middleware.cors import CORSMiddleware

import prediction_service as service
from predefined_learning_files import PredefinedLearningFile
from fastapi.security import APIKeyHeader

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key")
api_keys = [
    "4e9d8822-2eaf-46c4-b48c-4e9bdb3317c4"
]


def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )


@app.get("/health")
def health():
    return {"status": "UP"}


@app.post("/predict/{predefined_file}", response_class=JSONResponse)
def post_predict_existing_file(predefined_file: PredefinedLearningFile, split_percentage: float = 0.67, api_key: str = Security(get_api_key)):
    result = service.predict(predefined_file.path(), split_percentage)
    result_map_encoded = jsonable_encoder(result)
    return JSONResponse(content=result_map_encoded)


@app.post("/predict", response_class=JSONResponse)
async def post_predict(file: UploadFile = File(...), split_percentage: float = 0.67, api_key: str = Security(get_api_key)):
    result = service.predict(file.file, split_percentage)
    result_map_encoded = jsonable_encoder(result)
    return JSONResponse(content=result_map_encoded)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
