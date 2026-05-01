from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.inference import predict_fish
from app.schemas import PredictResponse

app = FastAPI(
    title="Fish Freshness API",
    description="API untuk mendeteksi kesegaran ikan berbasis YOLOv8",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Send POST request with an image to /predict"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File must be an image!"})
    
    try:
        image_bytes = await file.read()
        result = predict_fish(image_bytes)
        
        if result.get("status") == "warning":
            return JSONResponse(status_code=400, content=result)
            
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})