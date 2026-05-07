from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from app.inference import process_inference
from app.schemas import PredictionResponse

app = FastAPI(
    title="Fish Freshness API",
    description="API untuk mendeteksi kesegaran ikan menggunakan YOLOv8 dan EfficientNet",
)

@app.get("/")
def read_root():
    return {"message": "Silakan masukkan gambar untuk memulai prediksi di endpoint /predict"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_freshness(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (.png, .jpg, .jpeg, .webp)")
    
    try:
        image_bytes = await file.read()
        
        result_dict = process_inference(image_bytes, file.filename)
        
        return result_dict
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)