from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from twin_utils import predict_twin_similarity
from model_loader import load_feature_extractor, load_siamese_model

app = FastAPI()

feature_extractor = load_feature_extractor()
siamese_model = load_siamese_model()

@app.get("/")
def read_root():
    return {"message": "Twin Face Similarity API is running 🚀"}

@app.post("/compare")
async def compare_twins(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    img1_path = f"temp/{img1.filename}"
    img2_path = f"temp/{img2.filename}"

    with open(img1_path, "wb") as f1, open(img2_path, "wb") as f2:
        shutil.copyfileobj(img1.file, f1)
        shutil.copyfileobj(img2.file, f2)

    score = predict_twin_similarity(img1_path, img2_path, feature_extractor, siamese_model)
    
    result = "Same Twin" if score > 0.5 else "Different Twins"

    os.remove(img1_path)
    os.remove(img2_path)

    return JSONResponse(content={"similarity_score": float(score), "result": result})