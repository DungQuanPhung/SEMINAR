from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pipeline_ABSA import load_all_models, run_full_pipeline

app = FastAPI(title="ABSA API")

models = load_all_models()

@app.get("/")
def root():
    return {"message": "✅ ABSA backend running on Railway!"}

@app.post("/api/analyze")
async def analyze(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)
    try:
        df = run_full_pipeline(text, models)
        return JSONResponse(df.to_dict(orient="records"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
