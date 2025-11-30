# app/main.py
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from app.api.extract import process_url_document
import uvicorn

app = FastAPI(
    title="BFHL Datathon Bill Extractor",
    description="OCR + Table Detection + LLM Mapping Pipeline",
    version="1.0.0",
)

@app.get("/extract")
async def extract_get(url: str = Query(..., description="Document URL (PNG/JPG/PDF)")):
    """
    Usage:
    GET /extract?url=https://example.com/sample1.png
    """
    try:
        result = process_url_document(url)
        return JSONResponse({"is_success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[Error] {str(e)}")


@app.post("/extract")
async def extract_post(url: str = Query(..., description="Document URL (PNG/JPG/PDF)")):
    """
    Usage:
    POST /extract?url=https://example.com/sample1.png
    No JSON body required.
    """
    try:
        result = process_url_document(url)
        return JSONResponse({"is_success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[Error] {str(e)}")


@app.get("/")
def health():
    return {"status": "ok", "message": "BFHL Extractor running"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
