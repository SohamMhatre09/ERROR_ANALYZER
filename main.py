from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from error_analysis_system import ErrorAnalysisSystem
import os

app = FastAPI()

# Initialize the ErrorAnalysisSystem
model_path = 'error_classifier_model.joblib'
api_key = os.environ.get('GOOGLE_API_KEY')
system = ErrorAnalysisSystem(model_path, api_key)

class ErrorRequest(BaseModel):
    error_message: str

@app.post("/analyze_error")
async def analyze_error(request: ErrorRequest):
    try:
        result = system.process_error(request.error_message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)