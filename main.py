from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from error_analysis_system import ErrorAnalysisSystem
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the ErrorAnalysisSystem
model_path = os.path.join(os.getcwd(), 'error_classifier_model.joblib')
api_key = os.environ.get('GOOGLE_API_KEY')

if not api_key:
    logger.error("GOOGLE_API_KEY environment variable is not set.")
    raise RuntimeError("Missing required environment variable: GOOGLE_API_KEY")

try:
    system = ErrorAnalysisSystem(model_path, api_key)
    logger.info("ErrorAnalysisSystem initialized successfully.")
except FileNotFoundError:
    logger.error(f"Model file not found at path: {model_path}")
    raise RuntimeError(f"Model file not found at path: {model_path}")
except Exception as e:
    logger.error(f"Error initializing ErrorAnalysisSystem: {str(e)}")
    raise RuntimeError("Failed to initialize the ErrorAnalysisSystem")

class ErrorRequest(BaseModel):
    error_message: str

@app.post("/analyze_error")
async def analyze_error(request: ErrorRequest):
    try:
        result = system.process_error(request.error_message)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
