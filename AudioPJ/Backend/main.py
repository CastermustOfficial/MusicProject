from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import sys
import os
import logging

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Aggiungi la directory corrente al path per permettere l'import di yue_client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import PIPELINE_MODE

# Import the appropriate pipeline based on configuration
if PIPELINE_MODE == "huggingface":
    from yue_hf_client import run_pipeline_hq as run_pipeline
    logger.info("Using HuggingFace pipeline (high quality, slower)")
else:
    from yue_client import run_pipeline
    logger.info("Using GGUF pipeline (fast, lower quality)")

app = FastAPI()

# Create outputs directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    logger.info("Backend Server Started! Logging is working.")
    logger.info(f"Outputs directory: {OUTPUT_DIR}")
    print("Backend Server Started! Logging is working.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the outputs directory to serve audio files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class GenRequest(BaseModel):
    genre: str
    prompt: str
    lyrics: str = ""

jobs = {}

def task_wrapper(job_id, req):
    logger.info(f"Starting job {job_id} with prompt: {req.prompt[:50]}...")
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0.1
        # Chiama la funzione pesante
        # Mappiamo: req.prompt -> mood, req.lyrics -> prompt_text
        jobs[job_id]['progress'] = 0.3
        result_path = run_pipeline(req.lyrics, req.genre, req.prompt)
        if result_path:
            logger.info(f"Job {job_id} completed successfully. Result: {result_path}")
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 1.0
            jobs[job_id]['result_url'] = f"/outputs/{result_path}" if result_path else None
            jobs[job_id]['message'] = f"Successfully generated: {result_path}"
        else:
            logger.error(f"Job {job_id} failed: Pipeline returned None")
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['progress'] = 0.0
            jobs[job_id]['error'] = 'Pipeline returned None'
    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {e}", exc_info=True)
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['progress'] = 0.0
        jobs[job_id]['error'] = str(e)

@app.post("/api/generate")
async def generate(req: GenRequest, bg_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'task_id': job_id
    }
    bg_tasks.add_task(task_wrapper, job_id, req)
    logger.info(f"Created job {job_id}")
    return {
        "task_id": job_id,
        "status": "queued",
        "message": "Task created successfully"
    }

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id, None)
    if not job:
        return {
            'task_id': job_id,
            'status': 'not_found',
            'progress': 0.0,
            'error': 'Task not found'
        }

    # Ensure all required fields are present
    response = {
        'task_id': job_id,
        'status': job.get('status', 'unknown'),
        'progress': job.get('progress', 0.0),
    }

    if 'result_url' in job:
        response['result_url'] = job['result_url']
    if 'message' in job:
        response['message'] = job['message']
    if 'error' in job:
        response['error'] = job['error']

    logger.info(f"Status check for {job_id}: {response['status']} ({response['progress']*100}%)")
    return response
