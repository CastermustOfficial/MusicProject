from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class GenerationStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerationRequest(BaseModel):
    prompt: str
    genre: str
    lyrics: Optional[str] = None
    reference_audio_path: Optional[str] = None
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    task_id: str
    status: GenerationStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: GenerationStatus
    progress: float
    result_url: Optional[str] = None
    stems_url: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    message: Optional[str] = None
