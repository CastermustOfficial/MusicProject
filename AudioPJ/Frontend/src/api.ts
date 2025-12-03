import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export interface GenerationRequest {
    prompt: string;
    genre: string;
    lyrics?: string;
    reference_audio_path?: string;
    seed?: number;
}

export interface GenerationResponse {
    task_id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    message: string;
}

export interface TaskStatusResponse {
    task_id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress: number;
    result_url?: string;
    stems_url?: Record<string, string>;
    error?: string;
    message?: string;
}

export const generateMusic = async (request: GenerationRequest): Promise<GenerationResponse> => {
    const response = await axios.post(`${API_URL}/generate`, request);
    return response.data;
};

export const getTaskStatus = async (taskId: string): Promise<TaskStatusResponse> => {
    const response = await axios.get(`${API_URL}/status/${taskId}`);
    return response.data;
};
