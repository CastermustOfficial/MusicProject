import { useState, useEffect } from 'react';
import { GenerationForm } from './components/GenerationForm';
import { TaskMonitor } from './components/TaskMonitor';
import { generateMusic, getTaskStatus } from './api';
import type { GenerationRequest, TaskStatusResponse } from './api';
import { Disc3 } from 'lucide-react';

function App() {
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async (data: GenerationRequest) => {
    try {
      setIsLoading(true);
      const response = await generateMusic(data);
      setCurrentTaskId(response.task_id);
    } catch (error) {
      console.error('Failed to start generation:', error);
      alert('Failed to start generation. Make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!currentTaskId) return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await getTaskStatus(currentTaskId);
        setTaskStatus(status);

        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Failed to poll status:', error);
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [currentTaskId]);

  return (
    <div className="min-h-screen bg-background text-white p-8">
      <header className="mb-12">
        <div className="flex items-center justify-center gap-4 mb-4">
          <div className="w-16 h-16 bg-gradient-to-br from-primary to-secondary rounded-2xl flex items-center justify-center shadow-lg shadow-primary/20">
            <Disc3 className="w-10 h-10 text-white animate-spin-slow" />
          </div>
          <h1 className="text-5xl font-black tracking-tighter bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
            AUDIO INSANE
          </h1>
        </div>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Extreme High-Fidelity Neural Music Synthesis Infrastructure
        </p>
      </header>

      <main className="max-w-4xl mx-auto space-y-12">
        <GenerationForm onSubmit={handleGenerate} isLoading={isLoading} />

        {taskStatus && (
          <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
            <TaskMonitor task={taskStatus} />
          </div>
        )}
      </main>

      <footer className="mt-24 text-center text-gray-600 text-sm">
        <p>Powered by YuE-s1-7B • UVR5 • RVC • Matchering</p>
        <p className="mt-2">On-Premise Infrastructure v1.0</p>
      </footer>
    </div>
  );
}

export default App;
