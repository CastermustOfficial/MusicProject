import React from 'react';
import type { TaskStatusResponse } from '../api';
import { Loader2, CheckCircle2, AlertCircle, Music } from 'lucide-react';

interface Props {
    task: TaskStatusResponse;
}

export const TaskMonitor: React.FC<Props> = ({ task }) => {
    const getStatusColor = () => {
        switch (task.status) {
            case 'completed': return 'text-green-500';
            case 'failed': return 'text-red-500';
            default: return 'text-primary';
        }
    };

    const getStatusIcon = () => {
        switch (task.status) {
            case 'completed': return <CheckCircle2 className="w-8 h-8 text-green-500" />;
            case 'failed': return <AlertCircle className="w-8 h-8 text-red-500" />;
            default: return <Loader2 className="w-8 h-8 text-primary animate-spin" />;
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto mt-8 bg-surface p-6 rounded-2xl border border-white/5">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4">
                    {getStatusIcon()}
                    <div className="text-left">
                        <h3 className="text-lg font-semibold text-white">Generation Status</h3>
                        <p className={`text-sm ${getStatusColor()} capitalize`}>{task.status}</p>
                    </div>
                </div>
                <span className="text-2xl font-bold text-white/20">
                    {Math.round(task.progress * 100)}%
                </span>
            </div>

            <div className="w-full h-2 bg-black/50 rounded-full overflow-hidden mb-6">
                <div
                    className="h-full bg-gradient-to-r from-primary via-secondary to-accent transition-all duration-500 ease-out"
                    style={{ width: `${task.progress * 100}%` }}
                />
            </div>

            {task.status === 'completed' && task.result_url && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                <Music className="w-5 h-5 text-primary" />
                            </div>
                            <div className="text-left">
                                <p className="font-medium text-white">Final Master.wav</p>
                                <p className="text-xs text-gray-400">Ready for download</p>
                            </div>
                        </div>
                        <audio controls className="h-8 w-64" src={task.result_url} />
                    </div>
                </div>
            )}

            {task.status === 'completed' && !task.result_url && task.message && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 mt-4">
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                <Music className="w-5 h-5 text-primary" />
                            </div>
                            <div className="text-left">
                                <p className="font-medium text-white">Generated Output</p>
                                <p className="text-xs text-gray-400">Text/Token Result</p>
                            </div>
                        </div>
                        <div className="p-3 bg-black/50 rounded-lg text-sm text-gray-300 font-mono whitespace-pre-wrap max-h-60 overflow-y-auto">
                            {task.message}
                        </div>
                    </div>
                </div>
            )}

            {task.error && (
                <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                    Error: {task.error}
                </div>
            )}
        </div>
    );
};
