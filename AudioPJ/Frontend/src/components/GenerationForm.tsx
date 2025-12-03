import React, { useState } from 'react';
import type { GenerationRequest } from '../api';
import { Music, Mic, FileAudio, Sparkles } from 'lucide-react';

interface Props {
    onSubmit: (data: GenerationRequest) => void;
    isLoading: boolean;
}

export const GenerationForm: React.FC<Props> = ({ onSubmit, isLoading }) => {
    const [prompt, setPrompt] = useState('');
    const [genre, setGenre] = useState('');
    const [lyrics, setLyrics] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit({ prompt, genre, lyrics: lyrics || undefined });
    };

    return (
        <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto bg-surface p-8 rounded-2xl shadow-2xl border border-white/5">
            <div className="flex items-center gap-3 mb-8">
                <Sparkles className="w-6 h-6 text-primary" />
                <h2 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                    Create New Track
                </h2>
            </div>

            <div className="space-y-6">
                <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
                        <Music className="w-4 h-4" /> Musical Style / Prompt
                    </label>
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g. A cinematic pop song with emotional piano..."
                        className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                        required
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
                        <FileAudio className="w-4 h-4" /> Genre
                    </label>
                    <input
                        type="text"
                        value={genre}
                        onChange={(e) => setGenre(e.target.value)}
                        placeholder="e.g. Pop, Rock, Electronic"
                        className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-secondary focus:border-transparent outline-none transition-all"
                        required
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
                        <Mic className="w-4 h-4" /> Lyrics (Optional)
                    </label>
                    <textarea
                        value={lyrics}
                        onChange={(e) => setLyrics(e.target.value)}
                        placeholder="Enter lyrics here, or leave empty for instrumental..."
                        rows={6}
                        className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 focus:ring-2 focus:ring-accent focus:border-transparent outline-none transition-all resize-none"
                    />
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-gradient-to-r from-primary to-secondary hover:from-primary/80 hover:to-secondary/80 text-white font-bold py-4 rounded-xl transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                    {isLoading ? (
                        <span className="animate-pulse">Generating...</span>
                    ) : (
                        <>
                            <Sparkles className="w-5 h-5" /> Generate Masterpiece
                        </>
                    )}
                </button>
            </div>
        </form>
    );
};
