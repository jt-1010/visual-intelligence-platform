'use client';

import { useState } from 'react';

export function TrafficSession() {
  const [promptInput, setPromptInput] = useState('');
  const [terminalLines, setTerminalLines] = useState([
    " > SYSTEM: Initiating Vision-to-GenAI Traffic Controller...",
    " > SYSTEM: Connecting to local backend (qwen2.5:7b-instruct)... OK",
    " > AGENT: Model loaded. Observing intersection feed...",
    " > AGENT: [OBSERVATION] Northbound queue reaching critical length (12 vehicles).",
    " > AGENT: [ANALYSIS] Fixed-timing cycle will cause NB spillback in 18s.",
    " > AGENT: [RECOMMENDATION] Extend NB/SB green phase by +15s."
  ]);

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptInput.trim()) return;
    setTerminalLines(prev => [...prev, ` > OPERATOR: ${promptInput}`, " > AGENT: Processing instruction..."]);
    setPromptInput('');
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2 h-[calc(100vh-6rem)] font-mono">
      
      {/* LEFT HALF: Screen & Detection */}
      <div className="flex flex-col gap-6 min-h-0">
        
        {/* Video Screen */}
        <div className="relative overflow-hidden border border-slate-700 bg-black aspect-video flex-shrink-0">
          <iframe 
            src="https://www.youtube.com/embed/1EiC9bvVGnk?autoplay=1&mute=1&controls=0&showinfo=0&rel=0&loop=1&playlist=1EiC9bvVGnk"
            className="absolute inset-0 h-full w-full object-cover opacity-60 pointer-events-none grayscale contrast-125"
            allow="autoplay; encrypted-media"
          />
          <div className="absolute top-4 left-4 z-20 flex gap-3 text-xs">
            <span className="bg-black/80 px-2 py-1 text-red-500 border border-red-500/30 flex items-center gap-2">
              <span className="h-2 w-2 bg-red-500 animate-pulse rounded-full"></span> LIVE
            </span>
            <span className="bg-black/80 px-2 py-1 text-[#00ffcc] border border-[#00ffcc]/30">
              TRACKER: YOLO_PIPELINE (PENDING T1)
            </span>
          </div>
        </div>

        {/* Detection Metrics */}
        <div className="border border-slate-700 bg-black p-5 flex-1 min-h-0 overflow-y-auto">
          <p className="text-xs text-slate-500 mb-4 border-b border-slate-800 pb-2">TELEMETRY_DATA</p>
          <div className="grid grid-cols-2 gap-px bg-slate-800 border border-slate-800">
            <div className="bg-black p-4">
              <h3 className="text-xs text-slate-400">NB_QUEUE</h3>
              <p className="mt-1 text-2xl text-red-400">12</p>
            </div>
            <div className="bg-black p-4">
              <h3 className="text-xs text-slate-400">SB_QUEUE</h3>
              <p className="mt-1 text-2xl text-[#00ffcc]">4</p>
            </div>
            <div className="bg-black p-4">
              <h3 className="text-xs text-slate-400">EB_QUEUE</h3>
              <p className="mt-1 text-2xl text-[#ffcc00]">8</p>
            </div>
            <div className="bg-black p-4">
              <h3 className="text-xs text-slate-400">WB_QUEUE</h3>
              <p className="mt-1 text-2xl text-[#00ffcc]">2</p>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT HALF: Terminal & Prompt */}
      <div className="flex flex-col gap-6 min-h-0">
        
        {/* Top Row: Terminal */}
        <div className="flex-1 border border-slate-700 bg-black flex flex-col min-h-0">
          <div className="border-b border-slate-800 bg-slate-950 px-4 py-2 flex justify-between items-center text-xs text-slate-500">
            <span>LLM_TERMINAL</span>
            <span className="text-[#00ffcc] flex items-center gap-2">
              <span className="h-1.5 w-1.5 bg-[#00ffcc] rounded-full"></span>
              IDLE
            </span>
          </div>
          <div className="flex-1 p-4 overflow-y-auto text-sm leading-relaxed space-y-2">
            {terminalLines.map((line, i) => (
              <div key={i} className={line.includes('OPERATOR') ? 'text-[#ffcc00]' : (line.includes('RECOMMENDATION') ? 'text-red-400' : 'text-[#00ffcc]')}>
                {line}
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Row: Understood & Prompt */}
        <div className="border border-slate-700 bg-black flex flex-col flex-shrink-0">
          <div className="border-b border-slate-800 p-4">
            <div className="flex items-start gap-4">
              <div className="text-[#00ffcc] text-2xl">✓</div>
              <div>
                <h3 className="text-[#00ffcc] text-sm mb-1">UNDERSTOOD</h3>
                <p className="text-slate-400 text-xs">The agent has identified the congestion and is awaiting operator instructions to adjust the signal architecture sequence.</p>
              </div>
            </div>
          </div>
          <form onSubmit={handlePromptSubmit} className="p-4 bg-slate-950 flex gap-3">
            <span className="text-[#00ffcc] mt-2.5">{'>'}</span>
            <input 
              type="text" 
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              placeholder="Type your prompt to the agent..."
              className="flex-1 bg-transparent border-none text-[#00ffcc] placeholder:text-slate-700 focus:outline-none focus:ring-0 py-2"
              autoComplete="off"
            />
            <button 
              type="submit"
              className="px-4 py-2 bg-slate-800 text-slate-300 text-xs hover:bg-slate-700 hover:text-white transition-colors border border-slate-700"
            >
              EXECUTE
            </button>
          </form>
        </div>
        
      </div>
    </div>
  );
}

