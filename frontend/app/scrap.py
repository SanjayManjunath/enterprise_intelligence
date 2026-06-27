"use client";

import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, User, Bot, Paperclip, Terminal, Database, Sparkles, 
  FileText, Loader2, CheckCircle2, AlertCircle, Copy, BarChart3, Maximize2, Trash2, Settings, ShieldCheck, ChevronRight, Filter
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; 
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import dynamic from 'next/dynamic';

/**
 * DYNAMIC PLOTLY ENGINE: 
 * Prevents SSR Hydration errors by loading strictly on the client.
 */
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false, 
  loading: () => (
    <div className="h-72 flex flex-col items-center justify-center bg-slate-50 rounded-[2.5rem] border border-slate-100 animate-pulse">
      <Loader2 className="animate-spin text-blue-500 mb-3" size={32} />
      <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Initializing Strategic Viz Engine</p>
    </div>
  ) 
});

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function AIClientInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: 'assistant', 
      content: "Strategic Intent Layer Online. Ready for your audit, Sanjay." 
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [mounted, setMounted] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- HYDRATION & SESSION PERSISTENCE ---
  useEffect(() => {
    setMounted(true);
    const existingSession = localStorage.getItem('intelligence_session_id');
    const newSession = existingSession || `session_${Math.floor(Math.random() * 100000)}`;
    localStorage.setItem('intelligence_session_id', newSession);
    setSessionId(newSession);
  }, []);

  useEffect(() => { 
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); 
  }, [messages]);

  // --- FILE UPLOAD HANDLER ---
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !sessionId) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`http://localhost:8000/upload?session_id=${sessionId}`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `✅ **Context Bridge Active:** \`${file.name}\` successfully ingested.` 
        }]);
      } else {
        throw new Error("Upload failed");
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "❌ **Worker Error:** Data worker connection failed." 
      }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = ''; 
    }
  };

  // --- MESSAGE HANDLER ---
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          history: messages.map(m => `${m.role}: ${m.content}`),
          session_id: sessionId
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
      } else {
        throw new Error(data.message);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "⚠️ **Engine Stalled:** Potential context overflow or backend timeout." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([{ role: 'assistant', content: "Context cleared. System re-initialized for fresh audit." }]);
  };

  if (!mounted) return <div className="h-screen bg-slate-900" />;

  return (
    <div className="flex h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-100">
      
      {/* --- SIDEBAR: STRATEGIC ENGINES --- */}
      <aside className="w-80 bg-slate-900 text-white flex flex-col p-8 shadow-2xl z-20">
        <div className="flex items-center gap-4 mb-16 px-2">
          <div className="bg-blue-600 p-3 rounded-2xl shadow-2xl shadow-blue-600/30">
            <Sparkles size={26} className="text-white" />
          </div>
          <div>
            <h1 className="font-black text-xl tracking-tighter uppercase italic tracking-widest">Gemini</h1>
            <p className="text-[9px] text-slate-500 font-black tracking-[0.4em] uppercase">Enterprise RAG</p>
          </div>
        </div>
        
        <nav className="flex-1 space-y-4">
          <div className="text-[10px] text-slate-600 font-black uppercase tracking-[0.3em] mb-6 px-2">Knowledge Domains</div>
          
          <button className="w-full flex items-center justify-between px-5 py-4 text-sm bg-slate-800/80 border border-slate-700/50 rounded-[1.5rem] hover:bg-slate-800 transition-all duration-300 group">
            <div className="flex items-center gap-4">
              <Database size={20} className="text-blue-400" /> 
              <span className="font-bold">SQL Database</span>
            </div>
            <div className="w-2.5 h-2.5 rounded-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)] animate-pulse"></div>
          </button>

          <button className="w-full flex items-center justify-between px-5 py-4 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-[1.5rem] transition-all">
            <div className="flex items-center gap-4">
              <Terminal size={20} /> <span>Vector Cluster</span>
            </div>
            <ChevronRight size={14} className="opacity-20" />
          </button>

          <button className="w-full flex items-center justify-between px-5 py-4 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-[1.5rem] transition-all">
            <div className="flex items-center gap-4">
              <FileText size={20} /> <span>Project Specs</span>
            </div>
            <ChevronRight size={14} className="opacity-20" />
          </button>
        </nav>

        <div className="mt-auto space-y-6">
          <button onClick={clearChat} className="w-full flex items-center gap-3 px-6 py-4 text-xs font-black text-slate-500 hover:text-red-400 transition-colors uppercase tracking-widest">
            <Trash2 size={16} /> Reset Engine
          </button>
          
          <div className="flex items-center gap-4 px-5 py-5 bg-slate-800/50 rounded-[2rem] border border-slate-800 shadow-2xl">
             <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center text-sm font-black">SM</div>
             <div className="flex-1 overflow-hidden">
                <p className="text-xs font-black text-white truncate">Sanjay Manjunath</p>
                <p className="text-[9px] text-slate-500 font-mono tracking-tighter truncate opacity-80 uppercase">
                  NODE: {sessionId.slice(-6).toUpperCase()}
                </p>
             </div>
             <Settings size={16} className="text-slate-600 hover:text-white cursor-pointer transition-colors" />
          </div>
        </div>
      </aside>

      {/* --- MAIN INTERFACE --- */}
      <main className="flex-1 flex flex-col relative bg-white overflow-hidden shadow-[inset_20px_0_40px_-20px_rgba(0,0,0,0.05)]">
        
        <header className="h-24 border-b border-slate-100 flex items-center px-12 justify-between bg-white/80 backdrop-blur-3xl sticky top-0 z-10">
           <div className="flex items-center gap-4 text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">
              <ShieldCheck size={18} className="text-green-500" />
              Engine Status: Strategic Intelligence Active
           </div>
           
           <div className="flex items-center gap-6">
             {isUploading && (
                <div className="flex items-center gap-3 text-blue-600 text-[10px] font-black bg-blue-50/50 px-5 py-2.5 rounded-full border border-blue-100 uppercase tracking-widest">
                  <Loader2 size={16} className="animate-spin" /> Ingesting Relational Data...
                </div>
             )}
             <div className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] bg-slate-50 px-4 py-2.5 rounded-full border border-slate-100">
                Hybrid Architecture 2.0
             </div>
           </div>
        </header>

        <div className="flex-1 overflow-y-auto p-12 md:p-24 space-y-20 scroll-smooth bg-gradient-to-b from-white via-white to-slate-50/30">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-10 max-w-6xl mx-auto ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-4 duration-500`}>
              {msg.role === 'assistant' && (
                <div className="w-14 h-14 rounded-[1.5rem] bg-white border border-slate-200 shadow-xl flex items-center justify-center text-blue-600 flex-shrink-0 mt-1 transform hover:rotate-6 transition-transform">
                  <Bot size={30} />
                </div>
              )}
              
              <div className={`group relative max-w-[85%] px-10 py-8 rounded-[3rem] ${
                msg.role === 'user' 
                ? 'bg-slate-900 text-white rounded-tr-none shadow-[0_30px_60px_-15px_rgba(15,23,42,0.3)]' 
                : 'bg-white border border-slate-100 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.03)] rounded-tl-none ring-1 ring-slate-200/50'
              }`}>
                {msg.role === 'user' && <div className="text-[10px] font-black opacity-30 uppercase mb-2 tracking-widest">User Request</div>}
                
                {/* ISSUE #1 FIX: prose-invert ensures white text visibility for user queries */}
                <div className={`prose prose-slate max-w-none prose-sm md:prose-base leading-[1.8] font-medium break-words ${msg.role === 'user' ? 'text-white prose-invert' : 'text-slate-700'}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table({children}) {
                        return (
                          <div className="my-10 overflow-x-auto rounded-[2rem] border border-slate-200 shadow-2xl shadow-slate-200/20">
                            <table className="w-full text-sm text-left border-collapse bg-white">{children}</table>
                          </div>
                        );
                      },
                      thead({children}) { return <thead className="bg-slate-50 border-b border-slate-200 text-slate-700 uppercase text-[10px] font-black tracking-[0.2em]">{children}</thead>; },
                      th({children}) { return <th className="px-8 py-5 font-black">{children}</th>; },
                      td({children}) { return <td className="px-8 py-5 border-b border-slate-50 text-slate-600 font-medium">{children}</td>; },
                      
                      code({node, inline, className, children, ...props}: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        const codeContent = String(children).replace(/\n$/, '').trim();
                        
                        /**
                         * HARDENED PLOTLY BRIDGE:
                         * Detects homogeneity and enforces categorical axes.
                         */
                        if (!inline && (match?.[1] === 'json_plotly' || codeContent.startsWith('{"data":'))) {
                          try {
                            const plotData = JSON.parse(codeContent);
                            const mainTrace = plotData.data[0];
                            
                            // LOGIC: Scaling and Homogeneity Correction
                            const allSameY = mainTrace.y?.every((val: any) => val === mainTrace.y[0]);
                            const isCategorical = mainTrace.x?.some((val: any) => typeof val === 'string');

                            const layout = {
                              ...plotData.layout,
                              autosize: true,
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0)',
                              margin: { t: 40, r: 20, b: 80, l: 60 },
                              xaxis: { 
                                ...plotData.layout.xaxis, 
                                type: isCategorical ? 'category' : 'auto' // FIX: Categorical mapping
                              },
                              yaxis: { 
                                ...plotData.layout.yaxis, 
                                // FIX: Force range for homogeneous data
                                range: allSameY ? [0, mainTrace.y[0] * 1.5 || 5] : undefined 
                              },
                              font: { family: 'Inter, sans-serif', size: 12, color: '#64748b' }
                            };

                            return (
                              <div className="my-12 border-2 border-slate-100 rounded-[2.5rem] overflow-hidden bg-white shadow-2xl transition-all hover:scale-[1.01]">
                                 <div className="bg-slate-50/50 border-b border-slate-100 px-10 py-5 flex justify-between items-center">
                                   <div className="flex items-center gap-4">
                                     <div className="p-2 bg-blue-100 rounded-xl text-blue-600"><BarChart3 size={20}/></div>
                                     <span className="text-[11px] font-black text-slate-500 uppercase tracking-[0.25em]">Evidence Visualization Engine</span>
                                   </div>
                                   <div className="flex gap-4">
                                     <span className="text-[9px] bg-green-100 text-green-700 px-3 py-1 rounded-full font-black uppercase">Live Plotly</span>
                                     <button className="p-2.5 text-slate-400 hover:text-slate-900 hover:bg-white rounded-xl transition-all shadow-sm"><Maximize2 size={18}/></button>
                                   </div>
                                 </div>
                                 <div className="p-10 bg-white min-h-[450px]">
                                    <Plot 
                                      data={plotData.data} 
                                      layout={layout} 
                                      useResizeHandler={true} 
                                      style={{width: "100%", height: "100%"}} 
                                      config={{ responsive: true, displayModeBar: false }}
                                    />
                                 </div>
                              </div>
                            );
                          } catch (e) { 
                            return <div className="p-6 bg-red-50 border border-red-100 rounded-3xl text-red-600 text-xs font-mono italic">⚠️ Structural JSON Violation in Visualization Engine</div>; 
                          }
                        }

                        return !inline && match ? (
                          <div className="my-12 rounded-[2.5rem] overflow-hidden border border-slate-800 shadow-2xl group/code">
                             <div className="bg-slate-900 px-10 py-5 text-[10px] text-slate-500 font-black flex justify-between items-center border-b border-slate-800">
                                <span className="flex items-center gap-3 tracking-[0.3em] uppercase"><Terminal size={16}/> {match[1]} LAYER</span>
                                <button onClick={() => navigator.clipboard.writeText(codeContent)} className="flex items-center gap-2 hover:text-white transition-all text-[10px] bg-slate-800 px-4 py-2 rounded-xl"><Copy size={16}/> COPY SOURCE</button>
                             </div>
                             <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" className="!m-0 !p-10 !text-xs !bg-slate-900/95 leading-[1.8] font-mono">{codeContent}</SyntaxHighlighter>
                          </div>
                        ) : (<code className="bg-blue-50 px-2.5 py-1 rounded-xl text-blue-700 font-black border border-blue-100 mx-1" {...props}>{children}</code>);
                      }
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>
              </div>

              {msg.role === 'user' && (
                <div className="w-14 h-14 rounded-[1.5rem] bg-slate-900 flex items-center justify-center text-white flex-shrink-0 shadow-2xl mt-1 transform hover:-rotate-6 transition-transform">
                  <User size={30} />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* --- INPUT INTERFACE --- */}
        <div className="p-12 bg-white border-t border-slate-100 shadow-[0_-30px_60px_-15px_rgba(0,0,0,0.05)] sticky bottom-0 z-10 backdrop-blur-xl">
          <div className="max-w-5xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative group">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isLoading ? "Auditing multi-modal data streams..." : "Request an audit breakdown or a predictive churn model..."}
                disabled={isLoading || isUploading}
                className="w-full pl-10 pr-48 py-8 bg-slate-50 border-2 border-transparent rounded-[3rem] shadow-sm focus:bg-white focus:border-blue-500/30 focus:ring-[20px] focus:ring-blue-500/5 outline-none transition-all duration-700 disabled:opacity-50 font-bold text-lg text-slate-800 placeholder:text-slate-400 placeholder:font-medium"
              />
              <div className="absolute right-6 top-1/2 -translate-y-1/2 flex gap-5">
                <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" />
                <button type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading || isLoading} className="p-5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-[1.5rem] transition-all"><Paperclip size={28} /></button>
                <button type="submit" disabled={isLoading || !input.trim()} className="p-5 bg-blue-600 text-white rounded-[1.5rem] hover:bg-blue-700 shadow-2xl shadow-blue-600/40 transition-all active:scale-95 disabled:opacity-20"><Send size={28} /></button>
              </div>
            </form>
            <div className="flex justify-center gap-12 mt-8">
               <div className="flex items-center gap-3 text-[10px] text-slate-400 font-black uppercase tracking-[0.3em]"><CheckCircle2 size={14} className="text-green-500" /> SECURE SESSION ACTIVE</div>
               <div className="flex items-center gap-3 text-[10px] text-slate-400 font-black uppercase tracking-[0.3em]"><CheckCircle2 size={14} className="text-green-500" /> VIZ ENGINE ONLINE</div>
               <div className="flex items-center gap-3 text-[10px] text-slate-400 font-black uppercase tracking-[0.3em]"><AlertCircle size={14} className="text-blue-500" /> PRINCIPAL AUDIT LOCK</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}