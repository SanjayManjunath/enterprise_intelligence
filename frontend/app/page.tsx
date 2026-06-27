"use client";

import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Plus, 
  MessageSquare, 
  Database, 
  ChevronLeft,
  ChevronRight,
  Loader2,
  Trash2,
  CheckCircle2,
  XCircle,
  Settings,
  User,
  Paperclip,
  Mic,
  Volume2,
  VolumeX
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// --- TYPES ---
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Thread {
  id: string;
  title: string;
  lastUpdated: Date;
}

// --- PRIVACY FILTER (CLIENT-SIDE DATA MASKING) ---
const maskPII = (text: string): string => {
  let masked = text;
  // 1. Credit Cards (Standard 16 digits with optional spaces/dashes)
  masked = masked.replace(/\b(?:\d{4}[-\s]?){3}\d{4}\b/g, '[REDACTED_CREDIT_CARD]');
  // 2. Email Addresses
  masked = masked.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[REDACTED_EMAIL]');
  // 3. Social Security Numbers (SSN)
  masked = masked.replace(/\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/g, '[REDACTED_SSN]');
  // 4. Phone Numbers (Standard formats)
  masked = masked.replace(/\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-\s]?\d{4}\b/g, '[REDACTED_PHONE]');
  return masked;
};

// --- PLOTLY COMPONENT ---
const PlotlyChart = ({ chartData }: { chartData: any }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const renderChart = () => {
      if (chartRef.current && (window as any).Plotly && chartData) {
        
        // ENTERPRISE THEME INJECTION
        const enterpriseLayout = {
          ...chartData.layout,
          font: { family: 'inherit', color: '#475569' },
          plot_bgcolor: 'transparent',
          paper_bgcolor: 'transparent',
          margin: { t: 50, r: 20, b: 50, l: 50 },
          xaxis: { ...chartData.layout?.xaxis, showgrid: true, gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0' },
          yaxis: { ...chartData.layout?.yaxis, showgrid: true, gridcolor: '#f1f5f9', zerolinecolor: '#e2e8f0' }
        };
        
        const enterpriseData = chartData.data.map((trace: any) => ({
          ...trace,
          marker: { ...trace.marker, size: 12, color: '#0f172a', opacity: 0.9, line: { width: 1.5, color: '#ffffff' } },
          line: { ...trace.line, width: 3, color: '#0f172a' }
        }));

        (window as any).Plotly.newPlot(chartRef.current, enterpriseData, enterpriseLayout, { 
          responsive: true,
          displayModeBar: true, 
          displaylogo: false
        });
      }
    };

    if (typeof window !== 'undefined') {
      if (!(window as any).Plotly) {
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
        script.onload = renderChart;
        document.head.appendChild(script);
      } else {
        renderChart();
      }
    }
  }, [chartData]);
  
  return <div ref={chartRef} className="w-full h-[380px] my-6 rounded-2xl border border-slate-200 bg-[#FAFAFA] p-4 shadow-sm" />;
};

// --- CUSTOM MARKDOWN RENDERER ---
const EnterpriseMarkdown = ({ content }: { content: string }) => {
  return (
    <ReactMarkdown 
      remarkPlugins={[remarkGfm]}
      components={{
        table: ({node, ...props}) => (
          <div className="table-container relative group my-6">
            <button 
              onClick={(e) => {
                const table = e.currentTarget.nextElementSibling?.querySelector('table');
                if(!table) return;
                const rows = Array.from(table.querySelectorAll('tr'));
                const csv = rows.map(row => {
                  return Array.from(row.querySelectorAll('th, td'))
                    .map(cell => `"${(cell as HTMLElement).innerText.replace(/"/g, '""')}"`)
                    .join(',');
                }).join('\n');
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'enterprise_audit_data.csv';
                link.click();
              }}
              className="absolute -top-3 right-4 hidden group-hover:block bg-slate-900 text-white text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 rounded-md shadow-md z-10 transition-all active:scale-95"
            >
              Download CSV
            </button>
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
              <table className="min-w-full text-sm text-left border-collapse" {...props} />
            </div>
          </div>
        ),
        th: ({node, ...props}) => <th className="bg-slate-50 px-4 py-3 font-semibold text-slate-900 border-b border-slate-200 whitespace-nowrap" {...props} />,
        td: ({node, ...props}) => <td className="px-4 py-3 border-b border-slate-100 text-slate-700" {...props} />,
        p: ({node, ...props}) => <p className="mb-4 last:mb-0" {...props} />,
        strong: ({node, ...props}) => <strong className="font-semibold text-slate-900" {...props} />
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default function EnterpriseAI() {
  // --- STATE ---
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  // --- VOICE STATE ---
  const [isListening, setIsListening] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const recognitionRef = useRef<any>(null);
  
  // --- STATE BLEED FIX: Mapped Dictionaries ---
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [loadingStates, setLoadingStates] = useState<Record<string, boolean>>({});
  const [uploadingStates, setUploadingStates] = useState<Record<string, boolean>>({});

  // Derived current states for clean JSX rendering
  const currentDraft = drafts[activeThreadId] || "";
  const isCurrentLoading = loadingStates[activeThreadId] || false;
  const isCurrentUploading = uploadingStates[activeThreadId] || false;
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- INITIALIZE SPEECH RECOGNITION ---
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;
      }
    }
  }, []);

  // --- VOICE LOGIC: SPEECH-TO-TEXT ---
  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }
    
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.onresult = (event: any) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setDrafts(prev => {
            const current = prev[activeThreadId] || "";
            return { ...prev, [activeThreadId]: current + (current.endsWith(' ') ? '' : ' ') + finalTranscript };
          });
        }
      };
      recognitionRef.current.onerror = () => setIsListening(false);
      recognitionRef.current.onend = () => setIsListening(false);
      
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {
        setIsListening(false);
      }
    }
  };

  // --- VOICE LOGIC: TEXT-TO-SPEECH ---
  const speakAsHuman = (text: string) => {
    if (!voiceEnabled || typeof window === 'undefined' || !window.speechSynthesis) return;
    
    window.speechSynthesis.cancel(); // Stop any current speech to prevent queue overlap
    
    // Scrub the text of markdown tables, plotly JSON, code blocks, and bold asterisks for a natural read
    let cleanText = text.replace(/```[\s\S]*?```/g, ' Code block omitted. ');
    cleanText = cleanText.replace(/\|.*\|/g, ''); 
    cleanText = cleanText.replace(/\[json_plotly\][\s\S]*?}/g, '');
    cleanText = cleanText.replace(/[*#_>]/g, '');
    cleanText = cleanText.trim();
    
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const voices = window.speechSynthesis.getVoices();
    
    // Seek out premium or natural-sounding system voices
    const preferredVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Premium')) || voices[0];
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    utterance.rate = 1.05;
    window.speechSynthesis.speak(utterance);
  };

  // --- BACKEND HEALTH CHECK ---
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health");
        if (res.ok) setBackendStatus('online');
        else setBackendStatus('offline');
      } catch {
        setBackendStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // --- THREAD INITIALIZATION ---
  useEffect(() => {
    const savedThreads = localStorage.getItem('ent_ai_threads');
    if (savedThreads) {
      const parsed = JSON.parse(savedThreads);
      setThreads(parsed);
      if (parsed.length > 0) setActiveThreadId(parsed[0].id);
      else createNewChat();
    } else {
      createNewChat();
    }
  }, []);

  // --- MESSAGES LOGIC ---
  useEffect(() => {
    if (activeThreadId) {
      const threadMessages = localStorage.getItem(`messages_${activeThreadId}`);
      setMessages(threadMessages ? JSON.parse(threadMessages) : []);
    }
  }, [activeThreadId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loadingStates]);

  // --- ACTIONS ---
  const createNewChat = () => {
    const newId = crypto.randomUUID();
    const newThread: Thread = { id: newId, title: "New Session", lastUpdated: new Date() };
    const updated = [newThread, ...threads];
    setThreads(updated);
    setActiveThreadId(newId);
    setMessages([]);
    localStorage.setItem('ent_ai_threads', JSON.stringify(updated));
  };

  const deleteThread = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = threads.filter(t => t.id !== id);
    setThreads(filtered);
    
    // Cleanup state maps
    setDrafts(prev => { const newDrafts = {...prev}; delete newDrafts[id]; return newDrafts; });
    
    localStorage.setItem('ent_ai_threads', JSON.stringify(filtered));
    localStorage.removeItem(`messages_${id}`);
    if (activeThreadId === id) {
      if (filtered.length > 0) setActiveThreadId(filtered[0].id);
      else createNewChat();
    }
  };

  // --- FILE UPLOAD LOGIC ---
  const handleFileClick = () => fileInputRef.current?.click();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !activeThreadId) return;

    setUploadingStates(prev => ({ ...prev, [activeThreadId]: true }));
    const formData = new FormData();
    formData.append('file', file);
    
    // FIX: Properly passing session_id and clearance_level as FormData fields
    formData.append('session_id', activeThreadId);
    formData.append('clearance_level', 'public');

    try {
      // FIX: Removed the query string from the fetch URL
      const res = await fetch(`/api/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        const systemMsg: Message = { 
          id: crypto.randomUUID(), 
          role: 'assistant', 
          content: `### 📎 File Ingested\n${data.message}`, 
          timestamp: new Date() 
        };
        const newHistory = [...messages, systemMsg];
        setMessages(newHistory);
        localStorage.setItem(`messages_${activeThreadId}`, JSON.stringify(newHistory));
      }
    } catch {
      alert("Failed to upload file. Ensure backend is running.");
    } finally {
      setUploadingStates(prev => ({ ...prev, [activeThreadId]: false }));
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const renderContent = (content: string) => {
    if (!content) return <span className="text-slate-400 italic">No response received.</span>;
    
    const markerIndex = content.indexOf('json_plotly]');
    
    if (markerIndex !== -1) {
      const textAfterMarker = content.substring(markerIndex + 'json_plotly]'.length);
      const firstBrace = textAfterMarker.indexOf('{');
      
      if (firstBrace !== -1) {
        const jsonString = textAfterMarker.substring(firstBrace);
        let chartData = null;
        let validJsonEnd = -1;
        
        let lastBrace = jsonString.lastIndexOf('}');
        while (lastBrace !== -1) {
          try {
            const attempt = jsonString.substring(0, lastBrace + 1);
            chartData = JSON.parse(attempt);
            validJsonEnd = lastBrace + 1;
            break; 
          } catch (e) {
            lastBrace = jsonString.lastIndexOf('}', lastBrace - 1);
          }
        }
        
        if (chartData) {
          let markerStart = content.lastIndexOf('[', markerIndex);
          if (markerStart === -1 || markerIndex - markerStart > 20) {
            markerStart = content.lastIndexOf(',', markerIndex); 
          }
          if (markerStart === -1 || markerIndex - markerStart > 20) {
            markerStart = markerIndex; 
          }

          let textBefore = content.substring(0, markerStart).trim();
          if (textBefore.endsWith('```json')) textBefore = textBefore.substring(0, textBefore.length - 7).trim();
          else if (textBefore.endsWith('```')) textBefore = textBefore.substring(0, textBefore.length - 3).trim();

          let textAfter = textAfterMarker.substring(firstBrace + validJsonEnd).trim();
          if (textAfter.startsWith('```')) textAfter = textAfter.substring(3).trim();
          
          return (
            <div className="space-y-4">
              <EnterpriseMarkdown content={textBefore} />
              <PlotlyChart chartData={chartData} />
              <EnterpriseMarkdown content={textAfter} />
            </div>
          );
        }
      }
    }
    
    return <EnterpriseMarkdown content={content} />;
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentDraft.trim() || isCurrentLoading) return;

    // Stop listening if user manually hits send
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    // --- EXECUTE PRIVACY FILTER ---
    const rawMsgContent = currentDraft;
    const safeMsgContent = maskPII(rawMsgContent);

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: safeMsgContent, timestamp: new Date() };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    localStorage.setItem(`messages_${activeThreadId}`, JSON.stringify(newHistory));
    
    // --- SMART TITLE EXTRACTOR FIX ---
    if (messages.length === 0) {
      const cleanPrompt = safeMsgContent.replace(/^(generate a|create a|find all|what is|how to|give me a|can you)\s+/i, '');
      const words = cleanPrompt.split(' ');
      const displayTitle = words.slice(0, 5).join(' ') + (words.length > 5 ? '...' : '');
      const finalTitle = displayTitle.charAt(0).toUpperCase() + displayTitle.slice(1);

      setThreads(prev => {
        const updated = prev.map(t => t.id === activeThreadId ? { ...t, title: finalTitle } : t);
        localStorage.setItem('ent_ai_threads', JSON.stringify(updated));
        return updated;
      });
    }

    setDrafts(prev => ({ ...prev, [activeThreadId]: "" }));
    setLoadingStates(prev => ({ ...prev, [activeThreadId]: true }));

    try {
      const res = await fetch(`/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Send the scrubbed content to the backend to protect API connections
        body: JSON.stringify({ question: safeMsgContent, thread_id: activeThreadId }),
      });
      const data = await res.json();
      const outputText = data.reply || data.final_output || data.response || "";
      const aiMsg: Message = { id: crypto.randomUUID(), role: 'assistant', content: outputText, timestamp: new Date() };
      
      const finalHistory = [...newHistory, aiMsg];
      setMessages(finalHistory);
      localStorage.setItem(`messages_${activeThreadId}`, JSON.stringify(finalHistory));
      
      // TRIGGER VOICE IF ENABLED
      speakAsHuman(outputText);
      
    } catch {
      setMessages(prev => [...prev, { id: 'err', role: 'assistant', content: "### 🔴 System Offline\nThe Enterprise AI backend is unreachable.", timestamp: new Date() }]);
    } finally { 
      setLoadingStates(prev => ({ ...prev, [activeThreadId]: false })); 
    }
  };

  return (
    <div className="flex h-screen bg-white text-[#1D1D1F] font-sans overflow-hidden">
      {/* HIDDEN FILE INPUT */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
        accept=".csv,.xlsx,.odt,.ipynb,.mp3,.wav,.pdf"
      />

      {/* SIDEBAR */}
      <aside className={`${isSidebarOpen ? 'w-[300px]' : 'w-0'} transition-all duration-300 bg-[#FBFBFA] border-r flex flex-col shrink-0`}>
        <div className="p-6">
          <div className="flex items-center justify-between mb-8 px-1">
            <div className="flex items-center gap-2.5">
              <div className="size-8 bg-black rounded-lg flex items-center justify-center text-white">
                <Database size={16} />
              </div>
              <span className="font-bold text-lg tracking-tight">Enterprise AI</span>
            </div>
            <button onClick={() => setIsSidebarOpen(false)} className="text-slate-400 hover:text-black">
              <ChevronLeft size={20} />
            </button>
          </div>
          <button onClick={createNewChat} className="w-full flex items-center gap-3 px-4 py-3 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all text-sm font-semibold shadow-sm">
            <Plus size={18} className="text-slate-500" /> New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          <div className="px-3 mb-2 text-[10px] font-bold text-slate-400 uppercase tracking-widest">History</div>
          {threads.map(t => (
            <div key={t.id} onClick={() => setActiveThreadId(t.id)} className={`group flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all ${activeThreadId === t.id ? 'bg-[#EFEFEF]' : 'hover:bg-[#F3F3F2]'}`}>
              <MessageSquare size={16} className={activeThreadId === t.id ? 'text-black' : 'text-slate-400'} />
              <span className={`text-sm truncate flex-1 ${activeThreadId === t.id ? 'font-semibold text-black' : 'font-medium text-slate-600'}`}>
                {t.title}
              </span>
              <button onClick={(e) => deleteThread(t.id, e)} className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-500">
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="p-6 border-t flex items-center justify-between text-slate-400">
          <button className="flex items-center gap-2 text-xs font-bold uppercase tracking-tighter hover:text-black">
            <Settings size={14} /> Config
          </button>
          <span className="text-[10px] font-bold tracking-tighter opacity-50">v1.1.5</span>
        </div>
      </aside>

      {/* MAIN VIEW */}
      <main className="flex-1 flex flex-col relative bg-white overflow-hidden">
        {!isSidebarOpen && (
          <button onClick={() => setIsSidebarOpen(true)} className="absolute top-6 left-6 z-20 size-10 bg-white border border-slate-200 rounded-xl flex items-center justify-center text-slate-400 hover:text-black shadow-sm">
            <ChevronRight size={20} />
          </button>
        )}

        <header className="h-16 flex items-center justify-end px-8 border-b border-slate-50 bg-white/50 backdrop-blur-xl z-10">
          <div className="flex items-center gap-6">
            <button 
              onClick={() => {
                setVoiceEnabled(!voiceEnabled);
                if (voiceEnabled && typeof window !== 'undefined' && window.speechSynthesis) {
                  window.speechSynthesis.cancel();
                }
              }}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-100 rounded-full hover:bg-slate-100 transition-colors"
              title={voiceEnabled ? "Mute Voice Output" : "Enable Voice Output"}
            >
              {voiceEnabled ? <Volume2 size={14} className="text-blue-600" /> : <VolumeX size={14} className="text-slate-400" />}
            </button>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-100 rounded-full">
              {backendStatus === 'online' ? <CheckCircle2 size={12} className="text-green-500" /> : <XCircle size={12} className="text-red-500" />}
              <span className={`text-[9px] font-bold uppercase tracking-widest ${backendStatus === 'online' ? 'text-green-700' : 'text-red-700'}`}>
                {backendStatus === 'online' ? 'Backend Live' : 'Backend Offline'}
              </span>
            </div>
            <div className="size-9 bg-slate-900 rounded-full flex items-center justify-center text-xs font-bold text-white border-2 border-white ring-1 ring-slate-200 shadow-sm">SM</div>
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto py-16 px-6">
            {messages.length === 0 && (
              <div className="py-24 text-center">
                <h2 className="text-4xl font-bold tracking-tight text-slate-200 mb-2 italic">Enterprise AI</h2>
                <p className="text-slate-400 text-sm font-medium">Precision audit engine active.</p>
              </div>
            )}
            <div className="space-y-12">
              {messages.map(m => (
                <div key={m.id} className={`flex gap-8 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`size-8 rounded-lg flex items-center justify-center shrink-0 mt-1 ${m.role === 'assistant' ? 'bg-black text-white' : 'bg-slate-100 text-slate-500'}`}>
                    {m.role === 'assistant' ? <Database size={15} /> : <User size={15} />}
                  </div>
                  <div className={`flex-1 text-[16px] leading-[1.6] ${m.role === 'user' ? 'text-right text-slate-800 font-medium' : 'text-left text-slate-700'}`}>
                    {renderContent(m.content)}
                  </div>
                </div>
              ))}
              {(isCurrentLoading || isCurrentUploading) && (
                <div className="flex gap-8">
                  <div className="size-8 rounded-lg bg-black text-white flex items-center justify-center animate-pulse">
                    <Database size={15} />
                  </div>
                  <div className="flex items-center gap-3">
                    <Loader2 size={16} className="animate-spin text-slate-300" />
                    <span className="text-slate-400 text-sm font-medium">{isCurrentUploading ? "Ingesting file..." : "Processing request..."}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* INPUT FORM WITH FILE ATTACHMENT AND VOICE DICTATION */}
        <div className="pb-12 pt-4 px-6 bg-gradient-to-t from-white via-white to-transparent">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={sendMessage} className="relative bg-[#F4F4F3] rounded-[24px] p-2 focus-within:ring-1 focus-within:ring-slate-300 shadow-sm transition-all">
              <div className="absolute left-3 bottom-3 flex items-center gap-1">
                <button 
                  type="button" 
                  onClick={handleFileClick}
                  disabled={isCurrentUploading || isCurrentLoading}
                  className="size-10 rounded-2xl flex items-center justify-center text-slate-400 hover:text-black hover:bg-slate-200 transition-all"
                  title="Upload Context Document"
                >
                  <Paperclip size={18} />
                </button>
                <button 
                  type="button" 
                  onClick={toggleListening}
                  disabled={isCurrentUploading || isCurrentLoading}
                  className={`size-10 rounded-2xl flex items-center justify-center transition-all ${
                    isListening ? 'bg-red-100 text-red-500 animate-pulse' : 'text-slate-400 hover:text-black hover:bg-slate-200'
                  }`}
                  title={isListening ? "Stop Listening" : "Start Voice Dictation"}
                >
                  <Mic size={18} />
                </button>
              </div>
              <textarea
                value={currentDraft}
                onChange={(e) => setDrafts(prev => ({ ...prev, [activeThreadId]: e.target.value }))}
                placeholder={isListening ? "Listening..." : "Message Enterprise AI..."}
                rows={1}
                className="w-full bg-transparent p-4 pl-[90px] pr-14 resize-none outline-none text-[16px] placeholder-slate-400 min-h-[56px]"
                onKeyDown={(e) => { 
                  if(e.key === 'Enter' && !e.shiftKey) { 
                    e.preventDefault(); 
                    sendMessage(e); 
                  } 
                }}
              />
              <button 
                type="submit" 
                disabled={!currentDraft.trim() || isCurrentLoading || isCurrentUploading}
                className={`absolute right-3 bottom-3 size-10 rounded-2xl flex items-center justify-center transition-all ${
                  currentDraft.trim() ? 'bg-black text-white active:scale-95' : 'bg-slate-200 text-slate-400'
                }`}
              >
                <Send size={18} />
              </button>
            </form>
            <p className="text-center text-[10px] text-slate-300 mt-4 font-bold uppercase tracking-[0.2em] select-none">
              Persistence Active • {activeThreadId.split('-')[0]}
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}