import React, { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";
const API_KEY = import.meta.env.VITE_API_KEY || "dev-key";

const VAD_THRESHOLD = 30;      // higher = less sensitive to background noise
const SILENCE_MS    = 2500;    // wait 2.5s of silence before sending
const MIN_RECORD_MS = 800;     // ignore clips shorter than this

function timeStr() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function md(text) {
  if (!text) return "";
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>")
    .replace(/^- (.+)/gm, "• $1");
}

function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={"row " + (isUser ? "row-user" : "row-bot")}>
      <div className={"avatar " + (isUser ? "av-user" : "av-bot")}>
        {isUser ? "🧑" : "⚕️"}
      </div>
      <div className={"bubble " + (isUser ? "b-user" : "b-bot")}>
        {msg.sub && <div className="sub">{msg.sub}</div>}
        <div className="btext" dangerouslySetInnerHTML={{ __html: md(msg.text) }} />
        <div className="btime">
          {msg.voice && <span className="vtag">🎤 </span>}
          {msg.time}
        </div>
      </div>
    </div>
  );
}

function Waveform({ active }) {
  return (
    <div className={"waveform " + (active ? "wf-on" : "")}>
      {Array.from({ length: 9 }).map(function(_, i) {
        return <div key={i} className="wbar" style={{ animationDelay: (i * 0.08) + "s" }} />;
      })}
    </div>
  );
}

function RAGPanel({ open }) {
  const [docs, setDocs]       = useState([]);
  const [title, setTitle]     = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy]       = useState(false);
  const [msg, setMsg]         = useState("");
  const fileRef = useRef();

  const load = async () => {
    try {
      const r = await fetch(API_URL + "/documents", { headers: { "x-api-key": API_KEY } });
      const d = await r.json();
      setDocs(d.documents || []);
    } catch(e) { setDocs([]); }
  };

  useEffect(function() { if (open) load(); }, [open]);

  const upload = async () => {
    if (!content.trim() && !(fileRef.current && fileRef.current.files && fileRef.current.files.length)) {
      setMsg("Paste text or select a file"); return;
    }
    setBusy(true); setMsg("Uploading...");
    try {
      const form = new FormData();
      form.append("title", title || "Medical Document");
      if (fileRef.current && fileRef.current.files && fileRef.current.files.length) {
        form.append("file", fileRef.current.files[0]);
      } else {
        form.append("content", content);
      }
      const r = await fetch(API_URL + "/upload-document", {
        method: "POST", headers: { "x-api-key": API_KEY }, body: form,
      });
      const d = await r.json();
      if (d.success) {
        setMsg("Added successfully"); setTitle(""); setContent("");
        if (fileRef.current) fileRef.current.value = "";
        load();
      } else { setMsg("Error: " + d.error); }
    } catch(e) { setMsg("Error: " + e.message); }
    finally { setBusy(false); }
  };

  const del = async (id) => {
    await fetch(API_URL + "/documents/" + id, { method: "DELETE", headers: { "x-api-key": API_KEY } });
    load();
  };

  if (!open) return null;

  return (
    <div className="rag-panel">
      <div className="rag-title">📚 Knowledge Base</div>
      <input className="ri" placeholder="Document title (optional)" value={title}
        onChange={function(e) { setTitle(e.target.value); }} />
      <textarea className="rt" rows={3}
        placeholder="Paste medical text — drug info, clinical guidelines, disease info..."
        value={content} onChange={function(e) { setContent(e.target.value); }} />
      <div className="rag-row">
        <label className="file-btn">
          📎 Upload .txt / .pdf
          <input ref={fileRef} type="file" accept=".txt,.md,.pdf" style={{ display: "none" }} />
        </label>
        <button className="up-btn" onClick={upload} disabled={busy}>
          {busy ? "..." : "Add to KB"}
        </button>
      </div>
      {msg && <div className="rag-msg">{msg}</div>}
      <div className="doc-hdr">
        <span>Stored ({docs.length})</span>
        <button className="ref-btn" onClick={load}>↻</button>
      </div>
      {docs.length === 0
        ? <p className="no-docs">No documents yet.</p>
        : docs.map(function(d) {
          return (
            <div key={d.id} className="doc-row">
              <div>
                <div className="doc-name">{d.title}</div>
                <div className="doc-pre">{d.preview}...</div>
              </div>
              <button className="del-btn" onClick={function() { del(d.id); }}>✕</button>
            </div>
          );
        })
      }
    </div>
  );
}

export default function App() {
  const [messages,  setMessages]  = useState([{
    id: 0, role: "assistant", time: timeStr(),
    text: "Hello! I am MediAssist. Click the mic button to start — I will listen continuously. Describe your symptoms or ask any medical question. You can interrupt me anytime by speaking.",
  }]);
  const [micOn,     setMicOn]     = useState(false);
  const [vadState,  setVadState]  = useState("idle");
  const [ragOpen,   setRagOpen]   = useState(false);
  const [wsStatus,  setWsStatus]  = useState("connecting");
  const [textInput, setTextInput] = useState("");

  const wsRef        = useRef(null);
  const streamRef    = useRef(null);
  const recorderRef  = useRef(null);
  const chunksRef    = useRef([]);
  const silTimerRef  = useRef(null);
  const recStartRef  = useRef(0);
  const isRecRef     = useRef(false);
  const audioRef     = useRef(null);
  const reqIdRef     = useRef(0);
  const vadRef       = useRef("idle");
  const micOnRef     = useRef(false);
  const cooldownRef  = useRef(false);
  const messagesEnd  = useRef(null);

  const setVad = function(s) { vadRef.current = s; setVadState(s); };
  const setMic = function(v) { micOnRef.current = v; setMicOn(v); };

  useEffect(function() {
    if (messagesEnd.current) messagesEnd.current.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const addMsg = useCallback(function(m) {
    setMessages(function(p) {
      return [...p, Object.assign({ id: Date.now() + Math.random(), time: timeStr() }, m)];
    });
  }, []);

  // ── WebSocket ─────────────────────────────────────────────────────────────
  const connectWS = useCallback(function() {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen  = function() { setWsStatus("connected"); };
    ws.onclose = function() { setWsStatus("disconnected"); setTimeout(connectWS, 3000); };
    ws.onerror = function() { setWsStatus("error"); };

    ws.onmessage = async function(ev) {
      try {
        const raw  = typeof ev.data === "string" ? ev.data : await ev.data.text();
        const data = JSON.parse(raw);

        if (data.type === "processing") { setVad("processing"); return; }

        if (data.type === "voice_response" || data.type === "text_response") {
          if (data.reqId !== undefined && data.reqId !== reqIdRef.current) return;
          if (data.transcription) addMsg({ role: "user",      text: data.transcription, voice: true });
          if (data.response)      addMsg({ role: "assistant", text: data.response });
          if (data.audio_base64)  { playAudio(data.audio_base64); return; }
          setVad(micOnRef.current ? "listening" : "idle");
        }

        if (data.type === "error") {
          addMsg({ role: "assistant", text: "Error: " + data.message });
          setVad(micOnRef.current ? "listening" : "idle");
        }
      } catch(e) {
        setVad(micOnRef.current ? "listening" : "idle");
      }
    };
  }, [addMsg]);

  useEffect(function() {
    connectWS();
    return function() { if (wsRef.current) wsRef.current.close(); };
  }, []);

  // ── Audio playback ────────────────────────────────────────────────────────
  function stopAudio() {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
  }

  function playAudio(b64) {
    stopAudio();
    const bytes = Uint8Array.from(atob(b64), function(c) { return c.charCodeAt(0); });
    const url   = URL.createObjectURL(new Blob([bytes], { type: "audio/webm" }));
    const audio = new Audio(url);
    audioRef.current = audio;
    setVad("playing");
    cooldownRef.current = false;

    audio.onended = function() {
      audioRef.current = null;
      // 2 second cooldown — prevents mic picking up speaker echo
      cooldownRef.current = true;
      setTimeout(function() {
        cooldownRef.current = false;
        setVad(micOnRef.current ? "listening" : "idle");
      }, 2000);
    };

    audio.play().catch(function() {
      audioRef.current = null;
      setVad(micOnRef.current ? "listening" : "idle");
    });
  }

  // ── Recording ─────────────────────────────────────────────────────────────
  const startRec = useCallback(function(stream) {
    if (isRecRef.current) return;
    isRecRef.current    = true;
    recStartRef.current = Date.now();
    chunksRef.current   = [];

    stopAudio();
    cooldownRef.current = false;

    const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recorderRef.current = mr;

    mr.ondataavailable = function(e) { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = function() {
      isRecRef.current = false;
      const dur = Date.now() - recStartRef.current;
      if (!chunksRef.current.length || dur < MIN_RECORD_MS) {
        setVad(micOnRef.current ? "listening" : "idle");
        return;
      }
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      chunksRef.current = [];
      sendAudio(blob);
    };

    mr.start();
    setVad("recording");
  }, []);

  const stopRec = useCallback(function() {
    clearTimeout(silTimerRef.current);
    silTimerRef.current = null;
    if (recorderRef.current && isRecRef.current) recorderRef.current.stop();
  }, []);

  function sendAudio(blob) {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const myId = ++reqIdRef.current;
    setVad("processing");
    const reader = new FileReader();
    reader.onloadend = function() {
      wsRef.current.send(JSON.stringify({
        type:  "voice",
        audio: reader.result.split(",")[1],
        reqId: myId,
      }));
    };
    reader.readAsDataURL(blob);
  }

  // ── VAD loop ──────────────────────────────────────────────────────────────
  const startVAD = useCallback(function(stream) {
    const ctx      = new AudioContext();
    const src      = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);

    const data = new Uint8Array(analyser.fftSize);

    function tick() {
      if (!streamRef.current) return;
      analyser.getByteTimeDomainData(data);

      var talking = false;
      for (var i = 0; i < data.length; i++) {
        if (Math.abs(data[i] - 128) > VAD_THRESHOLD) { talking = true; break; }
      }

      const st = vadRef.current;

      if (talking) {
        if (silTimerRef.current) { clearTimeout(silTimerRef.current); silTimerRef.current = null; }

        // skip if in post-playback cooldown (prevents echo loop)
        if (cooldownRef.current) {
          requestAnimationFrame(tick);
          return;
        }

        if (st === "listening" || st === "playing") {
          startRec(stream);
        }
      } else {
        if (st === "recording" && !silTimerRef.current) {
          silTimerRef.current = setTimeout(function() {
            silTimerRef.current = null;
            stopRec();
          }, SILENCE_MS);
        }
      }
      requestAnimationFrame(tick);
    }
    tick();
  }, [startRec, stopRec]);

  // ── Toggle mic ────────────────────────────────────────────────────────────
  const toggleMic = async function() {
    if (micOnRef.current) {
      stopRec();
      stopAudio();
      cooldownRef.current = false;
      if (streamRef.current) streamRef.current.getTracks().forEach(function(t) { t.stop(); });
      streamRef.current = null;
      setMic(false);
      setVad("idle");
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        setMic(true);
        setVad("listening");
        startVAD(stream);
      } catch(e) {
        addMsg({ role: "assistant", text: "Microphone access denied. Please allow microphone access in your browser." });
      }
    }
  };

  // ── Text send ─────────────────────────────────────────────────────────────
  const sendText = function() {
    const text = textInput.trim();
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    addMsg({ role: "user", text: text });
    setTextInput("");
    const myId = ++reqIdRef.current;
    wsRef.current.send(JSON.stringify({ type: "chat", message: text, reqId: myId }));
    setVad("processing");
  };

  // ── UI ────────────────────────────────────────────────────────────────────
  const statusColor = { connected: "#22c55e", connecting: "#f59e0b", disconnected: "#ef4444", error: "#ef4444" };
  const stateLabel  = {
    idle:       "Click mic to start",
    listening:  "Listening...",
    recording:  "Hearing you...",
    processing: "Thinking...",
    playing:    "Speaking — talk to interrupt",
  };
  const micIcon = { idle: "🎙️", listening: "🎙️", recording: "🔴", processing: "⏳", playing: "🔊" };

  return (
    <div className="app">
      <header className="hdr">
        <div className="hdr-l">
          <span className="logo">⚕️</span>
          <div>
            <div className="app-name">MediAssist</div>
            <div className="app-sub">AI Medical Voice Assistant</div>
          </div>
        </div>
        <div className="hdr-r">
          <div className="ws-dot">
            <span className="dot" style={{ background: statusColor[wsStatus] }} />
            <span className="ws-lbl">{wsStatus}</span>
          </div>
          <button className={"kb-btn " + (ragOpen ? "kb-active" : "")}
            onClick={function() { setRagOpen(function(o) { return !o; }); }}>
            📚 {ragOpen ? "Hide KB" : "Knowledge Base"}
          </button>
        </div>
      </header>

      <div className="body">
        <div className="chat-col">
          <div className="msgs">
            {messages.map(function(m) { return <Bubble key={m.id} msg={m} />; })}
            {vadState === "processing" && (
              <div className="row row-bot">
                <div className="avatar av-bot">⚕️</div>
                <div className="bubble b-bot typing">
                  <span /><span /><span />
                </div>
              </div>
            )}
            <div ref={messagesEnd} />
          </div>

          <div className={"status-bar sb-" + vadState}>
            <Waveform active={vadState === "recording" || vadState === "playing"} />
            <span className="sb-label">{stateLabel[vadState]}</span>
          </div>

          <div className="controls">
            <div className="mic-wrap">
              <button
                className={"mic-btn mic-" + vadState + (micOn ? " mic-on" : "")}
                onClick={toggleMic}
              >
                <span className="mic-icon">{micIcon[vadState]}</span>
              </button>
              {micOn && vadState === "listening"  && <div className="pulse-ring" />}
              {micOn && vadState === "recording"  && <div className="pulse-ring ring-red" />}
              {micOn && vadState === "playing"    && <div className="pulse-ring ring-green" />}
            </div>

            <div className="txt-row">
              <textarea
                className="txt-in"
                rows={1}
                value={textInput}
                onChange={function(e) { setTextInput(e.target.value); }}
                onKeyDown={function(e) {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
                }}
                placeholder="Or type your question here..."
              />
              <button
                className="send-btn"
                onClick={sendText}
                disabled={!textInput.trim() || wsStatus !== "connected"}
              >➤</button>
            </div>
          </div>

          <p className="disclaimer">
            ⚠️ For informational purposes only — not a substitute for professional medical advice.
          </p>
        </div>

        <div className={"rag-col " + (ragOpen ? "rag-open" : "")}>
          <RAGPanel open={ragOpen} />
        </div>
      </div>
    </div>
  );
}