import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUp,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Database,
  Headphones,
  MessageCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  User,
  Wifi,
  XCircle,
  Zap,
} from "lucide-react";

import "./App.css";


const API_URL = "http://127.0.0.1:8000";


const QUICK_PROMPTS = [
  {
    title: "Late delivery",
    message: "My order was supposed to arrive yesterday but it hasn't arrived yet.",
    icon: Clock3,
  },
  {
    title: "Duplicate charge",
    message: "I was charged twice for the same order.",
    icon: Zap,
  },
  {
    title: "Missing refund",
    message: "I returned my order but haven't received my refund yet.",
    icon: RefreshCw,
  },
  {
    title: "Account security",
    message: "Someone accessed my Amazon account without my permission.",
    icon: ShieldCheck,
  },
];


function App() {

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm your AI support assistant. Tell me what happened and I'll help you find the best next step.",
      meta: null,
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  const [analysis, setAnalysis] = useState({
    intent: "general_support",
    intentConfidence: 0,
    intentSource: "waiting",
    escalationProbability: 0,
    escalationRequired: false,
    status: "ready",
    isFollowUp: false,
    contextSimilarity: 0,
    ragCount: 0,
  });

  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);


  // ---------------------------------------------------------
  // Backend health
  // ---------------------------------------------------------

  useEffect(() => {

    checkBackend();

    const interval = setInterval(checkBackend, 15000);

    return () => clearInterval(interval);

  }, []);


  async function checkBackend() {

    try {

      const response = await fetch(`${API_URL}/health`);

      setBackendOnline(response.ok);

    } catch {

      setBackendOnline(false);

    }

  }


  // ---------------------------------------------------------
  // Auto scroll
  // ---------------------------------------------------------

  useEffect(() => {

    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });

  }, [messages, loading]);


  // ---------------------------------------------------------
  // Input resize
  // ---------------------------------------------------------

  function handleInputChange(event) {

    setInput(event.target.value);

    const textarea = textareaRef.current;

    if (!textarea) return;

    textarea.style.height = "auto";

    textarea.style.height =
      Math.min(textarea.scrollHeight, 150) + "px";

  }


  // ---------------------------------------------------------
  // Send message
  // ---------------------------------------------------------

  async function sendMessage(messageOverride = null) {

    const message =
      (messageOverride ?? input).trim();

    if (!message || loading) return;


    setError("");
    setLoading(true);

    setInput("");

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }


    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
      meta: null,
    };


    const updatedMessages = [
      ...messages,
      userMessage,
    ];

    setMessages(updatedMessages);


    try {

      const conversationHistory =
        updatedMessages
          .filter((item) => item.role !== "system")
          .slice(-6)
          .map((item) => {

            if (item.role === "user") {
              return {
                customer: item.content,
                agent: "",
              };
            }

            return {
              customer: "",
              agent: item.content,
            };

          });


      const lastAnalysis =
        analysis.intent &&
          analysis.intent !== "general_support"
          ? analysis
          : null;


      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({

            message,

            conversation_history:
              conversationHistory,

            previous_intent:
              lastAnalysis?.intent ?? null,

            previous_intent_confidence:
              lastAnalysis?.intentConfidence ?? 0,

          }),
        }
      );


      const data = await response.json();


      if (!response.ok || data.error) {
        throw new Error(
          data.details ||
          data.error ||
          "The support service returned an error."
        );
      }


      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          data.response ||
          "I'm sorry, I wasn't able to generate a response.",
        meta: data,
      };


      setMessages((current) => [
        ...current,
        assistantMessage,
      ]);


      setAnalysis({

        intent:
          data.intent || "general_support",

        intentConfidence:
          data.intent_confidence || 0,

        intentSource:
          data.intent_source || "classifier",

        escalationProbability:
          data.escalation_probability || 0,

        escalationRequired:
          Boolean(data.escalation_required),

        status:
          data.status || "resolved",

        isFollowUp:
          Boolean(data.is_follow_up),

        contextSimilarity:
          data.context_similarity || 0,

        ragCount:
          Array.isArray(data.retrieved_examples)
            ? data.retrieved_examples.length
            : 0,

      });


    } catch (err) {

      setError(
        err.message ||
        "Unable to connect to the support agent."
      );

      setBackendOnline(false);

    } finally {

      setLoading(false);

    }

  }


  // ---------------------------------------------------------
  // Keyboard handling
  // ---------------------------------------------------------

  function handleKeyDown(event) {

    if (event.key === "Enter" && !event.shiftKey) {

      event.preventDefault();

      sendMessage();

    }

  }


  // ---------------------------------------------------------
  // New conversation
  // ---------------------------------------------------------

  function resetConversation() {

    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content:
          "Hi! I'm your AI support assistant. Tell me what happened and I'll help you find the best next step.",
        meta: null,
      },
    ]);


    setAnalysis({

      intent: "general_support",
      intentConfidence: 0,
      intentSource: "waiting",
      escalationProbability: 0,
      escalationRequired: false,
      status: "ready",
      isFollowUp: false,
      contextSimilarity: 0,
      ragCount: 0,

    });


    setInput("");
    setError("");

  }


  const hasConversation =
    messages.length > 1;


  return (

    <div className="app-shell">

      {/* Ambient background */}

      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="grid-overlay" />


      {/* =====================================================
          HEADER
      ===================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            <Sparkles size={19} strokeWidth={2.4} />
          </div>

          <div>
            <div className="brand-name">
              Support<span>AI</span>
            </div>

            <div className="brand-caption">
              Intelligent customer support
            </div>
          </div>

        </div>


        <div className="header-right">

          <div className="system-status">

            <span
              className={
                backendOnline
                  ? "status-dot online"
                  : "status-dot offline"
              }
            />

            <span>
              {backendOnline
                ? "AI system online"
                : "AI system offline"}
            </span>

          </div>


          <button
            className="new-chat-button"
            onClick={resetConversation}
          >

            <RefreshCw size={15} />

            New conversation

          </button>

        </div>

      </header>


      {/* =====================================================
          MAIN
      ===================================================== */}

      <main className="main-layout">


        {/* ===================================================
            CHAT AREA
        =================================================== */}

        <section className="chat-panel">


          <div className="chat-header">

            <div className="chat-title-row">

              <div className="agent-avatar">
                <Bot size={23} />
              </div>

              <div>

                <h1>
                  AmazonHelp Support
                </h1>

                <div className="agent-subtitle">

                  <span className="live-dot" />

                  AI support assistant

                </div>

              </div>

            </div>


            <div className="powered-badge">

              <Zap size={13} />

              Powered by AI

            </div>

          </div>


          {/* Messages */}

          <div className="messages-container">


            {!hasConversation && (

              <div className="welcome-section">

                <div className="welcome-orbit">

                  <div className="orbit-ring ring-one" />
                  <div className="orbit-ring ring-two" />

                  <div className="welcome-icon">
                    <Headphones size={32} />
                  </div>

                </div>


                <div className="welcome-copy">

                  <h2>
                    How can we help?
                  </h2>

                  <p>
                    Describe your issue naturally.
                    I'll understand the intent, search
                    historical support knowledge, and
                    generate a grounded response.
                  </p>

                </div>


                <div className="quick-grid">

                  {QUICK_PROMPTS.map((prompt) => {

                    const Icon = prompt.icon;

                    return (

                      <button
                        key={prompt.title}
                        className="quick-card"
                        onClick={() =>
                          sendMessage(prompt.message)
                        }
                        disabled={loading}
                      >

                        <div className="quick-icon">
                          <Icon size={17} />
                        </div>

                        <div className="quick-text">

                          <span>
                            {prompt.title}
                          </span>

                          <small>
                            Try an example
                          </small>

                        </div>

                        <ChevronRight
                          size={16}
                          className="quick-arrow"
                        />

                      </button>

                    );

                  })}

                </div>

              </div>

            )}


            {messages.map((message) => (

              <div
                key={message.id}
                className={`message-row ${message.role}`}
              >

                {message.role === "assistant" && (

                  <div className="message-avatar assistant-avatar">
                    <Bot size={17} />
                  </div>

                )}


                <div className="message-content">

                  <div className="message-label">

                    {message.role === "assistant"
                      ? "SupportAI"
                      : "You"}

                  </div>

                  <div className="message-bubble">

                    {message.content}

                  </div>


                  {message.meta && (

                    <div className="message-tags">

                      {message.meta.intent && (

                        <span className="message-tag">

                          <Activity size={11} />

                          {message.meta.intent}

                        </span>

                      )}

                      {message.meta.is_follow_up && (

                        <span className="message-tag context-tag">

                          <MessageCircle size={11} />

                          Context used

                        </span>

                      )}

                      {message.meta.escalation_required && (

                        <span className="message-tag escalation-tag">

                          <AlertTriangle size={11} />

                          Human support

                        </span>

                      )}

                    </div>

                  )}

                </div>


                {message.role === "user" && (

                  <div className="message-avatar user-avatar">
                    <User size={17} />
                  </div>

                )}

              </div>

            ))}


            {loading && (

              <div className="message-row assistant">

                <div className="message-avatar assistant-avatar">
                  <Bot size={17} />
                </div>

                <div className="message-content">

                  <div className="message-label">
                    SupportAI
                  </div>

                  <div className="typing-bubble">

                    <span />
                    <span />
                    <span />

                    <em>
                      Analyzing your request...
                    </em>

                  </div>

                </div>

              </div>

            )}


            {error && (

              <div className="error-banner">

                <XCircle size={18} />

                <div>

                  <strong>
                    Unable to reach the AI agent
                  </strong>

                  <span>
                    {error}
                  </span>

                </div>

              </div>

            )}


            <div ref={messagesEndRef} />

          </div>


          {/* Composer */}

          <div className="composer-area">

            <div className="composer">

              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Describe your issue..."
                rows={1}
                disabled={loading}
              />


              <button
                className={
                  input.trim()
                    ? "send-button active"
                    : "send-button"
                }
                onClick={() => sendMessage()}
                disabled={!input.trim() || loading}
                aria-label="Send message"
              >

                <ArrowUp size={19} />

              </button>

            </div>


            <div className="composer-footer">

              <span>
                Press Enter to send
              </span>

              <span className="privacy-note">

                <ShieldCheck size={12} />

                Don't share passwords or OTPs

              </span>

            </div>

          </div>

        </section>


        {/* ===================================================
            INTELLIGENCE PANEL
        =================================================== */}

        <aside className="insight-panel">


          <div className="insight-heading">

            <div>

              <div className="eyebrow">
                LIVE INTELLIGENCE
              </div>

              <h2>
                Agent analysis
              </h2>

            </div>

            <div className="pulse-icon">
              <Activity size={17} />
            </div>

          </div>


          {/* Intent */}

          <div className="analysis-card">

            <div className="analysis-label">

              <span>
                Detected intent
              </span>

              <Activity size={14} />

            </div>

            <div className="intent-value">

              {analysis.intent}

            </div>

            <div className="confidence-row">

              <span>
                Confidence
              </span>

              <strong>
                {Math.round(
                  analysis.intentConfidence * 100
                )}%
              </strong>

            </div>

            <div className="progress-track">

              <div
                className="progress-fill"
                style={{
                  width:
                    `${Math.min(
                      analysis.intentConfidence * 100,
                      100
                    )}%`,
                }}
              />

            </div>

            <div className="source-label">

              Source:
              <span>
                {formatSource(
                  analysis.intentSource
                )}
              </span>

            </div>

          </div>


          {/* Escalation */}

          <div
            className={
              analysis.escalationRequired
                ? "analysis-card escalation-card danger"
                : "analysis-card escalation-card"
            }
          >

            <div className="analysis-label">

              <span>
                Escalation
              </span>

              {analysis.escalationRequired
                ? <AlertTriangle size={15} />
                : <ShieldCheck size={15} />
              }

            </div>


            <div className="escalation-status">

              <div
                className={
                  analysis.escalationRequired
                    ? "status-icon danger-icon"
                    : "status-icon success-icon"
                }
              >

                {analysis.escalationRequired
                  ? <AlertTriangle size={18} />
                  : <CheckCircle2 size={18} />
                }

              </div>


              <div>

                <strong>

                  {analysis.escalationRequired
                    ? "Human support required"
                    : "No escalation needed"}

                </strong>

                <span>

                  Risk probability:
                  {" "}
                  {Math.round(
                    analysis.escalationProbability * 100
                  )}%

                </span>

              </div>

            </div>

          </div>


          {/* RAG */}

          <div className="analysis-card">

            <div className="analysis-label">

              <span>
                Knowledge grounding
              </span>

              <Database size={14} />

            </div>


            <div className="rag-status">

              <div className="rag-check">
                <CheckCircle2 size={15} />
              </div>

              <div>

                <strong>
                  Historical RAG active
                </strong>

                <span>
                  {analysis.ragCount > 0
                    ? `${analysis.ragCount} examples retrieved`
                    : "Waiting for a customer message"}
                </span>

              </div>

            </div>

          </div>


          {/* Context */}

          <div className="analysis-card">

            <div className="analysis-label">

              <span>
                Conversation context
              </span>

              <MessageCircle size={14} />

            </div>


            <div className="context-row">

              <div>

                <strong>

                  {analysis.isFollowUp
                    ? "Follow-up detected"
                    : "New request"}

                </strong>

                <span>

                  Similarity
                  {" "}
                  {Math.round(
                    analysis.contextSimilarity * 100
                  )}%

                </span>

              </div>


              <div
                className={
                  analysis.isFollowUp
                    ? "context-badge active"
                    : "context-badge"
                }
              >

                {analysis.isFollowUp
                  ? "ACTIVE"
                  : "READY"}

              </div>

            </div>

          </div>


          {/* Pipeline */}

          <div className="pipeline-card">

            <div className="pipeline-title">
              Processing pipeline
            </div>


            <PipelineStep
              icon={<Activity size={14} />}
              label="Intent classification"
              active={hasConversation}
              done={hasConversation}
            />

            <PipelineStep
              icon={<ShieldCheck size={14} />}
              label="Escalation check"
              active={hasConversation}
              done={hasConversation}
            />

            <PipelineStep
              icon={<Database size={14} />}
              label="Historical retrieval"
              active={analysis.ragCount > 0}
              done={analysis.ragCount > 0}
            />

            <PipelineStep
              icon={<Sparkles size={14} />}
              label="Grounded response"
              active={hasConversation}
              done={
                hasConversation &&
                !loading
              }
              last
            />

          </div>


          {/* Footer */}

          <div className="insight-footer">

            <div className="footer-status">

              <Wifi size={13} />

              <span>
                Local AI infrastructure
              </span>

            </div>

            <span>
              v1.0
            </span>

          </div>


        </aside>

      </main>


      {/* =====================================================
          BOTTOM STATUS
      ===================================================== */}

      <footer className="bottom-bar">

        <div className="bottom-item">

          <span className="mini-dot green" />

          <span>
            AI Support
          </span>

        </div>


        <div className="bottom-item">

          <Database size={13} />

          <span>
            RAG Grounded
          </span>

        </div>


        <div className="bottom-item">

          <ShieldCheck size={13} />

          <span>
            Safe escalation
          </span>

        </div>


        <div className="bottom-spacer" />

        <span className="powered-text">
          Built with LangGraph · FAISS · LLM
        </span>

      </footer>

    </div>

  );
}


// ===========================================================
// Pipeline component
// ===========================================================

function PipelineStep({
  icon,
  label,
  active,
  done,
  last,
}) {

  return (

    <div className="pipeline-step">

      <div
        className={
          active
            ? "pipeline-icon active"
            : "pipeline-icon"
        }
      >

        {done
          ? <CheckCircle2 size={13} />
          : icon}

      </div>


      <span>
        {label}
      </span>


      {!last && (
        <div
          className={
            done
              ? "pipeline-line complete"
              : "pipeline-line"
          }
        />
      )}

    </div>

  );

}


// ===========================================================
// Helpers
// ===========================================================

function formatSource(source) {

  if (!source || source === "waiting") {
    return "waiting";
  }

  if (source === "conversation_context") {
    return "conversation context";
  }

  if (
    source ===
    "conversation_context_low_confidence"
  ) {
    return "context + classifier";
  }

  return source;

}


export default App;