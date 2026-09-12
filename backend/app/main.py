from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
import sys


# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Import LangGraph agent
# ---------------------------------------------------------

from importlib.util import spec_from_file_location, module_from_spec


agent_path = PROJECT_ROOT / "11_langgraph_agent.py"

spec = spec_from_file_location("langgraph_agent", agent_path)
agent_module = module_from_spec(spec)
spec.loader.exec_module(agent_module)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="AmazonHelp AI Support Agent",
    description="AI-powered customer support agent with intent classification, RAG and escalation.",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []
    previous_intent: str | None = None
    previous_intent_confidence: float = 0.0


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AmazonHelp AI Support Agent",
    }


# ---------------------------------------------------------
# Chat
# ---------------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {
            "error": "Message cannot be empty."
        }

    try:

        result = agent_module.agent.invoke({

            "customer_message": message,

            "conversation_history":
                request.conversation_history,

            "previous_intent":
                request.previous_intent,

            "previous_intent_confidence":
                request.previous_intent_confidence
        })

        return {
            "message": message,

            "response":
                result.get("response", ""),

            "intent":
                result.get("intent", ""),

            "intent_confidence":
                result.get("intent_confidence", 0.0),

            "intent_source":
                result.get("intent_source", ""),

            "is_follow_up":
                result.get("is_follow_up", False),

            "context_similarity":
                result.get("context_similarity", 0.0),

            "escalation_probability":
                result.get(
                    "escalation_probability",
                    0.0
                ),

            "escalation_required":
                result.get(
                    "escalation_required",
                    False
                ),

            "status":
                result.get("status", ""),

            "retrieved_examples":
                result.get(
                    "retrieved_examples",
                    []
                )
        }

    except Exception as e:

        return {
            "error":
                "Unable to process the message.",

            "details":
                str(e)
        }

# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AmazonHelp AI Support Agent API",
        "status": "running",
    }