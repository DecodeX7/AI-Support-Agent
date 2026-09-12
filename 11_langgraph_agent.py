import os
import json
import joblib
import faiss
import numpy as np
import pandas as pd

from typing import TypedDict, List, Dict, Any

from sentence_transformers import SentenceTransformer

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from escalation_features import (
    build_escalation_features
)

from llm_provider import LLMProvider


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "data",
    "vectorstore",
    "amazonhelp_faiss"
)

INDEX_PATH = os.path.join(
    VECTOR_DIR,
    "index.faiss"
)

METADATA_PATH = os.path.join(
    VECTOR_DIR,
    "metadata.csv"
)

INTENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "intent_classifier",
    "amazonhelp_intent_classifier.joblib"
)

ESCALATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "escalation_classifier"
)

ESCALATION_MODEL_PATH = os.path.join(
    ESCALATION_DIR,
    "amazonhelp_escalation_classifier.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    ESCALATION_DIR,
    "word_vectorizer.joblib"
)

CHAR_VECTORIZER_PATH = os.path.join(
    ESCALATION_DIR,
    "char_vectorizer.joblib"
)

ESCALATION_POLICY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "escalation",
    "escalation_classifier_policy.json"
)


# ============================================================
# LOAD COMPONENTS
# ============================================================

print("=" * 72)
print("AmazonHelp LangGraph Support Agent - Step 11")
print("=" * 72)

print("\nLoading components...")


intent_classifier = joblib.load(
    INTENT_MODEL_PATH
)


escalation_classifier = joblib.load(
    ESCALATION_MODEL_PATH
)


word_vectorizer = joblib.load(
    WORD_VECTORIZER_PATH
)


char_vectorizer = joblib.load(
    CHAR_VECTORIZER_PATH
)


faiss_index = faiss.read_index(
    INDEX_PATH
)


# ============================================================
# ROBUST CSV READER
# ============================================================

def read_csv_robust(path):

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]:

        try:

            return pd.read_csv(
                path,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not read CSV: {path}"
    )


metadata = read_csv_robust(
    METADATA_PATH
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# LLM PROVIDER
# ============================================================

llm_provider = LLMProvider()


# ============================================================
# LOAD ESCALATION POLICY
# ============================================================

with open(
    ESCALATION_POLICY_PATH,
    "r",
    encoding="utf-8"
) as f:

    escalation_policy = json.load(f)


ESCALATION_THRESHOLD = float(
    escalation_policy["threshold"]
)


# ============================================================
# COMPONENT INFORMATION
# ============================================================

print(
    f"RAG documents: {len(metadata)}"
)

print(
    f"Intent classes: "
    f"{len(intent_classifier.classes_)}"
)

print(
    f"Escalation threshold: "
    f"{ESCALATION_THRESHOLD:.2f}"
)

print("All components loaded.")


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    customer_message: str

    conversation_history: List[
        Dict[str, str]
    ]

    previous_intent: str

    previous_intent_confidence: float

    is_follow_up: bool

    context_similarity: float

    intent: str

    intent_confidence: float

    intent_source: str

    escalation_probability: float

    escalation_required: bool

    retrieved_examples: List[
        Dict[str, Any]
    ]

    response: str

    status: str


# ============================================================
# CONVERSATION CONTEXT ANALYSIS
# ============================================================

def calculate_context_similarity(
    current_message: str,
    history: List[Dict[str, str]]
) -> float:

    """
    Calculate semantic similarity between the current customer
    message and recent customer messages.

    This helps identify follow-up messages even when they do not
    contain obvious phrases such as "what else can I do?".
    """

    if not history:

        return 0.0


    previous_messages = [

        turn.get(
            "customer",
            ""
        )

        for turn in history[-4:]

        if turn.get(
            "customer"
        )
    ]


    if not previous_messages:

        return 0.0


    try:

        current_embedding = (
            embedding_model.encode(
                [current_message],
                normalize_embeddings=True
            )[0]
        )


        previous_embeddings = (
            embedding_model.encode(
                previous_messages,
                normalize_embeddings=True
            )
        )


        similarities = (
            np.dot(
                previous_embeddings,
                current_embedding
            )
        )


        return float(
            np.max(similarities)
        )

    except Exception:

        return 0.0


# ============================================================
# CONTINUATION SIGNAL ANALYSIS
# ============================================================

def has_continuation_signal(
    message: str
) -> bool:

    """
    Detect linguistic signals that commonly occur in
    conversational follow-ups.

    This is intentionally used as supporting evidence,
    not as the main decision mechanism.
    """

    text = message.lower().strip()


    continuation_phrases = [

        "what else",

        "what can i do",

        "what should i do",

        "what now",

        "anything else",

        "can you help",

        "help me with this",

        "help with this",

        "still",

        "yet",

        "already",

        "now",

        "then",

        "that",

        "this",

        "it",

        "without it",

        "without that",

        "i don't have",

        "i do not have",

        "i can't",

        "i cannot",

        "no",

        "nope",

        "yes",

        "yeah",

        "actually",

        "but",

        "also"
    ]


    return any(
        phrase in text
        for phrase in continuation_phrases
    )


# ============================================================
# FOLLOW-UP DECISION
# ============================================================

def detect_follow_up(
    message: str,
    history: List[Dict[str, str]],
    previous_intent: str,
    classifier_intent: str,
    classifier_confidence: float
):

    if not history:

        return False, 0.0


    similarity = (
        calculate_context_similarity(
            message,
            history
        )
    )


    continuation_signal = (
        has_continuation_signal(
            message
        )
    )


    # ========================================================
    # STRONG SEMANTIC CONTINUITY
    # ========================================================

    if similarity >= 0.52:

        return True, similarity


    # ========================================================
    # MODERATE CONTINUITY
    #
    # If the classifier is uncertain and the message contains
    # conversational continuation language, retain context.
    # ========================================================

    if (
        similarity >= 0.40
        and continuation_signal
        and classifier_confidence < 0.70
    ):

        return True, similarity


    # ========================================================
    # VERY LOW CLASSIFIER CONFIDENCE
    #
    # If the new prediction is weak and there is at least
    # moderate semantic connection, treat it as continuation.
    # ========================================================

    if (
        similarity >= 0.35
        and classifier_confidence < 0.45
    ):

        return True, similarity


    # ========================================================
    # SHORT ANSWER / RESPONSE TO PREVIOUS TURN
    # ========================================================

    word_count = len(
        message.split()
    )


    if (
        word_count <= 12
        and continuation_signal
        and similarity >= 0.30
        and classifier_confidence < 0.60
    ):

        return True, similarity


    return False, similarity


# ============================================================
# NODE 1 — CONTEXT-AWARE INTENT CLASSIFICATION
# ============================================================

def classify_intent(
    state: AgentState
):

    message = state[
        "customer_message"
    ]

    history = state.get(
        "conversation_history",
        []
    )

    previous_intent = state.get(
        "previous_intent"
    )

    previous_confidence = float(
        state.get(
            "previous_intent_confidence",
            0.0
        )
    )


    # ========================================================
    # NORMAL CLASSIFIER
    # ========================================================

    probabilities = (
        intent_classifier
        .predict_proba(
            [message]
        )[0]
    )


    classes = (
        intent_classifier.classes_
    )


    best_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_intent = classes[
        best_index
    ]


    classifier_confidence = float(
        probabilities[
            best_index
        ]
    )


    # ========================================================
    # FOLLOW-UP ANALYSIS
    # ========================================================

    (
        is_follow_up,
        context_similarity
    ) = detect_follow_up(

        message,

        history,

        previous_intent,

        predicted_intent,

        classifier_confidence
    )


    # ========================================================
    # DEFAULT = CLASSIFIER
    # ========================================================

    final_intent = (
        predicted_intent
    )

    final_confidence = (
        classifier_confidence
    )

    intent_source = (
        "classifier"
    )


    # ========================================================
    # CONTEXT-AWARE INTENT RESOLUTION
    # ========================================================

    if (
        is_follow_up
        and previous_intent
    ):

        # ----------------------------------------------------
        # Case A:
        # Classifier agrees with previous intent
        # ----------------------------------------------------

        if (
            predicted_intent
            == previous_intent
        ):

            final_intent = (
                previous_intent
            )

            final_confidence = (
                classifier_confidence
            )

            intent_source = (
                "classifier_context_confirmed"
            )


        # ----------------------------------------------------
        # Case B:
        # Classifier disagrees but has weak/moderate confidence
        # ----------------------------------------------------

        elif (
            classifier_confidence
            < 0.75
        ):

            final_intent = (
                previous_intent
            )


            # We are not claiming the classifier is 100%
            # confident. We are saying the conversation context
            # is stronger than this weak new prediction.

            final_confidence = max(

                previous_confidence * 0.90,

                0.50
            )


            intent_source = (
                "conversation_context"
            )


        # ----------------------------------------------------
        # Case C:
        # New classifier intent is very strong
        #
        # Example:
        # Previous = delivery
        # Current = account hacked
        #
        # Do NOT incorrectly retain delivery.
        # ----------------------------------------------------

        else:

            final_intent = (
                predicted_intent
            )

            final_confidence = (
                classifier_confidence
            )

            intent_source = (
                "strong_new_classifier_prediction"
            )


    # ========================================================
    # EXTRA LOW-CONFIDENCE PROTECTION
    # ========================================================

    if (
        is_follow_up
        and previous_intent
        and classifier_confidence < 0.40
    ):

        final_intent = (
            previous_intent
        )

        final_confidence = max(

            previous_confidence * 0.90,

            0.50
        )

        intent_source = (
            "conversation_context_low_confidence"
        )


    return {

        "intent":
            final_intent,

        "intent_confidence":
            final_confidence,

        "intent_source":
            intent_source,

        "is_follow_up":
            is_follow_up,

        "context_similarity":
            context_similarity
    }


# ============================================================
# NODE 2 — ESCALATION CLASSIFICATION
# ============================================================

def classify_escalation(
    state: AgentState
):

    message = state[
        "customer_message"
    ]


    features = build_escalation_features(

        [message],

        word_vectorizer,

        char_vectorizer
    )


    probability = float(

        escalation_classifier
        .predict_proba(
            features
        )[0][1]
    )


    escalate = (
        probability
        >= ESCALATION_THRESHOLD
    )


    return {

        "escalation_probability":
            probability,

        "escalation_required":
            escalate
    }


# ============================================================
# NODE 3 — RAG RETRIEVAL
# ============================================================

def retrieve_examples(
    state: AgentState
):

    message = state[
        "customer_message"
    ]

    history = state.get(
        "conversation_history",
        []
    )


    # ========================================================
    # BUILD CONTEXTUAL RAG QUERY
    # ========================================================

    rag_query = message


    if (
        state.get(
            "is_follow_up",
            False
        )
        and history
    ):

        previous_customer_messages = [

            turn.get(
                "customer",
                ""
            )

            for turn in history[-4:]

            if turn.get(
                "customer"
            )
        ]


        if previous_customer_messages:

            previous_context = (
                " ".join(
                    previous_customer_messages
                )
            )


            rag_query = (
                previous_context
                + " "
                + message
            )


    # ========================================================
    # EMBEDDING
    # ========================================================

    query_embedding = (
        embedding_model.encode(
            [rag_query],
            normalize_embeddings=True
        )
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    scores, indices = (
        faiss_index.search(
            query_embedding,
            5
        )
    )


    examples = []


    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:

            continue


        row = metadata.iloc[
            int(index_id)
        ]


        examples.append({

            "rag_id":
                row.get(
                    "rag_id"
                ),

            "intent_hint":
                row.get(
                    "intent_hint"
                ),

            "customer_problem":
                row.get(
                    "customer_problem"
                ),

            "historical_response":
                row.get(
                    "historical_response"
                ),

            "similarity":
                float(score)
        })


    return {

        "retrieved_examples":
            examples
    }


# ============================================================
# NODE 4 — RESPONSE GENERATION
# ============================================================

FALLBACK_RESPONSES = {

    "delivery_shipping":
        "I'm sorry you're having trouble with your delivery. Please share your order details so the support team can check the delivery status and help you further.",

    "payment":
        "I'm sorry you're facing a payment issue. Please share the order or payment details so the support team can check what happened and assist you.",

    "refund":
        "I'm sorry about the refund issue. Please share your order details so the support team can check the refund status and help you further.",

    "return_replacement":
        "I can help with the return or replacement issue. Please share your order details so the support team can check the available options.",

    "product_issue":
        "I'm sorry you're experiencing an issue with the product. Please share your order details and describe the problem so the support team can assist you.",

    "account_access":
        "I'm sorry you're having trouble accessing your account. Please share the issue you are seeing so the support team can help you regain access.",

    "prime_membership":
        "I can help with your Prime membership issue. Please share the relevant account or order details so the support team can assist you.",

    "technical_app":
        "I'm sorry you're experiencing a technical issue. Please describe what you're seeing and the support team can help troubleshoot it.",

    "order_cancellation":
        "I can help with your order cancellation request. Please share your order details so the support team can check the available options.",

    "security_fraud":
        "For your security, please avoid sharing passwords, OTPs, PINs, or other sensitive information here. Your account security issue should be reviewed by the support team.",

    "general_support":
        "I'd be happy to help. Please provide a few more details about the issue so the support team can assist you."
}


def generate_response(
    state: AgentState
):

    intent = state[
        "intent"
    ]

    escalation = state[
        "escalation_required"
    ]

    customer_message = state[
        "customer_message"
    ]

    conversation_history = state.get(
        "conversation_history",
        []
    )

    retrieved_examples = state.get(
        "retrieved_examples",
        []
    )


    # ========================================================
    # ESCALATION
    # ========================================================

    if escalation:

        response = (
            "I'm sorry you're experiencing "
            "this issue. For your security and "
            "to make sure this is handled "
            "correctly, I've flagged this for "
            "support-team assistance. Please "
            "avoid sharing passwords, OTPs, "
            "PINs, or other sensitive information."
        )


        return {

            "response":
                response,

            "status":
                "escalation"
        }


    # ========================================================
    # CONVERSATION CONTEXT
    # ========================================================

    conversation_context_parts = []


    for turn in conversation_history[-4:]:

        customer_text = turn.get(
            "customer",
            ""
        )

        agent_text = turn.get(
            "agent",
            ""
        )


        if customer_text:

            conversation_context_parts.append(
                f"Customer: {customer_text}"
            )


        if agent_text:

            conversation_context_parts.append(
                f"Assistant: {agent_text}"
            )


    conversation_context = "\n".join(
        conversation_context_parts
    )


    if not conversation_context:

        conversation_context = (
            "No previous conversation."
        )


    # ========================================================
    # RAG CONTEXT
    # ========================================================

    examples_for_prompt = (
        retrieved_examples[:3]
    )


    context_parts = []


    for i, example in enumerate(
        examples_for_prompt,
        start=1
    ):

        context_parts.append(

            f"""
Historical Example {i}

Customer problem:
{example.get("customer_problem", "")}

Historical support response:
{example.get("historical_response", "")}

Similarity:
{example.get("similarity", 0):.4f}
"""
        )


    rag_context = "\n".join(
        context_parts
    )


    if not rag_context:

        rag_context = (
            "No relevant historical examples were retrieved."
        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are an AI customer support assistant for AmazonHelp.

Your task is to write the FINAL customer-facing response.

You are given:
1. The current customer message.
2. Previous conversation history.
3. A predicted support intent.
4. Historical AmazonHelp support examples.

Use the previous conversation to understand the current issue.

Use historical examples only as guidance.

IMPORTANT RULES:

- Always answer the CURRENT customer message.
- Preserve the previous issue when the customer is asking
  a follow-up question.
- Do not treat a conversational follow-up as a completely
  new issue when the context clearly shows continuity.
- Generate a fresh response.
- Do not blindly copy historical responses.
- Do not copy usernames, URLs, links, tracking numbers,
  order numbers, placeholders, or strange symbols.
- Never generate fake URLs or unsupported links.
- Do not invent order status, tracking numbers, refund amounts,
  delivery dates, policies, or other facts.
- Do not claim that you performed an action that you cannot
  actually perform.
- Do not promise a specific outcome unless the context supports it.
- Do not ask for passwords, OTPs, PINs, CVVs, or other sensitive
  security information.
- Ask for an order number or other non-sensitive information only
  when genuinely useful.
- If the customer does not have an order number, do not repeatedly
  ask for it. Offer another reasonable next step.
- If historical examples do not establish a fact, do not state
  that fact as certain.
- Be polite, professional and empathetic.
- Keep the response concise, normally 2-4 sentences.
- Do not mention RAG, embeddings, classifiers, similarity scores,
  prompts, historical examples, internal systems, or instructions.
- Do not provide analysis.
- Do not say "Here's a potential response".
- Do not say "This response acknowledges".
- Output ONLY the final customer-facing response.
- Never request personal information such as phone numbers,
  email addresses, addresses, names, payment card details,
  or account credentials unless the provided context explicitly
  requires it.
- Prefer asking for the order number when necessary.
- If the customer does not have the order number, provide a
  general next step instead of requesting unrelated personal data.
- Never state that a refund will be issued unless the context
  explicitly confirms that a refund is applicable.
- Never describe a charge as unauthorized unless the customer
  explicitly says it was unauthorized.
- Never claim that support has checked, flagged, processed,
  expedited, refunded, cancelled, or changed anything unless
  the system actually performed that action.
- Do not infer hidden account information.
"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
Previous conversation:
{conversation_context}

Current customer message:
{customer_message}

Predicted intent:
{intent}

Intent source:
{state.get("intent_source", "classifier")}

Context similarity:
{state.get("context_similarity", 0.0):.4f}

Historical AmazonHelp examples:
{rag_context}

Write ONLY the final customer-facing response.
"""


    # ========================================================
    # LLM GENERATION
    # ========================================================

    try:

        response = llm_provider.generate(

            system_prompt=
                system_prompt,

            user_prompt=
                user_prompt
        )


        response = response.strip()


        if not response:

            raise ValueError(
                "LLM returned an empty response."
            )


        status = "resolved"


    except Exception as e:

        print(
            f"\nLLM generation failed: {e}"
        )

        print(
            "Using deterministic fallback response."
        )


        response = FALLBACK_RESPONSES.get(

            intent,

            FALLBACK_RESPONSES[
                "general_support"
            ]
        )


        status = "resolved_fallback"


    return {

        "response":
            response,

        "status":
            status
    }


# ============================================================
# GRAPH ROUTING
# ============================================================

def route_after_escalation(
    state: AgentState
):

    if state.get(
        "escalation_required",
        False
    ):

        return "generate_response"


    return "retrieve"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    AgentState
)


graph_builder.add_node(
    "classify_intent",
    classify_intent
)


graph_builder.add_node(
    "classify_escalation",
    classify_escalation
)


graph_builder.add_node(
    "retrieve",
    retrieve_examples
)


graph_builder.add_node(
    "generate_response",
    generate_response
)


# ============================================================
# EDGES
# ============================================================

graph_builder.add_edge(
    START,
    "classify_intent"
)


graph_builder.add_edge(
    "classify_intent",
    "classify_escalation"
)


graph_builder.add_conditional_edges(

    "classify_escalation",

    route_after_escalation,

    {

        "retrieve":
            "retrieve",

        "generate_response":
            "generate_response"
    }
)


graph_builder.add_edge(
    "retrieve",
    "generate_response"
)


graph_builder.add_edge(
    "generate_response",
    END
)


# ============================================================
# COMPILE
# ============================================================

agent = graph_builder.compile()


# ============================================================
# INTERACTIVE DEMO
# ============================================================

print("\n")

print("=" * 72)
print("LANGGRAPH AGENT READY")
print("=" * 72)

print(
    "Type a customer message."
)

print(
    "The agent maintains conversational context."
)

print(
    "Type 'reset' to start a new conversation."
)

print(
    "Type 'exit' to stop."
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []

previous_intent = None

previous_intent_confidence = 0.0


# ============================================================
# INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":

    while True:

        message = input(
            "\nCustomer: "
        ).strip()

        # --------------------------------------------------------
        # EXIT
        # --------------------------------------------------------

        if message.lower() == "exit":

            break

        # --------------------------------------------------------
        # RESET
        # --------------------------------------------------------

        if message.lower() == "reset":

            conversation_history = []

            previous_intent = None

            previous_intent_confidence = 0.0

            print(
                "\nConversation context reset."
            )

            continue

        # --------------------------------------------------------
        # EMPTY MESSAGE
        # --------------------------------------------------------

        if not message:

            continue

        # ========================================================
        # INVOKE AGENT
        # ========================================================

        result = agent.invoke({

            "customer_message":
                message,

            "conversation_history":
                conversation_history,

            "previous_intent":
                previous_intent,

            "previous_intent_confidence":
                previous_intent_confidence
        })

        # ========================================================
        # DISPLAY RESULT
        # ========================================================

        print(
            "\n--- Agent Result ---"
        )

        print(
            f"Intent: "
            f"{result.get('intent')}"
        )

        print(
            f"Intent confidence: "
            f"{result.get('intent_confidence', 0):.2%}"
        )

        print(
            f"Intent source: "
            f"{result.get('intent_source', 'classifier')}"
        )

        print(
            f"Follow-up detected: "
            f"{result.get('is_follow_up', False)}"
        )

        print(
            f"Context similarity: "
            f"{result.get('context_similarity', 0):.4f}"
        )

        print(
            f"Escalation probability: "
            f"{result.get('escalation_probability', 0):.2%}"
        )

        print(
            f"Escalation required: "
            f"{result.get('escalation_required')}"
        )

        print(
            f"Status: "
            f"{result.get('status')}"
        )

        # ========================================================
        # RETRIEVED EXAMPLES
        # ========================================================

        retrieved = result.get(
            "retrieved_examples",
            []
        )

        if retrieved:

            print(
                "\nRetrieved historical examples:"
            )

            for example in retrieved[:3]:

                print(
                    f"  "
                    f"[{example['similarity']:.4f}] "
                    f"{example['intent_hint']}: "
                    f"{str(example['customer_problem'])[:160]}"
                )

        # ========================================================
        # FINAL RESPONSE
        # ========================================================

        print(
            "\nAgent response:"
        )

        response = result.get(
            "response",
            ""
        )

        print(
            response
        )

        # ========================================================
        # UPDATE CONVERSATION MEMORY
        # ========================================================

        conversation_history.append({

            "customer":
                message,

            "agent":
                response
        })

        if len(
            conversation_history
        ) > 6:

            conversation_history = (
                conversation_history[-6:]
            )

        # ========================================================
        # UPDATE PREVIOUS INTENT
        # ========================================================

        previous_intent = result.get(
            "intent"
        )

        previous_intent_confidence = float(
            result.get(
                "intent_confidence",
                0.0
            )
        )