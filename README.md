# 🤖 AmazonHelp AI Support Agent

> An end-to-end AI customer support agent builted using the **Customer Support on Twitter** dataset.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)](https://github.com/facebookresearch/faiss)

---

## 🎯 Overview

This project implements an AI-powered support system for **AmazonHelp** that processes customer messages and generates grounded responses using historical customer-support conversations.

The system combines:

- Intent classification
- Escalation detection
- Semantic retrieval with FAISS
- Historical support examples
- Context-aware follow-up handling
- LLM-based response generation
- Deterministic safe handoffs for high-risk cases
- Quantitative evaluation

The complete workflow is orchestrated using **LangGraph** and exposed through a **FastAPI backend** with a **React frontend**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏷️ Intent Classification | Classifies customer messages into 11 support intents |
| 🔎 Semantic Retrieval | Retrieves similar historical support conversations using FAISS |
| 💬 Response Generation | Generates grounded customer-facing responses using an LLM |
| 🚨 Escalation Detection | Identifies conversations that should be handled by human support |
| 🧠 Conversation Context | Maintains context across follow-up messages |
| 🛡️ Safe Handoff | High-risk cases bypass automated response generation |
| 📊 Evaluation | Includes intent, retrieval and escalation evaluation artifacts |
| 🔌 Multiple LLM Providers | Supports local Ollama and Groq/OpenAI-compatible APIs |

---

## 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │   Customer Message   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Intent Classification│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Escalation Detection │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                  Escalation                 Safe to automate
                       │                         │
                       ▼                         ▼
             ┌──────────────────┐      ┌──────────────────┐
             │ Safe Human       │      │ FAISS Retrieval  │
             │ Handoff          │      └────────┬─────────┘
             └──────────────────┘               │
                                                ▼
                                     ┌─────────────────────┐
                                     │ Historical Support  │
                                     │ Context             │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │ LLM Response        │
                                     │ Generation           │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │ Customer Response    │
                                     └─────────────────────┘
```

### LangGraph Flow

```text
classify_intent
       ↓
classify_escalation
       ↓
   ┌───┴────┐
   │        │
ESCALATE   SAFE
   │        │
   ↓        ↓
HANDOFF  retrieve
            ↓
       generate_response
```

---

## 🎯 Supported Intents

The system uses 11 manually defined support categories:

| # | Intent |
|---:|---|
| 1 | `delivery_shipping` |
| 2 | `payment` |
| 3 | `refund` |
| 4 | `return_replacement` |
| 5 | `product_issue` |
| 6 | `account_access` |
| 7 | `prime_membership` |
| 8 | `technical_app` |
| 9 | `order_cancellation` |
| 10 | `security_fraud` |
| 11 | `general_support` |

---

## 📊 Dataset

The project uses the public **Customer Support on Twitter** dataset and focuses on the **AmazonHelp** brand.

### Dataset Processing

| Metric | Result |
|---|---:|
| AmazonHelp conversations | 100,503 |
| Usable conversations | 97,809 |
| Multi-turn conversations | 43,406 |
| RAG conversations | 5,000 |
| Golden evaluation examples | 200 |

The raw dataset is intentionally excluded from the repository because of its large file size.

The repository instead contains the processed artifacts required to run the application and reproduce the evaluation workflow.

---

## 🔎 Retrieval System

The RAG pipeline uses:

- `all-MiniLM-L6-v2` for sentence embeddings
- FAISS `IndexFlatIP`
- Normalized embeddings for cosine similarity
- 5,000 curated historical support conversations

The 200 golden evaluation examples are explicitly excluded from the RAG corpus to prevent evaluation leakage.

### Retrieval Evaluation

Evaluated on 200 manually verified golden examples.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **48.00%** |
| Recall@3 | **71.50%** |
| Recall@5 | **77.00%** |
| Recall@10 | **86.00%** |
| MRR | **0.6128** |
| Mean Latency | **~24.6 ms** |

These retrieval metrics use intent consistency as an evaluation proxy rather than claiming perfect human relevance.

---

## 🏷️ Intent Classification Evaluation

The intent classifier uses:

- TF-IDF word n-grams
- Logistic Regression
- 11 support categories

Evaluated on 200 manually verified golden examples.

| Metric | Score |
|---|---:|
| Accuracy | **74.00%** |
| Macro F1 | **0.7311** |
| Weighted F1 | **0.7314** |

---

## 🚨 Escalation Detection

A dedicated escalation classifier is used before response generation.

The system intentionally prioritizes catching risky conversations rather than allowing the LLM to respond automatically in uncertain/high-risk situations.

5-fold stratified out-of-fold evaluation:

| Metric | Score |
|---|---:|
| Accuracy | **71.50%** |
| Precision | **61.74%** |
| Recall | **84.52%** |
| F1 | **71.36%** |

The selected threshold is **0.35**, chosen to provide stronger recall for escalation-sensitive cases.

---

## 💬 Context-Aware Conversations

The agent supports follow-up messages by maintaining recent conversation context.

Example:

```text
Customer:
Where is my package?

Agent:
I can help with your delivery issue...

Customer:
It still hasn't arrived.

Agent:
I understand. Since this is a follow-up to your delivery
issue, let me help you with the next appropriate step.
```

The system combines:

- Recent conversation history
- Previous intent
- Intent confidence
- Semantic similarity
- Current customer message

to determine whether a message is a follow-up.

---

## 🛡️ Safety & Grounding

The response generator is instructed not to invent information or claim actions that were not actually performed.

The agent avoids fabricating:

- Order status
- Tracking information
- Refund confirmation
- Cancellation confirmation
- Account information
- Actions supposedly performed by support

It also avoids requesting unnecessary sensitive information such as:

- Passwords
- OTPs
- PINs
- CVV/card details
- Unrelated personal information

High-risk conversations are routed to a deterministic human-support handoff instead of automated generation.

---

## 🛠️ Tech Stack

### Backend & AI

- Python
- FastAPI
- LangGraph
- Scikit-learn
- Sentence Transformers
- FAISS
- Logistic Regression
- Joblib

### LLM

- Ollama
- Llama 3.2 3B
- Groq / OpenAI-compatible API support

### Frontend

- React
- Vite
- JavaScript
- CSS

---

## 📁 Project Structure

```text
AI-Support-Agent/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 .gitignore
│
├── 🧠 AI Pipeline
│   ├── 01_extract_amazonhelp.py
│   ├── 02_analyze_amazonhelp.py
│   ├── 02_analyze_amazonhelp_v2.py
│   ├── 03_discover_amazonhelp_intents.py
│   ├── 04_create_golden_candidates.py
│   ├── 05_create_rag_corpus.py
│   ├── 06_build_vector_store.py
│   ├── 07_retrieval_evaluation.py
│   ├── 08_train_intent_classifier.py
│   ├── 09_response_generator.py
│   ├── 10_escalation_classifier.py
│   ├── 11_langgraph_agent.py
│   └── 12_test_llm_provider.py
│
├── 🔧 Supporting Modules
│   ├── escalation_features.py
│   ├── llm_provider.py
│   ├── conversation_reconstruction.py
│   ├── profile_dataset.py
│   ├── brand_analysis.py
│   └── conversation_analysis.py
│
├── 🖥️ backend/
│   └── app/
│       └── main.py
│
├── 🎨 frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
└── 📦 data/
    │
    ├── evaluation/
    │   ├── retrieval_evaluation_results.csv
    │   └── retrieval_evaluation_summary.json
    │
    ├── models/
    │   ├── intent_classifier/
    │   │   ├── amazonhelp_intent_classifier.joblib
    │   │   ├── confusion_matrix.csv
    │   │   ├── golden_intent_predictions.csv
    │   │   └── intent_classifier_metrics.json
    │   │
    │   └── escalation_classifier/
    │       ├── amazonhelp_escalation_classifier.joblib
    │       ├── char_vectorizer.joblib
    │       └── word_vectorizer.joblib
    │
    ├── processed/
    │   ├── amazonhelp_data_profile.json
    │   ├── amazonhelp_golden_final.csv
    │   ├── amazonhelp_rag_corpus.csv
    │   ├── amazonhelp_rag_corpus.json
    │   ├── amazonhelp_rag_corpus_summary.json
    │   └── escalation/
    │       ├── escalation_classifier_metrics.json
    │       ├── escalation_classifier_oof_predictions.csv
    │       ├── escalation_classifier_policy.json
    │       └── escalation_classifier_thresholds.csv
    │
    └── vectorstore/
        └── amazonhelp_faiss/
            ├── index.faiss
            ├── metadata.csv
            └── config.json
```

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/DecodeX7/AI-Support-Agent.git
cd AI-Support-Agent
```

### 2. Create a Python virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

---

## 🤖 Configure the LLM

Create a `.env` file from `.env.example`.

### Option 1 — Ollama

The default configuration uses Ollama.

Install the required model:

```bash
ollama pull llama3.2:3b
```

Example:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
```

Make sure Ollama is running before starting the backend.

### Option 2 — Groq

Add your Groq API key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The actual `.env` file is excluded from Git.

---

## ▶️ Run the Application

### Start Backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### Start Frontend

Open another terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL displayed in the terminal.

---

## 🧪 Example Test Cases

### Delivery

```text
Where is my package? It was supposed to arrive yesterday.
```

Expected:

```text
Intent: delivery_shipping
```

### Payment

```text
My payment failed while placing an order.
```

Expected:

```text
Intent: payment
```

### Follow-up

```text
Where is my package?

It still hasn't arrived.
```

The second message should retain delivery context.

### Security / Fraud

```text
Someone used my account without my permission.
```

Expected behavior:

```text
Escalation → Safe Human Handoff
```

---

## 🎥 Demo Video

A complete walkthrough of the application is available here:

**[▶️ Watch the Project Demo](https://drive.google.com/drive/folders/1xDOxW7uccKuaWyvneRbG03uluMdXPMyM?usp=drive_link)**

The demo covers:

- Application UI
- Intent classification
- Historical retrieval
- Response generation
- Follow-up context
- Escalation handling

---

## 📦 Included Artifacts

The repository includes the important processed and runtime artifacts required by the application:

- ✅ Golden evaluation set
- ✅ RAG corpus
- ✅ RAG corpus summary
- ✅ FAISS vector database
- ✅ Embedding metadata
- ✅ Intent classifier
- ✅ Escalation classifier
- ✅ Escalation policy
- ✅ Retrieval evaluation results
- ✅ Intent evaluation results
- ✅ Dataset profile

The original raw dataset and large intermediate files are intentionally excluded.

---

## 🔐 Security

No API keys or credentials are committed to the repository.

The `.env` file is excluded using `.gitignore`.

Only `.env.example` is included as a configuration template.

---

## 🔮 Future Improvements

Possible future improvements include:

- Larger manually verified evaluation datasets
- Fine-tuned transformer-based intent classification
- Intent-aware retrieval ranking
- Better multi-intent handling
- LLM-as-a-judge response evaluation
- Production monitoring and feedback loops
- Improved escalation calibration
- Streaming responses
- Persistent conversation storage

---

## 👨‍💻 Project

**AmazonHelp AI Support Agent**


**Repository:**  
https://github.com/DecodeX7/AI-Support-Agent

---

---

# 📋 Assignment Report

## 1. Problem Framing

### What does "good" mean for AmazonHelp?

A good support agent should:

1. Correctly understand the customer's issue.
2. Identify the appropriate support intent.
3. Use historically similar AmazonHelp conversations as evidence.
4. Generate a concise and useful response without inventing account or order information.
5. Preserve context across follow-up messages.
6. Escalate risky or sensitive cases instead of attempting unsafe automation.

The goal is therefore not simply fluent text generation. The system should be **useful, grounded, context-aware, and appropriately cautious**.

### What I chose not to build

To keep the project focused and reproducible, I intentionally did not build:

- Full order-management integration
- Real Amazon account access
- Real refund/cancellation execution
- Real-time shipment tracking
- Production authentication
- Human-agent ticketing infrastructure
- A fully fine-tuned large language model
- Full-dataset production-scale processing

The focus is the core support-agent workflow: **classification, retrieval, grounded response generation, context handling, and escalation**.

---

## 2. Golden Evaluation Set

I created a manually verified golden evaluation set containing **200 examples**, within the required 150–250 example range.

Each example was reviewed and assigned:

- Intent
- Escalation requirement
- Escalation reason
- Expected action
- Resolution criteria

The final golden set is:

```text
data/processed/amazonhelp_golden_final.csv
```

The 200 golden examples were explicitly excluded from the RAG corpus to reduce evaluation leakage.

---

## 3. Results vs Baselines

### Baseline 1 — Majority-Class Classifier

A majority-class classifier provides a trivial lower-bound baseline for the imbalanced intent distribution.

It predicts the most frequent class for every input, requiring no understanding of the customer message.

The final classifier is reported using both Accuracy and Macro F1 so that performance across all 11 intents is visible.

### Baseline 2 — TF-IDF + Logistic Regression

The implemented intent classifier is a deliberately simple classical ML baseline:

```text
Customer Message
       ↓
TF-IDF word n-grams
       ↓
Logistic Regression
       ↓
11 Intent Classes
```

It was selected because it is fast, interpretable, reproducible, and provides a useful reference before introducing more complex models.

The final system adds retrieval, context handling, LLM generation, and escalation around this classifier.

---

## 4. Failure Analysis — Top 5 Failure Modes

### 1. Ambiguous customer messages

Example:

```text
"Can you help me with this?"
```

**Hypothesis:** There is insufficient information to confidently identify the issue.

**Improvement:** Add an explicit insufficient-context route and use conversation history more aggressively.

### 2. Security and account-access overlap

Example:

```text
"I can't access my account and I think someone changed something."
```

**Hypothesis:** The message contains both account-access and security signals.

**Improvement:** Add multi-intent detection and prioritize security signals during escalation routing.

### 3. Product and technical issues overlap

Example:

```text
"The Amazon app isn't working when I try to use my device."
```

**Hypothesis:** Real customer messages often mix product, application, and device context.

**Improvement:** Use hierarchical or multi-label classification.

### 4. Top-1 retrieval ranking is imperfect

Top-1 retrieval accuracy is **48%**, while Recall@10 reaches **86%**.

This indicates that relevant historical examples are often present among the retrieved results but are not always ranked first.

**Hypothesis:** Pure embedding similarity does not always capture the exact support intent or resolution stage.

**Improvement:** Combine semantic similarity, intent probability, keyword matching, and reranking.

### 5. Low-confidence intent predictions

The classifier has a mean confidence of approximately **0.44**, and many golden examples fall below high-confidence thresholds.

**Hypothesis:** The 11 intents contain overlapping language and the source conversations are noisy.

**Improvement:** Calibrate probabilities and introduce an uncertainty/clarification route rather than forcing every message into a confident-looking category.

---

# ⚠️ What Is Misleading About My Headline Number?

The most attractive headline number is:

> **74% intent classification accuracy**

However, this does **not** mean that 74% of customer conversations will be solved correctly.

### Accuracy hides class imbalance

Some intents contain more examples than others. A model can achieve reasonable accuracy while performing poorly on smaller or harder classes.

That is why Macro F1 (**0.7311**) is also reported.

### Intent accuracy is not response quality

Correctly predicting `refund` does not guarantee that the generated response is correct.

The response could still:

- Give an inappropriate resolution
- Miss important context
- Invent information
- Fail to escalate a risky case

### Retrieval metrics are proxies

Retrieval evaluation measures intent consistency rather than complete human judgment of whether a retrieved example is the best possible historical support example.

### Escalation has asymmetric costs

Missing a risky conversation can be more harmful than unnecessarily escalating a normal conversation.

Therefore, escalation recall is particularly important.

### Bottom line

The **74% accuracy figure is one component of system quality**, not proof that 74% of conversations will be resolved correctly.

A more meaningful view is:

```text
Intent Classification
        +
Historical Retrieval
        +
Grounded Generation
        +
Escalation
        +
Context Handling
```

The system should therefore be evaluated as a pipeline rather than by a single headline metric.

---

## 5. Response Evaluation

The current automated evaluation focuses on:

- Intent classification metrics
- Retrieval metrics
- Escalation metrics
- Golden-set evaluation
- Retrieval latency

The response-quality rubric is based on:

| Criterion | Evaluation Question |
|---|---|
| Grounding | Is the response supported by retrieved historical context? |
| Correctness | Does it address the customer's actual issue? |
| Helpfulness | Does it provide a useful next step? |
| Safety | Does it avoid invented actions or sensitive information? |
| Relevance | Is it concise and directly related to the issue? |
| Escalation | Should this conversation have been handed to a human? |

> **Current limitation:** A completed judge-vs-human agreement experiment is not claimed here because it was not part of the implemented evaluation run. This is an explicit area for improvement rather than an invented metric.

---

## 6. Decision Log

### 1. Selected AmazonHelp
Selected one brand to keep the problem focused and allow deeper analysis.

### 2. Defined 11 intents
Kept the intent set small enough to be operationally useful while covering the major support patterns.

### 3. Created a manually verified golden set
Used human-reviewed examples rather than relying entirely on automatically generated labels.

### 4. Excluded golden examples from RAG
Reduced evaluation leakage by keeping evaluation examples outside the retrieval corpus.

### 5. Used 5,000 RAG documents
Used a curated subset to keep retrieval fast and reproducible.

### 6. Used Sentence Transformers
Used `all-MiniLM-L6-v2` for lightweight semantic embeddings.

### 7. Used FAISS
Used local vector search without requiring an external vector database.

### 8. Used TF-IDF + Logistic Regression
Provided a transparent and strong classical baseline.

### 9. Added a dedicated escalation classifier
Separated escalation from response generation.

### 10. Escalation happens before generation
High-risk messages bypass normal automated generation.

### 11. Added follow-up context handling
Used recent conversation history and previous intent for short follow-ups.

### 12. Added an LLM provider abstraction
Supported both local Ollama and Groq/OpenAI-compatible APIs.

### 13. Used deterministic handoffs
Used controlled human-support responses for high-risk cases.

### 14. Excluded the raw dataset from Git
Avoided committing the very large original dataset.

### 15. Kept runtime artifacts in the repository
Included trained models, FAISS index, metadata, RAG corpus, golden set, and evaluation artifacts.

---

## 7. What I Would Do With One More Week

I would focus on reliability rather than simply adding more features.

### Better intent classification
- Review misclassified golden examples
- Add difficult training examples
- Try a sentence-transformer classifier
- Calibrate confidence scores

### Better retrieval
Introduce hybrid retrieval:

```text
Semantic Similarity
        +
Intent Probability
        +
Keyword Matching
        ↓
Reranking
```

### Better response evaluation
Build a larger response-quality benchmark with:

- LLM-as-judge
- Human review
- Judge-vs-human agreement analysis

### Better escalation calibration
Review false positives and false negatives and tune the threshold according to business cost.

### Robustness testing
Test:

- Ambiguous messages
- Very short messages
- Multi-intent messages
- Follow-ups
- Adversarial requests
- Sensitive-information requests

### Production hardening
Add:

- Persistent conversation storage
- Structured logging
- Monitoring
- API authentication
- More comprehensive automated tests

---

## 8. Reproducibility

The repository contains the processed artifacts required to reproduce the main evaluation workflow.

Important artifacts include:

```text
data/
├── processed/
│   ├── amazonhelp_data_profile.json
│   ├── amazonhelp_golden_final.csv
│   ├── amazonhelp_rag_corpus.csv
│   ├── amazonhelp_rag_corpus.json
│   └── amazonhelp_rag_corpus_summary.json
│
├── models/
│   ├── intent_classifier/
│   └── escalation_classifier/
│
├── evaluation/
│   ├── retrieval_evaluation_results.csv
│   └── retrieval_evaluation_summary.json
│
└── vectorstore/
    └── amazonhelp_faiss/
        ├── index.faiss
        ├── metadata.csv
        └── config.json
```

The original multi-million-row raw dataset is not committed because of its size.

---

# 🎥 Demo Video

A complete walkthrough of the working application is available here:

**[▶️ Watch the Project Demo](https://drive.google.com/drive/folders/1xDOxW7uccKuaWyvneRbG03uluMdXPMyM?usp=drive_link)**

The demo covers:

- Application UI
- Intent classification
- Historical retrieval
- Response generation
- Follow-up context
- Escalation handling

---

# 🔐 Security

No API keys or credentials are committed to the repository.

The `.env` file is excluded using `.gitignore`.

Only `.env.example` is included as a configuration template.

---

# 🔮 Future Improvements

Possible future improvements include:

- Larger manually verified evaluation datasets
- Fine-tuned transformer-based intent classification
- Intent-aware retrieval ranking
- Better multi-intent handling
- LLM-as-a-judge response evaluation with human agreement analysis
- Production monitoring and feedback loops
- Improved escalation calibration
- Streaming responses
- Persistent conversation storage

---

# 👨‍💻 Project

**AmazonHelp AI Support Agent**

Built for the **Hiver SDE Intern Take-Home Assignment**.

### Repository

**https://github.com/DecodeX7/AI-Support-Agent**

### Demo

**https://drive.google.com/drive/folders/1xDOxW7uccKuaWyvneRbG03uluMdXPMyM?usp=drive_link**

---

> **The proof matters more than the system.**
