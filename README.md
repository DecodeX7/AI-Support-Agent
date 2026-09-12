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
