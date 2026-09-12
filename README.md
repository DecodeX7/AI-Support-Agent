\# AmazonHelp AI Support Agent



An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.



\## Overview



This project builds an end-to-end support system for AmazonHelp that can:



\- Classify incoming customer messages into 11 support intents

\- Retrieve relevant historical support conversations using semantic search

\- Generate grounded customer-facing responses

\- Detect conversations that should be escalated

\- Maintain context across follow-up messages

\- Provide deterministic safe handoffs for high-risk cases

\- Evaluate intent classification and retrieval quality



\## Supported Intents



The system uses 11 intents:



1\. delivery\_shipping

2\. payment

3\. refund

4\. return\_replacement

5\. product\_issue

6\. account\_access

7\. prime\_membership

8\. technical\_app

9\. order\_cancellation

10\. security\_fraud

11\. general\_support



\## Architecture



Customer Message

&#x20;      |

&#x20;      v

Intent Classification

&#x20;      |

&#x20;      v

Escalation Classification

&#x20;      |

&#x20;      +---- Escalation ----> Safe Human Handoff

&#x20;      |

&#x20;      v

FAISS Semantic Retrieval

&#x20;      |

&#x20;      v

Historical Support Context

&#x20;      |

&#x20;      v

LLM Response Generation

&#x20;      |

&#x20;      v

Customer Response



The application is implemented using LangGraph for orchestration.



\## Dataset



The project uses the public Customer Support on Twitter dataset.



Brand selected:



AmazonHelp



Dataset processing produced:



\- 100,503 AmazonHelp conversations

\- 97,809 usable conversations

\- 43,406 multi-turn conversations

\- 5,000 high-quality historical conversations used for RAG

\- 200 manually verified golden examples for evaluation



The raw dataset is intentionally not included in this repository because of its large size.



\## Evaluation



\### Intent Classification



Evaluated on 200 manually verified golden examples.



\- Accuracy: 74.00%

\- Macro F1: 0.7311

\- Weighted F1: 0.7314



\### Retrieval



Evaluated against the 200-example golden set.



\- Top-1 accuracy: 48.00%

\- Recall@3: 71.50%

\- Recall@5: 77.00%

\- Recall@10: 86.00%

\- MRR: 0.6128

\- Mean retrieval latency: \~24.6 ms



\### Escalation



5-fold stratified out-of-fold evaluation was used.



At the selected threshold of 0.35:



\- Accuracy: 71.50%

\- Precision: 61.74%

\- Recall: 84.52%

\- F1: 71.36%



The threshold was selected to prioritize balanced escalation performance, with emphasis on catching conversations requiring human intervention.



\## Tech Stack



\- Python

\- FastAPI

\- React + Vite

\- LangGraph

\- FAISS

\- Sentence Transformers

\- Scikit-learn

\- Logistic Regression

\- Ollama

\- Llama 3.2 3B

\- Groq / OpenAI-compatible API support



\## Project Structure



```text

AI Support Agent/

│

├── 01\_extract\_amazonhelp.py

├── 02\_analyze\_amazonhelp.py

├── 03\_discover\_amazonhelp\_intents.py

├── 04\_create\_golden\_candidates.py

├── 05\_create\_rag\_corpus.py

├── 06\_build\_vector\_store.py

├── 07\_retrieval\_evaluation.py

├── 08\_train\_intent\_classifier.py

├── 09\_response\_generator.py

├── 10\_escalation\_classifier.py

├── 11\_langgraph\_agent.py

├── 12\_test\_llm\_provider.py

│

├── escalation\_features.py

├── llm\_provider.py

├── conversation\_reconstruction.py

├── profile\_dataset.py

├── brand\_analysis.py

├── conversation\_analysis.py

│

├── backend/

│   └── app/

│       └── main.py

│

├── frontend/

│

├── data/

│   ├── processed/

│   ├── models/

│   └── vectorstore/

│

├── .env.example

├── .gitignore

└── requirements.txt



Setup

1\. Clone the repository

git clone <repository-url>

cd AI-Support-Agent

2\. Create virtual environment

python -m venv .venv



Windows:



.venv\\Scripts\\activate

3\. Install dependencies

pip install -r requirements.txt

4\. Configure environment variables



Create a .env file using .env.example.



Example:



LLM\_PROVIDER=ollama

OLLAMA\_MODEL=llama3.2:3b

OLLAMA\_BASE\_URL=http://localhost:11434



GROQ\_API\_KEY=

GROQ\_MODEL=openai/gpt-oss-20b



The actual .env file is intentionally excluded from Git.



Local LLM



The default configuration uses Ollama.



Install Ollama and pull the model:



ollama pull llama3.2:3b



Make sure Ollama is running before starting the backend.



The application also supports Groq through an OpenAI-compatible API.



Run Backend



From the project root:



uvicorn backend.app.main:app --reload



Backend:



http://127.0.0.1:8000

Run Frontend



Open another terminal:



cd frontend

npm install

npm run dev



Open the Vite URL shown in the terminal.

