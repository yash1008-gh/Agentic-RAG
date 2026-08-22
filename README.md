# 🤖 Agentic RAG with LangGraph

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph, LangChain, FAISS, Hugging Face Embeddings, and Groq LLMs**.

Unlike a traditional RAG pipeline that blindly retrieves documents and generates an answer, this system uses an **LLM-driven agent to select retrieval tools**, evaluates whether the retrieved documents are relevant, and **rewrites the user's query when retrieval quality is insufficient**.

The result is an iterative RAG workflow that can reason about **when and where to retrieve information** before generating an answer.

---

## 🚀 What Makes This Agentic RAG?

A traditional RAG pipeline generally looks like:

```text
User Query
    ↓
Retriever
    ↓
Relevant Documents
    ↓
LLM
    ↓
Answer
```

This project introduces decision-making and feedback into that process:

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │      Agent       │
                         │ Tool Selection   │
                         └────────┬─────────┘
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
          ┌──────────────────┐        ┌──────────────────┐
          │ LangGraph Search │        │ LangChain Search │
          │      Tool        │        │      Tool        │
          └────────┬─────────┘        └────────┬─────────┘
                   └─────────────┬─────────────┘
                                 ↓
                       ┌──────────────────┐
                       │ Document Grader  │
                       │ Relevant?        │
                       └────────┬─────────┘
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
                 Relevant              Not Relevant
                    ↓                       ↓
             ┌────────────┐          ┌────────────┐
             │  Generate  │          │   Rewrite  │
             │   Answer   │          │   Query    │
             └─────┬──────┘          └─────┬──────┘
                   ↓                       │
                  END                      └────→ Agent
```

The agent can therefore:

* Decide whether retrieval is required.
* Select between multiple retrieval tools.
* Retrieve information from the appropriate knowledge base.
* Evaluate the relevance of retrieved documents.
* Rewrite poorly formulated queries.
* Retry retrieval using the improved query.
* Generate an answer only after obtaining relevant context.

---

# 🧠 Architecture

The system consists of four major stages.

### 1. Knowledge Base Construction

Documentation from LangGraph and LangChain is loaded from their official documentation websites.

```text
Documentation URLs
       ↓
WebBaseLoader
       ↓
Document Objects
       ↓
RecursiveCharacterTextSplitter
       ↓
Text Chunks
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Store
```

Two independent vector stores are created:

```text
LangGraph Documentation
        ↓
LangGraph FAISS Index

LangChain Documentation
        ↓
LangChain FAISS Index
```

This allows the agent to treat each knowledge source as an independent retrieval tool.

---

# 🔎 2. Embeddings and Vector Search

The project uses:

```text
BAAI/bge-small-en-v1.5
```

to convert document chunks into dense vector representations.

For example:

```text
"What is LangGraph?"
        ↓
Embedding Model
        ↓
[0.12, -0.31, 0.87, ...]
```

During retrieval, the user query is embedded using the same model and compared against the stored vectors.

FAISS then returns the most semantically similar document chunks.

---

# 🛠️ 3. Retrieval Tools

Each vector store is exposed to the agent as a tool using LangChain's retriever tool abstraction.

### LangGraph Retriever

```text
retriever_vector_db_blog
```

Searches the LangGraph documentation vector store.

### LangChain Retriever

```text
retriever_vector_langchain_blog
```

Searches the LangChain documentation vector store.

The tools are provided to the LLM:

```python
tools = [
    retriever_tool_langgraph,
    retriever_tool_langchain
]
```

The agent can then decide which tool is appropriate for the user's question.

---

# 🧠 4. Agentic Decision Making

The agent is implemented using a Groq-hosted LLM:

```text
openai/gpt-oss-120b
```

The model is given access to the retrieval tools through:

```python
model.bind_tools(tools)
```

This allows the model to produce either:

```text
Normal response
```

or:

```text
Tool call
```

For example:

```text
User:
"What is LangGraph?"

Agent:
→ Call LangGraph Retriever
```

Whereas:

```text
User:
"What are LangChain guardrails?"

Agent:
→ Call LangChain Retriever
```

The LLM therefore performs **tool selection rather than relying on a hard-coded router**.

---

# ✅ Document Relevance Grading

Retrieval similarity alone does not guarantee that the retrieved documents actually answer the question.

Therefore, the project introduces a separate **document grading step**.

After retrieval, another LLM call evaluates:

```text
Question
+
Retrieved Context
        ↓
Relevance Grader
        ↓
"yes" / "no"
```

The grader uses structured output with a Pydantic schema:

```python
class Grade(BaseModel):
    binary_score: Literal["yes", "no"]
```

This makes the LLM's output machine-readable and allows the result to control the LangGraph workflow.

---

# 🔄 Query Rewriting

If the retrieved documents are judged irrelevant:

```text
Retrieved Documents
        ↓
     Grader
        ↓
       NO
        ↓
   Query Rewrite
```

The LLM reformulates the original question to improve retrieval.

For example:

```text
Original:
"What is Send API?"

Rewritten:
"How does LangGraph's Send API support dynamic
task distribution and map-reduce workflows?"
```

The rewritten query is then sent back to the agent:

```text
Rewrite
   ↓
Agent
   ↓
Retriever
   ↓
Grader
```

This creates a feedback loop instead of accepting poor retrieval results immediately.

---

# 🕸️ LangGraph Workflow

The complete workflow is implemented as a LangGraph state machine.

### State

The graph maintains message history using:

```python
class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]
```

The state is shared between graph nodes and allows the workflow to maintain information about:

* User questions
* Agent decisions
* Tool calls
* Retrieved documents
* Rewritten queries
* Generated responses

---

## Graph Nodes

### `agent`

Determines whether retrieval is necessary and selects an available retrieval tool.

### `retrieve`

Executes the selected retriever through LangGraph's `ToolNode`.

### `grade_documents`

Evaluates whether retrieved documents are relevant to the user's question.

### `rewrite`

Reformulates the query when retrieval quality is insufficient.

### `generate`

Uses the retrieved context to generate the final answer.

---

# 🔀 Workflow Routing

The graph uses conditional edges to control execution.

After the agent:

```text
Agent
  │
  ├── Tool call → Retrieve
  │
  └── No tool call → END
```

After retrieval:

```text
Retrieve
   │
   ├── Relevant → Generate
   │
   └── Not Relevant → Rewrite
```

After rewriting:

```text
Rewrite
   ↓
Agent
```

Therefore, the workflow can iterate:

```text
Agent
  ↓
Retrieve
  ↓
Grade
  ↓
Rewrite
  ↓
Agent
  ↓
Retrieve
  ↓
Grade
  ↓
Generate
  ↓
END
```

---

# 📦 Tech Stack

| Component              | Technology                     |
| ---------------------- | ------------------------------ |
| Workflow Orchestration | LangGraph                      |
| LLM Framework          | LangChain                      |
| LLM                    | Groq / `openai/gpt-oss-120b`   |
| Embeddings             | `BAAI/bge-small-en-v1.5`       |
| Vector Database        | FAISS                          |
| Document Loading       | WebBaseLoader                  |
| Text Splitting         | RecursiveCharacterTextSplitter |
| Structured Output      | Pydantic                       |
| Observability          | LangSmith                      |
| Language               | Python                         |

---

# 📁 Project Structure

A recommended project structure:

```text
agentic-rag/
│
├── app/
│   ├── graph.py
│   ├── state.py
│   ├── nodes/
│   │   ├── agent.py
│   │   ├── retrieve.py
│   │   ├── grader.py
│   │   ├── rewrite.py
│   │   └── generate.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vectorstores.py
│   │   └── tools.py
│   │
│   └── config.py
│
├── notebooks/
│   └── agentic_rag.ipynb
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd agentic-rag
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

Do not commit your `.env` file.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

---

# ▶️ Running the Project

Once the environment is configured, initialize the vector stores and compile the LangGraph workflow.

Example query:

```python
result = graph.invoke({
    "messages": "What is LangGraph? Explain it using the documentation."
})
```

Another example:

```python
result = graph.invoke({
    "messages": "What are guardrails in LangChain?"
})
```

The agent should identify the appropriate knowledge source and retrieve relevant documentation.

---

# 🧪 Example Queries

### LangGraph

```text
What is LangGraph?
```

Expected routing:

```text
Agent
 ↓
LangGraph Retriever
 ↓
Relevant
 ↓
Generate
```

### LangChain

```text
What are LangChain guardrails?
```

Expected routing:

```text
Agent
 ↓
LangChain Retriever
 ↓
Relevant
 ↓
Generate
```

### Poor Retrieval

```text
What is machine learning?
```

Since the knowledge bases primarily contain LangChain and LangGraph documentation, the retrieved context may not be relevant.

This can trigger:

```text
Retrieve
 ↓
Grade
 ↓
Not Relevant
 ↓
Rewrite
 ↓
Agent
```

This demonstrates the system's retrieval-quality feedback loop.

---

# 🔬 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Agentic RAG
* LangGraph state machines
* LLM tool calling
* Multiple retrieval tools
* Vector databases
* Dense embeddings
* Semantic search
* Structured LLM output
* Pydantic validation
* Conditional graph routing
* Query rewriting
* Retrieval relevance grading
* Iterative retrieval
* LangChain Runnable chains

---

# 🆚 Traditional RAG vs Agentic RAG

| Feature                    | Traditional RAG | This Project |
| -------------------------- | --------------- | ------------ |
| Vector retrieval           | ✅               | ✅            |
| Embeddings                 | ✅               | ✅            |
| LLM generation             | ✅               | ✅            |
| Multiple retrieval sources | Optional        | ✅            |
| Tool selection             | ❌               | ✅            |
| Retrieval grading          | Usually ❌       | ✅            |
| Query rewriting            | Usually ❌       | ✅            |
| Iterative retrieval        | Usually ❌       | ✅            |
| Conditional workflow       | Limited         | ✅            |
| Agentic behavior           | ❌               | ✅            |

---

# ⚠️ Current Limitations

This implementation is primarily an educational/experimental Agentic RAG system and has several areas that can be improved for production use.

### 1. Retry limits

The current workflow can potentially loop between:

```text
Agent → Retrieve → Grade → Rewrite → Agent
```

if retrieval repeatedly fails.

A production implementation should maintain a retry counter and enforce a maximum number of iterations.

### 2. Query state management

The current implementation primarily uses the message history to determine the question.

A cleaner implementation would maintain the active query separately from the conversation history.

### 3. Retrieval evaluation

Embedding and retrieval quality should be evaluated quantitatively using metrics such as:

* Recall@K
* MRR
* NDCG

rather than relying only on qualitative inspection.

### 4. Reranking

A dedicated reranker/cross-encoder could be added after initial retrieval:

```text
Query
 ↓
Vector Search
 ↓
Top 20
 ↓
Reranker
 ↓
Top 5
 ↓
LLM
```

### 5. Hybrid retrieval

Dense vector search could be combined with sparse retrieval such as BM25 to improve retrieval of exact technical terms, identifiers, and API names.

---

# 🚀 Future Improvements

Potential improvements include:

* Add a maximum retry/iteration limit.
* Add hybrid BM25 + dense retrieval.
* Add a cross-encoder reranker.
* Benchmark multiple embedding models.
* Add retrieval evaluation with Recall@K and MRR.
* Add document metadata filtering.
* Add source citations to generated answers.
* Add streaming responses.
* Add persistent vector stores.
* Add conversational memory.
* Add hallucination/answer-quality grading.
* Add LangSmith tracing and evaluation.
* Deploy using FastAPI and Streamlit.
* Add automated evaluation datasets.
* Add Docker deployment.

---

# 🧩 Core Learning

The main idea behind this project is that **retrieval itself should be treated as a decision-making process**.

Instead of assuming:

```text
"Retrieve once and trust the results."
```

the system asks:

```text
"Did I retrieve the right information?"
```

If the answer is no:

```text
"Can I formulate the query better?"
```

Then it retries.

This transforms a fixed RAG pipeline into a **stateful, iterative workflow controlled by an LLM and orchestrated by LangGraph**.

---

# 📜 License

This project is intended for educational and research purposes. Add the license appropriate for your repository here, such as MIT.

---

# 👨‍💻 Author

**Yash Vardhan Singh**

Built as a hands-on exploration of:

**LLMs → RAG → Tool Calling → Agents → LangGraph → Agentic RAG**
