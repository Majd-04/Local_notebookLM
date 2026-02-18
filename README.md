### How to use it

1. **Make the script executable:**
Before you can run a `.sh` file, you have to give it permission.
```bash
chmod +x setup.sh

```
2. **Run the script:**
```bash
./setup.sh

```
# 📚 Local Multi-PDF Conversational RAG System 

* **LangChain** – LLM orchestration framework
* **Ollama** – Local LLM runtime
* **Chroma** – Vector database
* **Llama 3.2** – Language model

---

# 🔹 1. Configuration Section

```python
PDF_FOLDER = "./my_notebook" 
DB_DIR = "./chroma_db"
MODEL_NAME = "llama3.2"
```

* `PDF_FOLDER` → Folder containing your PDFs
* `DB_DIR` → Directory where the vector database is stored
* `MODEL_NAME` → LLM used via Ollama

---

# 🔹 2. Multi-PDF Loading

## Function: `load_all_pdfs(folder_path)`

### Step 1 — Find All PDFs

```python
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
```

Finds all `.pdf` files inside the specified folder.

---

### Step 2 — Load Each PDF

```python
loader = PyPDFLoader(pdf)
all_docs.extend(loader.load())
```

* `PyPDFLoader` loads each PDF
* Converts every page into a `Document` object

Each `Document` contains:

* `page_content`
* `metadata`

---

### Step 3 — Split Into Chunks

```python
RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
```

Why splitting?

LLMs cannot process very large documents efficiently.
So we:

* Split into chunks of **800 characters**
* Keep **100 character overlap** to preserve context continuity

This improves retrieval quality.

---

# 🔹 3. Vector Database Initialization

## Embedding Model

```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

Embeddings convert text → numerical vectors.

This allows similarity search in vector space.

---

## Case 1 — Database Does NOT Exist

```python
Chroma.from_documents(...)
```

Steps:

1. Convert text chunks → embeddings
2. Store embeddings in Chroma
3. Persist database to disk

---

## Case 2 — Database Already Exists

```python
Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
```

Loads the existing vector index (much faster).

---

## Retriever

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

When a query is asked:

* Convert query → embedding
* Retrieve **top 5 most similar chunks**

---

# 🔹 4. Conversational RAG Logic

This system implements **History-Aware Retrieval**.

---

# 🧠 Part A — Question Rewriting

### The Problem

If a user asks:

> "What about its limitations?"

This question depends on previous context.

So we rewrite it into a **standalone question**.

---

## Contextualization Prompt

```python
contextualize_q_prompt = ChatPromptTemplate.from_messages(...)
```

Structure:

* System instructions
* Chat history
* Latest user question

---

## History-Aware Retriever Chain

```python
history_aware_retriever = (
    contextualize_q_prompt 
    | llm 
    | StrOutputParser() 
    | retriever
)
```

Pipeline:

```
User Question
      ↓
LLM rewrites question
      ↓
Standalone Question
      ↓
Retriever
      ↓
Relevant Documents
```

---

# 🧠 Part B — Final Answer Generation

Now we generate the answer using:

* Retrieved context
* Chat history
* Original user question

---

## QA Prompt

```python
qa_prompt = ChatPromptTemplate.from_messages(...)
```

The system message instructs:

* Use only the retrieved context
* If answer not found → say "I don't know"

This helps reduce hallucination.

---

## Formatting Retrieved Documents

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

This converts a list of documents into one combined text block.

---

# 🔥 The Main RAG Chain

```python
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(history_aware_retriever.invoke(x))
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)
```

---

## Step-by-Step Execution

When:

```python
rag_chain.invoke({
    "input": user_input,
    "chat_history": chat_history
})
```

It performs:

### 1️⃣ Retrieve Context

* Rewrites question
* Retrieves top 5 relevant chunks

### 2️⃣ Format Context

* Converts documents into text

### 3️⃣ Inject Into Prompt

* Adds context to QA prompt

### 4️⃣ Send to LLM

* Uses `llama3.2`

### 5️⃣ Parse Output

* Extract plain string response

---

# 🔹 5. Chat Loop

```python
while True:
```

Steps:

1. Receive user input
2. Call `rag_chain`
3. Print response
4. Update chat history
5. Keep only last 10 messages

---

## Why Limit Chat History?

```python
if len(chat_history) > 10:
    chat_history = chat_history[-10:]
```

Prevents:

* Memory overflow
* Large prompts
* Slow inference

---

# 🧠 Overall Architecture

```
User Question
      ↓
Question Rewriter (LLM)
      ↓
Retriever (Chroma)
      ↓
Top 5 Chunks
      ↓
QA Prompt
      ↓
LLM (Llama3.2)
      ↓
Final Answer
```

---

# ✅ System Features

✔ Multi-PDF ingestion
✔ Persistent vector database
✔ History-aware retrieval
✔ Retrieval-Augmented Generation
✔ Conversational memory
✔ Fully local (no external API)

---

# 🎯 Summary

This code implements a **production-style local conversational RAG pipeline** that:

* Loads multiple PDFs
* Splits them into chunks
* Embeds and stores them in Chroma
* Retrieves relevant context
* Rewrites dependent questions
* Generates grounded answers using LLM
