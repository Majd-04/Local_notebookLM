import os  # Provides functions for interacting with the operating system (like checking if folders exist)
import glob  # Used to retrieve files matching a specified pattern (like finding all .pdf files)
from langchain_community.document_loaders import PyPDFLoader  # Tool to load and extract text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits long text into smaller, manageable chunks
from langchain_ollama import OllamaEmbeddings, ChatOllama  # Interface for Ollama models to create embeddings and chat
from langchain_chroma import Chroma  # A vector database used to store and search through document embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Helpers to build structured AI prompts
from langchain_core.output_parsers import StrOutputParser  # Simplifies the AI output by extracting only the text string
from langchain_core.runnables import RunnablePassthrough  # Allows passing data through a chain without modification
from langchain_core.messages import HumanMessage, AIMessage  # Standard classes for representing chat history messages

# --- 1. CONFIGURATION ---
# Put all your PDFs in a folder named 'my_notebook'
PDF_FOLDER = "./my_notebook" 
DB_DIR = "./chroma_db"
MODEL_NAME = "llama3.2"

# --- 2. MULTI-PDF LOADING ---
def load_all_pdfs(folder_path):
    all_docs = []
    # Find all file paths ending in .pdf within the specified folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {folder_path}!")
        return []

    for pdf in pdf_files:
        print(f"Loading: {pdf}")
        loader = PyPDFLoader(pdf)
        all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(all_docs)

# --- 3. VECTOR DB INITIALIZATION ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if the database directory doesn't exist or is empty
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    print("Building new index for all PDFs...")
    docs = load_all_pdfs(PDF_FOLDER)
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=DB_DIR
    )
else:
    print("Loading existing index...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 4. THE CONVERSATIONAL LOGIC ---
# temperature 0 for factual, consistent responses
llm = ChatOllama(model=MODEL_NAME, temperature=0)

# This prompt tells the AI how to use context and history
# Instruction for the AI to transform follow-up questions into clear, independent search queries
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Create a prompt template that combines the system instruction, chat history, and new user input
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# A sub-chain that: 1. Rephrases the question -> 2. Sends it to the LLM -> 3. Cleans output -> 4. Searches the DB
# User Question
#       ↓
# LLM rewrites question
#       ↓
# Standalone Question
#       ↓
# Retriever
#       ↓
# Relevant Documents

history_aware_retriever = (
    contextualize_q_prompt 
    | llm 
    | StrOutputParser() 
    | retriever
)

# Define the instruction for the final answering step, incorporating the retrieved PDF snippets
qa_system_prompt = (
    "You are an expert research assistant. Use the following pieces of retrieved "
    "context to answer the user's question. If you don't know the answer, "
    "just say that you don't know. Keep the answer concise."
    """
    Use the context provided to answer 
    the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do 
    not know the answer to their question based on the context
    provided and that you are sorry.
    """
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The Main Chain
rag_chain = (
    # Step 1: Add a 'context' key to the input data by running the history-aware retriever
    RunnablePassthrough.assign(
        context=lambda x: format_docs(history_aware_retriever.invoke(x))
    )
    | qa_prompt         # Step 2: Pass the context, history, and input into the Q&A prompt
    | llm               # Step 3: Send the formatted prompt to the LLM
    | StrOutputParser() # Step 4: Extract the final string from the LLM's response
)

# --- 5. THE CHAT LOOP ---
chat_history = []

print(f"{'='*50}\n--- NotebookLM Local Chat Started ---\n{'='*50}") 
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Get Response
    ai_response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
    print(f"\nAI: {ai_response}\n{'-'*50}\n")
    
    # Update History
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=ai_response),
    ])
    
    # Optional: Keep history short to save memory
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]