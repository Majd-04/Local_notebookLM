#!/bin/bash

echo "🚀 Starting environment setup with Virtual Environment..."

# 1. Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "⬇️ Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 2. Create and Activate Virtual Environment
if [ ! -d "local_notebook" ]; then
    echo "🛠️ Creating virtual environment..."
    python3 -m venv local_notebook
fi

echo "🔌 Activating virtual environment..."
source local_notebook/bin/activate

# 3. Upgrade pip and install libraries
echo "📦 Installing Python libraries..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Pull models
echo "🧠 Pulling models..."
ollama pull llama3.2
ollama pull mxbai-embed-large

# 5. Folders
mkdir -p my_notebook

echo "✅ Setup complete!"
echo "⚠️  NOTE: Before running your python script, remember to type: source local_notebook/bin/activate"
echo "💡 Tip: Place your PDFs in the 'my_notebook' folder before starting."