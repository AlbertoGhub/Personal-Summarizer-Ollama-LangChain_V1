# 💬 Personal-Summarizer-Ollama-LangChain_V1 - Chat with Your PDFs Using Local Large Language Models

A local ```RAG-based``` chatbot that lets you interact with your ```PDF``` documents using open-source Large Language Models like ```Gemma 2``` via ```Ollama```, powered by ```LangChain``` and ```Chroma```.

## 🧠 Features

- 💡 Ask questions about your ```PDF``` content
- 🔎 ```RAG``` (Retrieval-Augmented Generation) pipeline for accurate answers and avoid hallucinations
- 🧱 Built with LangChain + Chroma for efficient document retrieval
- 🤖 Runs fully local using ```Ollama``` + ```Gemma 2``` (or any compatible LLM)
- 🗂️ Supports multiple ```PDF``` documents

## 🧰 Tech Stack

- [Ollama](https://ollama.com) – to run local LLMs like ```Gemma 2``` or any other based on hardware resources
- [LangChain](https://python.langchain.com/docs/tutorials/) – for chaining ```LLMs``` and retrieval logic
- [Chroma](https://www.trychroma.com) – vector store for embedding and retrieval
- [Gemma 2](https://ai.google.dev/gemma) – open-source ```LLM``` from Google (via Ollama)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) – for ```PDF``` text extraction

## 🚀 Getting Started

### 1. Install Ollama and a Model

Install [Ollama](https://ollama.com) and pull your preferred model (e.g., Gemma 2, or replace with another supported model, like llama3, mistral, etc.):

### 2. Clone This Repository

Go to the repository link and run the following command in your terminal.

```bash
git clone "link of the project"
```

### 3. Install Dependencies
Make sure Python 3.9+ is installed, then:

```bash
conda env create --prefix ./env -f environment.yml
```

### 4. Add Your PDFs
Place your ```PDF``` files in the ```data/PDFs/``` directory This serves as the local knowledge base, ensuring the model generates responses solely from the provided documents, thereby reducing the risk of hallucinations or unrelated outputs.

### 5. Run the App
The test can be run either as a Jupyter Notebook (```.ipynb file```) or as a Python application (```.py file```), both of which are located in the ```\notebook``` folder.

# 📦 Project Structure

```bash
project/
├── data/
│   └── PDFs/
│   │   └── sample.pdf
│   └── sample_question.md
├── notebook/
│   └── main.py
│   └── main.ipynb
├── src/
│   └── modules/
│       └── __init__.py
│       └── functions.py
│       └── ingesting.py
│       └── vector_store_module.py
```

## 🛡️ Notes

- Before executing the project, ensure that ollama is running.
- Bear in mind that all processing and inference are performed locally.
- Make sure your system supports running the selected ```LLM``` (some may require > ```8GB RAM```).

# 👨‍💻 Author

Made with ❤️ by Alberto AJ - AI/ML Engineer.

📢 GitHub Badges

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
