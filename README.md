# 🚀 Research Navigator: Semantic Paper Analyzer

**An advanced, full-stack AI platform for rapid document analysis, semantic synthesis, and academic insight extraction.**

![Project Banner](https://raw.githubusercontent.com/BSshishodia/ResearchNavigator-AI/main/static/papers.mp4) 
*(Note: Replace with a static image or GIF of your landing page for better GitHub viewing)*

## 📖 Project Overview
[cite_start]The exponential growth of academic literature presents a major challenge for researchers[cite: 5]. [cite_start]**Research Navigator** (PaperIQ) is a containerized solution that goes beyond simple keyword searches, enabling users to analyze local files and utilize a search-grounded Gemini AI Assistant to synthesize multiple web sources instantly[cite: 8].

### **Core Capabilities**
* [cite_start]**Hybrid Summarization:** Generates **Abstractive summaries** via BART-Large-CNN and **Extractive highlights** using TextRank[cite: 30, 31].
* [cite_start]**Search-Grounded AI Assistant:** Powered by **Gemini 2.5 Flash**, the bot performs real-time searches to fetch and synthesize up to 5 papers into one cohesive report[cite: 35, 37].
* [cite_start]**Deep-Level NLP Metrics:** Instant extraction of Named Entities (spaCy), Keywords (KeyBERT), and **Flesch-Kincaid Readability** grades[cite: 32, 33].
* [cite_start]**Persistent History:** A secure **MongoDB** backend persists all user profiles and analysis history[cite: 9, 23].

---

## 🏗️ System Architecture & Data Flow
[cite_start]The application follows a tightly integrated modular structure to ensure stability and data integrity[cite: 19].

### **1. The Data Flow**
1.  [cite_start]**Ingestion Layer:** Handles local PDF/DOCX uploads and web-scraped content from the AI Assistant[cite: 25, 26].
2.  [cite_start]**Preprocessing Layer:** Robust text cleaning and sentence segmentation using NLTK[cite: 27].
3.  [cite_start]**NLP Pipeline Layer:** * **NER:** Identifies organizations, people, and locations via spaCy[cite: 32].
    * [cite_start]**Keywords:** Scores relevance using KeyBERT[cite: 32].
    * [cite_start]**Summarization:** Concurrent execution of BART-CNN and TextRank models[cite: 30, 31].
4.  [cite_start]**Persistence Layer:** All results are committed to user-specific collections in MongoDB[cite: 23].

### **2. Workflow Diagram**

[cite_start]*User Action -> System Processing (NLP Pipeline) -> External Data (Gemini/Search) -> Persistent Storage (MongoDB)[cite: 104].*

---

## 🐳 Why Docker?
[cite_start]This project is fully containerized to ensure **Environment Parity** across different development and production machines[cite: 100].

* **Microservices Orchestration:** Uses `docker-compose` to manage the Flask web server and MongoDB database as separate, linked services.
* **Optimized Build:** The `Dockerfile` is configured to pre-fetch large transformer model weights and NLTK corpora (including `punkt_tab`) during the build stage to prevent runtime errors.
* [cite_start]**Portability:** Simplifies deployment by encapsulating complex ML dependencies like PyTorch, Transformers, and spaCy[cite: 44, 100].

---

## 🛠️ Technology Stack
| Category | Tools & Libraries |
| :--- | :--- |
| **Backend** | [cite_start]Python 3, Flask, Werkzeug [cite: 44] |
| **Database** | [cite_start]MongoDB (PyMongo) [cite: 44] |
| **AI/LLM** | [cite_start]Gemini 2.5 Flash API [cite: 44] |
| **NLP Pipeline** | [cite_start]BART-Large-CNN, KeyBERT, Sentence-Transformers, spaCy, NLTK [cite: 44] |
| **Data Science** | [cite_start]NumPy, Scikit-learn, NetworkX [cite: 44] |
| **Frontend** | [cite_start]Jinja2, Tailwind CSS, Chart.js [cite: 44] |
| **Document Handling** | [cite_start]PyMuPDF (PDF), python-docx (DOCX) [cite: 44] |

---

## 🚀 Quick Start

### **1. Prerequisites**
* [cite_start]**Docker Desktop** installed[cite: 100].
* [cite_start]**Gemini API Key** (Obtained from Google AI Studio)[cite: 68].

### **2. Installation**
```bash
# Clone the repository
git clone [https://github.com/BSshishodia/ResearchNavigator-AI.git](https://github.com/BSshishodia/ResearchNavigator-AI.git)
cd ResearchNavigator-AI

# Configure Environment Variables
# Create a .env file and add your credentials:
GEMINI_API_KEY=your_key_here
MONGO_URI=mongodb://db:27017/paperiq_db
SECRET_KEY=your_secret_key
