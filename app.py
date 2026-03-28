import os
import re
import fitz
import requests
from docx import Document
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import json
import smtplib
from email.mime.text import MIMEText
from typing import List, Dict, Union
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from transformers import pipeline, BartTokenizer
from rouge_score import rouge_scorer
import random
import string
from collections import Counter
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

# --- FLASK SETUP ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_paperiq_key")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# --- MONGO DB SETUP ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/paperiq_db")
client = None
db = None
users_collection = None
analyses_collection = None

try:
    client = MongoClient(MONGO_URI)
    db = client.get_database() # Uses the database name from the URI
    users_collection = db['users']
    analyses_collection = db['analyses']
    print("MongoDB connection successful.")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    # Set collections to None if connection fails
    users_collection = None
    analyses_collection = None

# --- GEMINI SETUP ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# --- NLP/MODEL SETUP (Initialized once) ---
try:
    nltk.download("punkt", quiet=True)
    # Check if a model is downloaded, if not, skip loading
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        print("SpaCy model 'en_core_web_sm' not found. Skipping entity extraction.")

    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    BART_MODEL_NAME = "facebook/bart-large-cnn"
    bart_pipeline = pipeline("summarization", model=BART_MODEL_NAME, device=-1)
    bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
    print("NLP Models loaded successfully.")
except Exception as e:
    print(f"Error loading NLP components: {e}. Analysis functions may be impaired.")


# --- CORE UTILITIES ---

def get_user_report_count(user_id: str) -> int:
    """Fetches the total number of reports saved by the user."""
    # NEW FUNCTION: Fetches the count from MongoDB
    if analyses_collection is not None:
        # ROBUST CHECK: Using is not None for collection object
        if analyses_collection is not None:
            return analyses_collection.count_documents({"user_id": user_id})
    return 0

def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_verification_email(user_email, otp):
    # This function depends on correct SMTP settings in the .env file
    try:
        sender_email = os.getenv("SMTP_USERNAME")
        sender_password = os.getenv("SMTP_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))

        if not all([sender_email, sender_password, smtp_server]):
             print("SMTP settings are incomplete. Skipping email.")
             return True # Simulate success for environments without email setup

        msg = MIMEText(f"Your PaperIQ verification code is: {otp}")
        msg['Subject'] = "PaperIQ Email Verification Code"
        msg['From'] = sender_email
        msg['To'] = user_email
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# --- DOCUMENT EXTRACTION LOGIC (Simplified for web content) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    if ext == '.pdf':
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text("text")
            return text.strip()
        except Exception as e:
            return f"[ERROR] Could not extract PDF: {str(e)}"
    elif ext == '.docx':
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text.strip() + "\n"
            return text.strip()
        except Exception as e:
            return f"[ERROR] Could not extract DOCX: {str(e)}"
    return "[ERROR] Unsupported file type."

def fetch_paper_content(url: str) -> str:
    """Simulates fetching the full text from a paper's URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # A basic content extraction attempt (Note: this is highly unreliable for complex sites)
        text = re.sub(r'<script.*?</script>|<style.*?</style>|<!--.*?-->', '', response.text, flags=re.DOTALL)
        text = re.sub(r'<[^>]*>', ' ', text)
        return clean_text(text)
    except requests.exceptions.RequestException as e:
        print(f"Fetch Error for {url}: {e}")
        return f"[FETCH_ERROR] Could not retrieve content from URL. Site blocked or network error: {e}"

# --- TEXT PREPROCESSING AND ANALYSIS LOGIC ---

def clean_text(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[’‘]", "'", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    # Use NLTK for robust sentence splitting
    return sent_tokenize(text)

def calculate_readability(text: str, sentences: List[str]) -> Dict[str, Union[float, int]]:
    """Calculates Flesch-Kincaid Grade Level and other basic metrics."""
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text_clean.lower().split()
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = 0
    
    # Simple (imperfect) syllable counting: count vowels per word
    for word in words:
        vowel_count = len(re.findall(r'[aeiouy]+', word))
        total_syllables += max(1, vowel_count) # Ensure every word has at least one syllable

    fk_grade = 0.0
    if total_words > 0 and total_sentences > 0:
        avg_words_per_sentence = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        # Flesch-Kincaid Grade Level Formula
        fk_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59

    return {
        "fk_grade": round(max(0, fk_grade), 1),
        "total_syllables": total_syllables
    }

class ExtractiveSummarizer:
    def __init__(self, embedder):
        self.embedder = embedder

    def _build_similarity_graph(self, sentences: List[str]) -> np.ndarray:
        if not sentences: return np.array([[]])
        embeds = self.embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        sim = cosine_similarity(embeds)
        np.fill_diagonal(sim, 0)
        return sim

    def summarize(self, text: str, num_sentences: int = 4) -> str:
        text = clean_text(text)
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences: return " ".join(sentences)
        
        sim = self._build_similarity_graph(sentences)
        graph = nx.from_numpy_array(sim)
        scores = nx.pagerank(graph, weight='weight')
        ranked = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
        selected = sorted(ranked[:num_sentences], key=lambda x: x[2])
        return " ".join([s for _, s, _ in selected])

class AbstractiveSummarizer:
    def __init__(self, pipe, tokenizer):
        self.pipe = pipe
        self.tokenizer = tokenizer

    def _chunk_text(self, text: str, max_tokens: int = 1000) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        token_chunks = [tokens[i:i + 1000] for i in range(0, len(tokens), 1000)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        text = clean_text(text)
        chunks = self._chunk_text(text)
        summaries = []

        for chunk in chunks:
            out = self.pipe(
                chunk,
                max_length=max_length,
                min_length=min_length,
                truncation=True
            )
            summaries.append(out[0]['summary_text'])

        if len(summaries) > 1:
            combined_summary = " ".join(summaries)
            second_pass_summary = self.pipe(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                truncation=True
            )[0]['summary_text']
            return second_pass_summary
        else:
            return summaries[0]

def rouge_scores(pred: str, ref: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: v.fmeasure for k, v in scores.items()}

def extract_entities(text: str) -> List[Dict[str, str]]:
    if nlp is None:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def analyze_entity_frequency(entities: List[Dict[str, str]]) -> Dict[str, Union[List, List]]:
    counts = Counter(e['label'] for e in entities)
    labels = list(counts.keys())
    data = list(counts.values())
    return {'labels': labels, 'counts': data}

def extract_keywords_with_scores(text: str, top_n: int = 10, diversity: float = 0.7) -> List[Dict[str, Union[str, float]]]:
    keywords_with_scores = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=diversity,
    )
    return [{'text': kw, 'score': float(score)} for kw, score in keywords_with_scores]

def extract_key_insights(sentences: List[str], top_k: int = 5) -> List[str]:
    if not sentences: return []
    keywords = ["results", "conclusion", "our model", "we found", "our findings", "significant", "performance", "accuracy", "future work"]
    keyword_embedding = embed_model.encode(keywords, convert_to_tensor=True).cpu().numpy().mean(axis=0)
    sentence_embeddings = embed_model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    scores = cosine_similarity([keyword_embedding], sentence_embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [sentences[i] for i in top_indices]

def get_summaries(text: str) -> Dict[str, str]:
    extractive_summarizer = ExtractiveSummarizer(embed_model)
    extractive_summary = extractive_summarizer.summarize(text)
    
    abstractive_summarizer = AbstractiveSummarizer(bart_pipeline, bart_tokenizer)
    abstractive_summary = abstractive_summarizer.summarize(text)

    return {
        "extractive_summary": extractive_summary,
        "abstractive_summary": abstractive_summary,
    }

def run_full_analysis(text: str, filename: str, is_ai_analysis: bool = False) -> Dict[str, Union[Dict, List, str, float]]:
    """Runs the full NLP pipeline and saves results to MongoDB."""
    
    # 1. Preprocessing and Metrics
    cleaned_text = clean_text(text)
    sentences = split_sentences(cleaned_text)
    
    if not cleaned_text:
        raise ValueError("Extracted text was empty or unreadable.")

    raw_text_sample = cleaned_text[:2000]
    full_text_for_analysis = cleaned_text 
    
    # 2. Analysis Modules
    summaries = get_summaries(full_text_for_analysis)
    abstractive_summary = summaries["abstractive_summary"]
    simplified_ref = summaries["extractive_summary"] 
    rouge_scores_data = rouge_scores(abstractive_summary, simplified_ref)
    
    entities = extract_entities(full_text_for_analysis)
    keywords_with_scores = extract_keywords_with_scores(full_text_for_analysis)
    readability_scores = calculate_readability(full_text_for_analysis, sentences)

    # 3. Structure Data for Visuals
    entity_counts = analyze_entity_frequency(entities)
    keyword_data = {
        'labels': [item['text'] for item in keywords_with_scores],
        'counts': [item['score'] for item in keywords_with_scores]
    }
    sentence_lengths = [len(s.split()) for s in sentences if 0 < len(s.split()) < 100] 

    visual_data = {
        "filename": filename,
        "keyword_data": keyword_data,
        "entity_counts": entity_counts,
        "sentence_lengths": sentence_lengths,
        "total_sentences": len(sentences),
        "total_words": len(full_text_for_analysis.split()),
        "fk_grade": readability_scores['fk_grade']
    }

    # 4. Structure Data for Summary Dashboard
    summary_data = {
        "filename": filename,
        "raw_text_sample": raw_text_sample,
        "extractive_summary": summaries["extractive_summary"],
        "abstractive_summary": abstractive_summary,
        "insights": extract_key_insights(sentences),
        "keywords": [item['text'] for item in keywords_with_scores][:5], 
        "rouge_scores": rouge_scores_data,
        "visuals_ready": True
    }
    
    # 5. Save to MongoDB (Persistent History)
    if analyses_collection is not None:
        analysis_document = {
            "user_id": session.get('user_id'),
            "username": session.get('username'),
            "timestamp": datetime.utcnow(),
            "filename": filename,
            "is_ai_analysis": is_ai_analysis,
            "summary_data": summary_data,
            "visual_data": visual_data
        }
        result = analyses_collection.insert_one(analysis_document)
        analysis_id = str(result.inserted_id)
    else:
        # Fallback if DB is down, store in session temporarily (non-persistent)
        analysis_id = "temp-" + "".join(random.choices(string.ascii_letters, k=10))
        session[analysis_id] = {"summary_data": summary_data, "visual_data": visual_data}

    # 6. Store current report in session for immediate redirect
    session['current_analysis_id'] = analysis_id
    
    return {"analysis_id": analysis_id}


# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def index_route():
    # Redirect authenticated users to the home page
    if session.get('logged_in'):
        return redirect(url_for('home_route'))
    # Otherwise, show the login screen
    return redirect(url_for('login_route'))

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # ROBUST CHECK: Using is not None for collection object
        if users_collection is not None:
            user = users_collection.find_one({"username": username})
        else:
            error = "Database connection error."
            return render_template('login.html', error=error)
            
        if user and user.get('verified') and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = str(user['_id'])
            return redirect(url_for('home_route'))
        else:
            error = "Invalid username, password, or account not verified."
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['POST'])
def register_route():
    username = request.form.get('reg_username')
    password = request.form.get('reg_password')
    confirm_password = request.form.get('reg_confirm_password')
    email = request.form.get('reg_email')
    
    # ROBUST CHECK: Ensure collection is available
    if users_collection is None:
        return jsonify({"success": False, "message": "Server error: Database not connected."}), 500

    if users_collection.find_one({"username": username}):
        return jsonify({"success": False, "message": "User already exists!"}), 409
    
    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match!"}), 400

    # Password strength check
    if not (len(password) >= 8 and any(c.isupper() for c in password) and any(c.islower() for c in password) and any(c.isdigit() for c in password) and any(not c.isalnum() for c in password)):
        return jsonify({"success": False, "message": "Password must be at least 8 characters long and contain uppercase, lowercase, number, and a special character."}), 400

    hashed_password = generate_password_hash(password)
    otp = generate_otp()
    
    user_document = {
        'username': username,
        'password': hashed_password,
        'email': email,
        'verified': False,
        'otp': otp
    }
    
    if send_verification_email(email, otp):
        users_collection.insert_one(user_document)
        return jsonify({"success": True, "message": "Registration successful! Please check your email for a verification code."})
    else:
        # If email fails, still save user but notify of email issue
        users_collection.insert_one(user_document)
        return jsonify({"success": True, "message": "Registration successful, but email failed to send. Please proceed to OTP verification."})


@app.route('/verify_otp', methods=['POST'])
def verify_otp_route():
    username = request.form.get('otp_username')
    otp_code = request.form.get('otp_code')
    
    if users_collection is None:
        return jsonify({"success": False, "message": "Server error: Database not connected."}), 500
    
    user = users_collection.find_one({"username": username})
    
    if user and user.get('otp') == otp_code:
        # Use $set and $unset for atomic update
        users_collection.update_one(
            {"username": username},
            {"$set": {"verified": True}, "$unset": {"otp": ""}}
        )
        return jsonify({"success": True, "message": "Email verified successfully! You can now log in."})
    else:
        return jsonify({"success": False, "message": "Invalid username or OTP. Please try again."}), 400

@app.route('/logout')
def logout_route():
    session.clear()
    return redirect(url_for('login_route'))

# --- NEW NAVIGATION ROUTES ---

@app.route('/home')
def home_route():
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
    
    # ADDED: Fetch report count for personalized greeting (Phase 6)
    report_count = get_user_report_count(session.get('user_id'))
    
    return render_template('home.html', username=session.get('username'), report_count=report_count)

@app.route('/chat')
def chat_route():
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
    
    # Check if key is available to enable/disable button
    gemini_key_available = bool(GEMINI_API_KEY)
    return render_template('chat.html', gemini_key_available=gemini_key_available)

@app.route('/history')
def history_route():
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
    
    # Load history from MongoDB (most recent first)
    if analyses_collection is not None:
        user_id = session.get('user_id')
        reports = analyses_collection.find(
            {"user_id": user_id}
        ).sort([('timestamp', -1)]).limit(50)
        
        # Convert MongoDB objects for rendering
        history_list = []
        for report in reports:
            summary_data = report.get('summary_data')
            
            # FIXED: Only include reports that have the necessary summary data structure
            if summary_data and summary_data.get('abstractive_summary'):
                history_list.append({
                    'id': str(report['_id']),
                    'filename': report['filename'],
                    'timestamp': report['timestamp'],
                    'is_ai_analysis': report.get('is_ai_analysis', False),
                    # Accessing the snippet safely:
                    'summary_snippet': summary_data['abstractive_summary'][:150] + "..."
                })
    else:
        history_list = []

    return render_template('history.html', history=history_list)

# --- ANALYSIS ROUTES ---

@app.route('/upload', methods=['GET', 'POST'])
def upload_route():
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
    
    if request.method == 'GET':
        return render_template('upload.html')

    # --- POST REQUEST: ANALYSIS START ---
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('upload.html', error='No selected file.')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        extracted_text = extract_text(filepath)
        os.remove(filepath) 
        
        if extracted_text.startswith("[ERROR]"):
            return render_template('upload.html', error=extracted_text)

        if not extracted_text:
            return render_template('upload.html', error="Error: Extracted text was empty or unreadable.")
        
        try:
            # Run the full analysis pipeline
            analysis_result = run_full_analysis(extracted_text, filename, is_ai_analysis=False)
            
            # Redirect to the new report route
            return redirect(url_for('report_route', analysis_id=analysis_result['analysis_id']))

        except Exception as e:
            print(f"Analysis Pipeline Error: {e}")
            return render_template('upload.html', error=f"Analysis failed due to model or data error: {e}")

    else:
        return render_template('upload.html', error='Invalid file type. Only PDF and DOCX are supported.')


# --- REPORT LOADING ROUTES ---

@app.route('/dashboard')
def dashboard_route():
    """Route to view the current report in the session."""
    analysis_id = session.get('current_analysis_id')
    if not analysis_id:
        return redirect(url_for('upload_route', error="No active report found."))
    
    return redirect(url_for('report_route', analysis_id=analysis_id))


@app.route('/report/<analysis_id>')
def report_route(analysis_id):
    """Loads a specific analysis report from MongoDB (or session fallback)."""
    
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
        
    user_id = session.get('user_id')
    report = None
    
    # 1. Try to load from MongoDB
    if analyses_collection is not None:
        try:
            report_doc = analyses_collection.find_one({
                "_id": ObjectId(analysis_id),
                "user_id": user_id
            })
            if report_doc:
                report = report_doc
        except:
            # ObjectId conversion failed or report not found
            pass 
    
    # 2. Try to load from Session (for temporary reports if DB was down)
    if report is None and analysis_id.startswith("temp-"):
        temp_data = session.get(analysis_id)
        if temp_data:
            report = {
                "summary_data": temp_data['summary_data'],
                "visual_data": temp_data['visual_data']
            }

    if not report:
        return redirect(url_for('history_route', error="Report not found or access denied."))

    # Store loaded report data in session for use by visuals route
    session['summary_data'] = report['summary_data']
    session['visual_data'] = report['visual_data']
    
    return render_template('results.html', results=report['summary_data'])


@app.route('/visuals')
def visuals_route():
    """Renders the visualizations page using data stored in the session."""
    if not session.get('logged_in'):
        return redirect(url_for('login_route'))
    
    visual_data = session.get('visual_data')
    
    if visual_data is None:
        return redirect(url_for('upload_route', error="No visualization data found. Please upload a document."))

    # Histogram binning logic for Chart.js
    lengths = visual_data.get("sentence_lengths", [])
    
    if lengths:
        max_len = max(lengths)
        bin_size = 10
        bins = list(range(0, max_len + bin_size, bin_size)) 
        
        hist_counts = [0] * (len(bins) - 1)
        
        for length in lengths:
            if 0 < length < 100: # Only count lengths under 100 words
                bin_index = min(int(length / bin_size), len(bins) - 2) 
                hist_counts[bin_index] += 1
        
        hist_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

        visual_data['sentence_length_hist'] = {
            'labels': hist_labels,
            'counts': hist_counts
        }
    else:
         visual_data['sentence_length_hist'] = {'labels': ["N/A"], 'counts': [0]}

    return render_template('visuals.html', visual_data=visual_data)


# --- AI ASSISTANT API ROUTES ---

def make_gemini_api_call(user_prompt: str, system_prompt: str, use_search: bool = True):
    """Helper function to execute the Gemini API call."""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key is missing from .env file."}, 503

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    headers = {'Content-Type': 'application/json'}
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        # Check for structured search results
        if use_search and "search_results" in text.lower():
            # Attempt to extract JSON from the text response
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                json_string = match.group(1)
                return json.loads(json_string), 200
        
        return {"message": text}, 200

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Failed: {e}"}, 500
    except json.JSONDecodeError:
        return {"error": f"API returned non-JSON search results: {text[:200]}..."}, 500
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}, 500


@app.route('/api/search_papers', methods=['POST'])
def search_papers_api():
    """Searches for papers using Gemini + Google Search, structured response."""
    data = request.get_json()
    query = data.get('query', '')

    # System prompt forces Gemini to return a structured JSON list of search results
    system_prompt = (
        "You are a research assistant. The user wants you to find documents related to their query. "
        "Use Google Search to find relevant academic papers or detailed reports. "
        "You MUST respond ONLY with a single JSON object containing a 'search_results' array. "
        "The array must contain up to 5 objects, each with 'title', 'snippet', and 'url'. "
        "If the query is a general question (e.g., 'What is AI?'), provide a direct answer in Markdown format instead, but wrap the entire response in a 'message' key, and do NOT use the 'search_results' key."
    )
    
    result, status = make_gemini_api_call(query, system_prompt, use_search=True)
    
    if status != 200:
        return jsonify({"success": False, "error": result.get('error', 'API call failed.')}), status

    if "search_results" in result:
        # Structured result for paper selection
        return jsonify({"success": True, "search_results": result['search_results']}), 200
    elif "message" in result:
        # Direct answer/chat result
        
        # If the chat query is a direct summary, run the full analysis on the text result
        if "summarize" in query.lower() or "analysis" in query.lower():
            try:
                # The 'message' text is the content to be analyzed
                analysis_result = run_full_analysis(result['message'], filename="AI Search Result", is_ai_analysis=True)
                analysis_url = url_for('report_route', analysis_id=analysis_result['analysis_id'])
                return jsonify({"success": True, "message": result['message'], "analysis_url": analysis_url}), 200
            except Exception as e:
                 return jsonify({"success": False, "error": f"Analysis failed on AI response: {e}"}), 500
        else:
            return jsonify({"success": True, "message": result['message']}), 200


@app.route('/api/multi_summarize', methods=['POST'])
def multi_summarize_api():
    """Fetches selected papers, combines content, and runs full analysis."""
    data = request.get_json()
    urls = data.get('urls', [])

    if not urls or len(urls) > 5:
        return jsonify({"success": False, "error": "Invalid number of URLs selected (max 5)."}), 400

    combined_content = []
    
    for url in urls:
        content = fetch_paper_content(url)
        if not content.startswith("[FETCH_ERROR]"):
            # Concatenate non-error content
            combined_content.append(content)

    if not combined_content:
        return jsonify({"success": False, "error": "Could not retrieve usable text from any of the selected URLs."}), 500

    full_text = "\n\n---\n\n".join(combined_content)
    
    # 2. Use Gemini to synthesize a single summary/report from the combined text
    synthesis_prompt = (
        f"You have been provided with the full text content from {len(urls)} documents. "
        "Synthesize this content into a single, cohesive, abstractive research report summary. "
        "The final summary must be highly readable and written in Markdown format. "
        "Document Content:\n\n"
        f"{full_text[:5000]}" # Pass a sample to Gemini for synthesis guidance
    )
    
    result, status = make_gemini_api_call(synthesis_prompt, "You are an expert research synthesist. Synthesize the provided texts into one cohesive report.", use_search=False)

    if status != 200:
        return jsonify({"success": False, "error": result.get('error', 'Synthesis failed.')}), status

    synthesized_text = result.get('message', 'Synthesis failed.')

    # 3. Run the full NLP pipeline on the synthesized content
    try:
        combined_filename = f"Multi-Analysis Report ({len(urls)} Papers)"
        analysis_result = run_full_analysis(full_text, filename=combined_filename, is_ai_analysis=True)
        
        analysis_url = url_for('report_route', analysis_id=analysis_result['analysis_id'])
        
        # Return synthesized text PLUS a link to the detailed report
        final_message = f"**Multi-Analysis Complete!**\n\n{synthesized_text}"

        return jsonify({"success": True, "message": final_message, "analysis_url": analysis_url}), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Detailed NLP analysis failed after synthesis: {e}"}), 500


if __name__ == '__main__':
    # Ensure all models are loaded before starting the app (non-critical, but good practice)
    if 'nlp' not in globals():
         print("WARNING: spaCy model not loaded. Entity extraction will be skipped.")
    if 'kw_model' not in globals():
         print("WARNING: KeyBERT model not loaded. Keyword extraction will be impaired.")
         
    # '0.0.0.0' is required to make the app accessible outside the Docker container
    # Port 5000 must match the port exposed in your Dockerfile and docker-compose.yml
    app.run(debug=True, host='0.0.0.0', port=5000)
