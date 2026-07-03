from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.corpus import wordnet, stopwords
from urllib.parse import urlparse, parse_qs
import re
from collections import Counter
import os
import requests
import numpy as np

nltk.data.path.append("./nltk_data")
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', download_dir="./nltk_data")
    stop_words = set(stopwords.words('english'))

# Use Hugging Face Inference API instead of local model to save memory
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def get_embeddings(texts):
    if not texts:
        return []
    if isinstance(texts, str):
        texts = [texts]
        
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": batch, "options": {"wait_for_model": True}}, timeout=30)
            if response.status_code == 200:
                embeddings.extend(response.json())
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                embeddings.extend([[0.0]*384 for _ in batch])
        except Exception as e:
            print(f"Exception calling HF API: {e}")
            embeddings.extend([[0.0]*384 for _ in batch])
    return embeddings

def cos_sim_1d_to_2d(query_emb, embs_list):
    if not embs_list or not query_emb:
        return []
    q = np.array(query_emb)
    embs = np.array(embs_list)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    
    dot_products = np.dot(embs, q)
    norms_q = np.linalg.norm(q)
    norms_embs = np.linalg.norm(embs, axis=1)
    
    denominators = np.maximum(norms_q * norms_embs, 1e-10)
    return (dot_products / denominators).tolist()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    video_url: str
    concept: str

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
    except Exception:
        pass
    return url

def get_expanded_concepts(keyword):
    synonyms = set([keyword.lower()])
    try:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' ').lower())
    except LookupError:
        pass
    return list(synonyms)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

def extract_frequent_keywords(transcript_list, max_keywords=40):
    text = " ".join([t['text'] for t in transcript_list]).lower()
    words = re.findall(r'\b[a-z]{4,}\b', text)
    filtered = [w for w in words if w not in stop_words]
    return [word for word, _ in Counter(filtered).most_common(max_keywords)]

def get_related_transcript_keywords(transcript_keywords, query, min_score=0.35, top_n=10):
    if not transcript_keywords:
        return []

    query_embedding = get_embeddings([query])[0]
    keyword_embeddings = get_embeddings(transcript_keywords)
    scores = cos_sim_1d_to_2d(query_embedding, keyword_embeddings)

    scored_keywords = sorted(zip(transcript_keywords, scores), key=lambda x: x[1], reverse=True)
    related = [word for word, score in scored_keywords if score >= min_score][:top_n]
    if not related:
        related = [word for word, _ in scored_keywords[:min(top_n, len(scored_keywords))]]
    return related

def get_similar_topics(transcript_list, concept):
    video_keywords = extract_frequent_keywords(transcript_list)
    concept_synsets = wordnet.synsets(concept)
    if not concept_synsets:
        concept_synsets = []
        for word in concept.split():
            concept_synsets.extend(wordnet.synsets(word))
    
    if not concept_synsets:
        return video_keywords[:5] 
        
    keyword_scores = []
    for kw in video_keywords:
        kw_syns = wordnet.synsets(kw)
        if kw_syns:
            max_sim = 0
            for s1 in concept_synsets:
                for s2 in kw_syns:
                    sim = s1.wup_similarity(s2)
                    if sim and sim > max_sim:
                        max_sim = sim
            keyword_scores.append((kw, max_sim))
            
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return [k[0] for k in keyword_scores[:5]]

# Path to cookies file — allows YouTube to treat requests as a logged-in user,
# bypassing IP blocks on cloud providers like AWS/Render.
COOKIES_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")

@app.post("/analyze")
def analyze_video(request: QueryRequest):
    video_id = extract_video_id(request.video_url)

    try:
        if os.path.exists(COOKIES_PATH):
            client = YouTubeTranscriptApi(cookies=COOKIES_PATH)
        else:
            client = YouTubeTranscriptApi()
        transcript = client.fetch(video_id).to_raw_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not retrieve transcript: {str(e)}")

    expanded_keywords = get_expanded_concepts(request.concept)
    transcript_keywords = extract_frequent_keywords(transcript, max_keywords=40)
    sentence_texts = [item['text'].lower() for item in transcript]

    query_embedding = get_embeddings([request.concept])[0]
    sentence_embeddings = get_embeddings(sentence_texts) if sentence_texts else []
    sentence_similarity_scores = cos_sim_1d_to_2d(query_embedding, sentence_embeddings) if sentence_embeddings else []
    related_keywords = get_related_transcript_keywords(transcript_keywords, request.concept)

    clusters = []
    current_cluster = None
    CLUSTER_THRESHOLD = 45.0 
    
    for idx, item in enumerate(transcript):
        text = item['text'].lower()
        matched_words = [kw for kw in expanded_keywords if kw in text]

        if not matched_words:
            for related_kw in related_keywords:
                if related_kw in text:
                    matched_words.append(related_kw)
                    break

        if not matched_words and sentence_similarity_scores and idx < len(sentence_similarity_scores):
            if sentence_similarity_scores[idx] >= 0.55:
                matched_words.append(request.concept)

        if matched_words:
            if current_cluster is None:
                current_cluster = {
                    "start": item['start'],
                    "end": item['start'] + item['duration'],
                    "texts": [item['text']],
                    "matched_concepts": set(matched_words)
                }
            else:
                if item['start'] - current_cluster['end'] <= CLUSTER_THRESHOLD:
                    current_cluster['end'] = max(current_cluster['end'], item['start'] + item['duration'])
                    current_cluster['texts'].append(item['text'])
                    current_cluster['matched_concepts'].update(matched_words)
                else:
                    clusters.append(current_cluster)
                    current_cluster = {
                        "start": item['start'],
                        "end": item['start'] + item['duration'],
                        "texts": [item['text']],
                        "matched_concepts": set(matched_words)
                    }
    if current_cluster:
        clusters.append(current_cluster)

    formatted_results = []
    for cluster in clusters:
        combined_text = " ".join(cluster['texts'])
        formatted_results.append({
            "timestamp": cluster['start'],
            "formatted_time": format_time(cluster['start']) + " - " + format_time(cluster['end']),
            "text": combined_text,
            "matched_concepts": list(cluster['matched_concepts'])
        })
        
    suggested_topics = []
    if not formatted_results:
        suggested_topics = get_similar_topics(transcript, request.concept)
            
    return {
        "video_id": video_id,
        "original_concept": request.concept,
        "expanded_concepts": expanded_keywords,
        "results": formatted_results,
        "suggested_topics": suggested_topics
    }

@app.get("/")
def read_root():
    return {"message": "NLP YouTube Lecture Navigator API is running"}
