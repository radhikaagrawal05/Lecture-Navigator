from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.corpus import wordnet, stopwords
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse, parse_qs
import re
from collections import Counter

nltk.data.path.append("./nltk_data")
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', download_dir="./nltk_data")
    stop_words = set(stopwords.words('english'))

# Load the sentence-transformer model once for efficiency so embeddings are not recomputed per request.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
            for l in syn.lemmas():
                synonyms.add(l.name().replace('_', ' ').lower())
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
    # Extract keywords dynamically from the transcript text instead of relying on a fixed domain map.
    # This removes stopwords, ignores short words, and keeps the most frequent meaningful words.
    text = " ".join([t['text'] for t in transcript_list]).lower()
    words = re.findall(r'\b[a-z]{4,}\b', text)
    filtered = [w for w in words if w not in stop_words]
    return [word for word, _ in Counter(filtered).most_common(max_keywords)]


def get_related_transcript_keywords(transcript_keywords, query, min_score=0.35, top_n=10):
    if not transcript_keywords:
        return []

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    keyword_embeddings = embedding_model.encode(transcript_keywords, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, keyword_embeddings)[0].tolist()

    # Determine top transcript keywords that are semantically related to the query.
    # This enables domain-independent matching even when the exact query term is absent.
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

@app.post("/analyze")
def analyze_video(request: QueryRequest):
    video_id = extract_video_id(request.video_url)
    
    try:
        client = YouTubeTranscriptApi()
        transcript = client.fetch(video_id).to_raw_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not retrieve transcript: {str(e)}")

    expanded_keywords = get_expanded_concepts(request.concept)
    transcript_keywords = extract_frequent_keywords(transcript, max_keywords=40)
    sentence_texts = [item['text'].lower() for item in transcript]

    query_embedding = embedding_model.encode(request.concept, convert_to_tensor=True)
    sentence_embeddings = embedding_model.encode(sentence_texts, convert_to_tensor=True) if sentence_texts else None
    sentence_similarity_scores = util.cos_sim(query_embedding, sentence_embeddings)[0].tolist() if sentence_embeddings is not None else []
    related_keywords = get_related_transcript_keywords(transcript_keywords, request.concept)

    # First exact keyword matching using WordNet expansions, then semantic matching using embeddings.
    # This supports domain-independent identification of related concepts from transcript content.
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

        # If no exact or related transcript keyword was found, fall back to semantic similarity
        # between the user query and the full sentence. This captures broader concept matches.
        if not matched_words and sentence_similarity_scores:
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
