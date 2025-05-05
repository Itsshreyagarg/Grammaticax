"""import joblib
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np

model = joblib.load("C:/Users/DELL/Documents/aes/api/saved_model/essay_scorer.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_tokens = len(tokens)
    num_sentences = len(list(doc.sents))
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / num_tokens if num_tokens > 0 else 0

    pos_counts = {
        "noun_count": sum(1 for token in doc if token.pos_ == "NOUN"),
        "verb_count": sum(1 for token in doc if token.pos_ == "VERB"),
        "adj_count": sum(1 for token in doc if token.pos_ == "ADJ"),
        "adv_count": sum(1 for token in doc if token.pos_ == "ADV"),
    }

    return np.array([
        num_tokens, num_sentences, type_token_ratio,
        pos_counts["noun_count"], pos_counts["verb_count"],
        pos_counts["adj_count"], pos_counts["adv_count"]
    ])

def predict_score(text):
    features = extract_features(text)
    embedding = embedding_model.encode([text])[0]
    final_input = np.concatenate([embedding, features])
    prediction = model.predict([final_input])[0]
    return prediction
"""
import joblib
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from textblob import TextBlob

# Load the pre-trained model and embeddings
model = joblib.load("C:/Users/DELL/Documents/aes/api/saved_model/essay_scorer.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# Function to extract features from the text using SpaCy
def extract_features(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_tokens = len(tokens)
    num_sentences = len(list(doc.sents))
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / num_tokens if num_tokens > 0 else 0

    pos_counts = {
        "noun_count": sum(1 for token in doc if token.pos_ == "NOUN"),
        "verb_count": sum(1 for token in doc if token.pos_ == "VERB"),
        "adj_count": sum(1 for token in doc if token.pos_ == "ADJ"),
        "adv_count": sum(1 for token in doc if token.pos_ == "ADV"),
    }

    return np.array([num_tokens, num_sentences, type_token_ratio,
                     pos_counts["noun_count"], pos_counts["verb_count"],
                     pos_counts["adj_count"], pos_counts["adv_count"]])

# Function to analyze text sentiment and grammar using TextBlob
def analyze_text(text):
    blob = TextBlob(text)
    
    # Sentiment analysis: Polarity (-1 to 1), Subjectivity (0 to 1)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    # Grammar score: Based on spelling and grammatical correctness
    grammar_score = 10 - len(blob.correct().string) / len(text) * 10 if len(text) > 0 else 0

    return sentiment_polarity, sentiment_subjectivity, grammar_score

# Function to predict essay score using the trained model
def predict_score(text):
    # Extract features from the essay
    features = extract_features(text)
    
    # Get embedding for the text using SentenceTransformers
    embedding = embedding_model.encode([text])[0]
    
    # Concatenate the features and embedding
    final_input = np.concatenate([embedding, features])
    
    # Predict the score using the trained model
    predicted_score = model.predict([final_input])[0]
    
    # Analyze sentiment and grammar using TextBlob
    sentiment_polarity, sentiment_subjectivity, grammar_score = analyze_text(text)
    
    # Content and structure scores can be static or dynamic based on model
    content_score = 9.6  # Example static content score
    structure_score = 9.0  # Example static structure score
    final_score = round((predicted_score + content_score + structure_score) / 3, 2)
    
    # Providing feedback based on sentiment polarity, grammar score, etc.
    feedback = "The essay has a solid structure and good vocabulary usage."
    if grammar_score < 7:
        feedback += " There are some grammatical issues that could be improved."
    elif sentiment_polarity < 0:
        feedback += " The tone of the essay could be more positive."
    elif sentiment_polarity > 0.2:
        feedback += " The tone is positive, good job on expressing your ideas."

    return {
        "final_score": final_score,
        "predicted_score": round(predicted_score, 2),
        "content_score": round(content_score, 2),
        "structure_score": round(structure_score, 2),
        "grammar_score": round(grammar_score, 2),
        "sentiment_polarity": round(sentiment_polarity, 2),
        "sentiment_subjectivity": round(sentiment_subjectivity, 2),
        "feedback": feedback
    }

