from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Configuration
EMBEDDING_DIM = 200  # Dimension of the embeddings
VOCAB_SIZE = 20000   # Size of vocabulary to use
MODEL_PATH = "tfidf_model.pkl"  # Path to save/load the model

# Initialize or load the model
if os.path.exists(MODEL_PATH):
    print(f"Loading existing TF-IDF model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        vectorizer = model_data['vectorizer']
        svd = model_data.get('svd')
else:
    print("Creating new TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        max_features=VOCAB_SIZE,
        stop_words='english',
        lowercase=True,
        analyzer='word'
    )
    svd = None  # Will be initialized during training

# Sample texts for initial fitting if needed
sample_texts = [
    "This is a sample document for TF-IDF initialization.",
    "Another example text to build initial vocabulary.",
    "The vectorizer needs some text to establish a vocabulary."
]

def train_model(texts):
    """Train or update the TF-IDF model with new texts."""
    global vectorizer, svd
    
    # Fit or transform with vectorizer
    if not hasattr(vectorizer, 'vocabulary_'):
        print("Fitting vectorizer on initial texts")
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        print("Transforming texts with existing vectorizer")
        tfidf_matrix = vectorizer.transform(texts)
    
    # Initialize or update SVD for dimensionality reduction
    if svd is None:
        print(f"Initializing SVD for reduction to {EMBEDDING_DIM} dimensions")
        svd = TruncatedSVD(n_components=EMBEDDING_DIM)
        svd.fit(tfidf_matrix)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'svd': svd}, f)
    
    print("Model saved successfully")

# Initialize the model if it doesn't exist
if not hasattr(vectorizer, 'vocabulary_'):
    train_model(sample_texts)

@app.route('/embed', methods=['POST'])
def embed_text():
    """Endpoint to generate embeddings for text."""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "Request must include 'text' field"}), 400
    
    text = data['text']
    
    # Handle both single texts and lists of texts
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Generate TF-IDF vectors
    tfidf_vectors = vectorizer.transform(texts)
    
    # Apply dimensionality reduction
    if svd is not None:
        reduced_vectors = svd.transform(tfidf_vectors)
        # Normalize vectors
        embeddings = [vec / np.linalg.norm(vec) for vec in reduced_vectors]
    else:
        # If SVD not available, use sparse vectors directly (not recommended)
        embeddings = [vec.toarray()[0] for vec in tfidf_vectors]
        embeddings = [vec / np.linalg.norm(vec) for vec in embeddings]
    
    # Return appropriate response format
    if is_single:
        return jsonify({"embedding": embeddings[0].tolist()})
    else:
        return jsonify({"embeddings": [emb.tolist() for emb in embeddings]})

@app.route('/train', methods=['POST'])
def update_model():
    """Endpoint to update the model with new texts."""
    data = request.json
    
    if not data or 'texts' not in data:
        return jsonify({"error": "Request must include 'texts' field"}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' field must be a list of strings"}), 400
    
    try:
        train_model(texts)
        return jsonify({"status": "Model updated successfully"})
    except Exception as e:
        return jsonify({"error": f"Error updating model: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def model_status():
    """Endpoint to check the model status."""
    vocabulary_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 0
    return jsonify({
        "status": "ready",
        "embedding_dimension": EMBEDDING_DIM,
        "vocabulary_size": vocabulary_size,
        "using_svd": svd is not None
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)
