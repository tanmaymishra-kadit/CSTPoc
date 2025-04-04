from flask import Flask, request, jsonify
from transformers import pipeline
import chromadb
import json
import pandas as pd
import os
from flask_cors import CORS
import transformers

# Suppress BioBERT warnings
transformers.logging.set_verbosity_error()

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Load BioBERT for Named Entity Recognition (NER)
try:
    nlp = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")
    print(" BioBERT model loaded successfully")
except Exception as e:
    print(f" Error loading BioBERT: {e}")

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./chromadb_store")
    collection = chroma_client.get_or_create_collection(name="proteins")
    print(" ChromaDB initialized successfully")
except Exception as e:
    print(f" Error initializing ChromaDB: {e}")
    collection = None

# Loading Protein Data from CSV and Insert into ChromaDB
def load_protein_data():
    try:
        file_path = "data/protein_data.csv"
        
        
        if not os.path.exists(file_path): # Checking if file exists
            print(f" Error: '{file_path}' not found.")
            return

        df = pd.read_csv(file_path)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        df.fillna("", inplace=True)

        # Ensure 'alt. names' exists before transformation
        df['synonyms'] = df['alt. names'].fillna("").astype(str).str.replace(', ', '|')

        # Fetch existing IDs from ChromaDB
        existing_ids = set(collection.get()["ids"]) if collection else set()

        for _, row in df.iterrows():
            protein_id = row.get("uniprot", f"unknown_{_}")

            if protein_id in existing_ids:
                print(f" Skipping duplicate ID: {protein_id}")
                continue  # Skip if the ID already exists

            document_data = {
                "protein": row.get("protein", "Unknown"),
                "gene": row.get("gene", "Unknown"),
                "uniprot": protein_id,
                "synonyms": row.get("synonyms", ""),
            }

            collection.add(
                documents=[json.dumps(document_data)],
                ids=[protein_id],
                metadatas=[{"protein": row.get("protein", "").lower()}]
            )

        print(" Protein data successfully loaded into ChromaDB")

    except Exception as e:
        print(f" Unexpected error while loading protein data: {e}")

# Load data on startup
load_protein_data()

# Search Function using NER and ChromaDB
def search_protein(query):
    query = query.lower()
    results = []

    if not collection:
        print(" ChromaDB collection not available.")
        return []

    matches = collection.query(query_texts=[query], n_results=10)

    if "documents" in matches and matches["documents"]:
        for doc in matches["documents"][0]:
            try:
                item = json.loads(doc)
                # Check if query matches gene name or any synonym
                if query in item["gene"].lower() or any(query in syn.lower() for syn in item["synonyms"].split("|")):
                    results.append(item)
            except json.JSONDecodeError:
                print(" Error decoding JSON:", doc)

    return results

# Flask API Endpoint
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip().lower()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = search_protein(query)

    if not results:
        return jsonify({
            "query": query,
            "results": [],
            "message": "Sorry, no matching results found in our database."
        })

    formatted_results = [
        {
            "protein": item.get("protein", ""),
            "gene": item.get("gene", ""),
            "uniprot": item.get("uniprot", ""),
            "synonyms": item.get("synonyms", "").split("|")
        }
        for item in results
    ]

    return jsonify({
        "query": query,
        "results": formatted_results
    })

# Health Check Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Protein Search API is running"})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5000)
