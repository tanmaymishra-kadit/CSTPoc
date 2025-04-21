from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import chromadb
import json
import pandas as pd
import os
import requests
import http.client
import urllib.parse
from flask_cors import CORS
import transformers

print(" Running BACKEND app.py")

transformers.logging.set_verbosity_error()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    nlp = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1")
    print(" BioBERT model loaded successfully")
except Exception as e:
    print(f" Error loading BioBERT: {e}")

try:
    chroma_client = chromadb.PersistentClient(path="./chromadb_store")
    collection = chroma_client.get_or_create_collection(name="proteins")
    print(" ChromaDB initialized successfully")
except Exception as e:
    print(f" Error initializing ChromaDB: {e}")
    collection = None

def load_protein_data():
    try:
        if collection is None:
            #print("Skipping data load: ChromaDB collection not initialized.")
            return

        file_path = "data/protein_data.csv"
        if not os.path.exists(file_path):
            print(f" Error: '{file_path}' not found.")
            return

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        df.fillna("", inplace=True)
        df['synonyms'] = df['alt. names'].fillna("").astype(str).str.replace(', ', '|')

        existing_ids = set(collection.get()["ids"]) if collection else set()

        for _, row in df.iterrows():
            protein_id = row.get("uniprot", f"unknown_{_}")
            if protein_id in existing_ids:
               # print(f" Skipping duplicate ID: {protein_id}")
                continue
            protein = row.get("protein", "").strip()
            gene = row.get("gene", "").strip()
            synonyms = row.get("synonyms", "").strip()
            organism = row.get("organism", "Unknown").strip()

            print(f" Loading: gene={gene}, protein={protein}, uniprot={protein_id}, synonyms={synonyms}, organism={organism}")

            document_data = {
                "protein": protein,
                "gene": gene,
                "uniprot": protein_id,
                "synonyms": synonyms,
                "organism": organism
            }

            collection.add(
                documents=[json.dumps(document_data)],
                ids=[protein_id],
                metadatas=[{"protein": protein.lower()}]
            )

        print(" Protein data successfully loaded into ChromaDB")

    except Exception as e:
        print(f" Unexpected error while loading protein data: {e}")

load_protein_data()

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
                if query in item["gene"].lower() or any(query in syn.lower() for syn in item["synonyms"].split("|")):
                    item["source"] = "local"
                    results.append(item)
            except json.JSONDecodeError:
                print(" Error decoding JSON:", doc)

    return results

def fetch_from_hgnc(query):
    url = f"https://rest.genenames.org/fetch/symbol/{query.upper()}"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if docs:
                results = []
                for doc in docs[:5]:
                    results.append({
                        "gene": doc.get("symbol", ""),
                        "protein": doc.get("name", ""),
                        "uniprot": ", ".join(doc.get("uniprot_ids", [])),
                        "synonyms": doc.get("alias_symbol", []),
                        "organism": "Homo sapiens",
                        "source": "hgnc"
                    })
                return results
    except Exception as e:
        print(" HGNC fetch error:", e)
    return []

def fetch_from_uniprot(query):
    try:
        conn = http.client.HTTPSConnection("rest.uniprot.org")
        encoded_query = urllib.parse.quote(f"({query}) AND (reviewed:true)")
        url = f"/uniprotkb/search?fields=comment_count,feature_count,length,structure_3d,annotation_score,protein_existence,lit_pubmed_id,accession,organism_name,protein_name,gene_names,reviewed,keyword,id&query=%28{encoded_query}%29&size=5&format=json"
        conn.request("GET", url)
        res = conn.getresponse()
        data = res.read()
        results = []

        parsed = json.loads(data)
        for result in parsed.get("results", []):
            results.append({
                "gene": result.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
                "protein": result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "uniprot": result.get("primaryAccession", ""),
                "synonyms": [g.get("geneName", {}).get("value", "") for g in result.get("genes", [])],
                "structure_3d": result.get("structure_3d", ""),
                "organism": result.get("organism", {}).get("scientificName", "Unknown"),
                "source": "uniprot"
            })
        return results
    except Exception as e:
        print(" UniProt fetch error:", e)
        return []

@app.route("/search", methods=["POST"])
def search():
    print(" Received POST at /search")
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data.get("query", "").strip().lower()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = search_protein(query)

    if not results:
        print(" No match in local DB, checking HGNC...")
        results = fetch_from_hgnc(query)

    if not results:
        print(" No match in HGNC, checking UniProt...")
        results = fetch_from_uniprot(query)

        if not results:
            return jsonify({
                "query": query,
                "results": [],
                "message": "No matching results found in local DB, HGNC, or UniProt."
            })

    formatted_results = [
        {
            "protein": item.get("protein", ""),
            "gene": item.get("gene", ""),
            "uniprot": item.get("uniprot", ""),
            "synonyms": item.get("synonyms", []),
            "structure_3d": item.get("structure_3d", ""),
            "organism": item.get("organism", "Unknown"),
            "source": item.get("source", "")
        }
        for item in results
    ]

    return jsonify({
        "query": query,
        "results": formatted_results,
        "message": "Source indicates whether the result is from local ChromaDB, HGNC, or UniProt."
    })

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
