# ===============================================================
# IF YOU ARE WORKING IN IDE OTHER THAN UDACITY WORKSPACE, USE THIS CODE.
# As it uses the latest versions of the packages as of 13-11-2025.
#
# If you need to run this in the Udacity Workspace,
# please use the code provided in the
# `Personalized_Real_Estate_Agent.ipynb` file in this repository.
# ===============================================================


# Use this pip command in terminal to install all required packages
# pip install -r requirements.txt

import os
import json
import textwrap
from io import StringIO
from typing import Optional

import pandas as pd
from pydantic import BaseModel

# LangChain core
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS, Chroma

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# OpenAI API
from openai import OpenAI



# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://openai.vocareum.com/v1"

if not api_key:
    raise ValueError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=api_key, base_url=base_url)

print("OpenAI client initialized.\n")

# ========================================================
# STEP 1 — Generate Listings CSV Using LLM
# ========================================================

prompt = """
Generate 10 synthetic house listings in STRICT CSV format.
Headers: Neighborhood,Price,Bedrooms,Bathrooms,House Size,Description

RULES:
- Output ONLY CSV (header + 10 rows)
- All fields must be enclosed in double quotes
- No commas inside numeric values
- Description must NOT contain commas
- No commentary outside CSV
"""

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)

csv_text = resp.choices[0].message.content.strip()

with open("homes.csv", "w", encoding="utf-8") as f:
    f.write(csv_text)

df = pd.read_csv(StringIO(csv_text), quotechar='"', skipinitialspace=False)

print("=== STEP 1: DataFrame ===")
print(df.head(10))
print("Shape:", df.shape)
print("\n")

# ========================================================
# STEP 2 — Documents + Embeddings + Vector DB
# ========================================================

documents = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

for _, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    for chunk in splitter.split_text(content):
        documents.append(
            Document(
                page_content=chunk,
                metadata={col: row[col] for col in df.columns}
            )
        )

# Normalize numeric metadata
def fix_meta(docs):
    for d in docs:
        md = d.metadata
        for k in ["Price", "Bedrooms", "Bathrooms", "House Size"]:
            try:
                md[k] = int(str(md[k]).replace(",", "").replace('"', ""))
            except:
                md[k] = None
        d.metadata = md
    return docs

documents = fix_meta(documents)

# Embeddings selection
try:
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_base=base_url)
    print("Using OpenAI embeddings.")
except:
    embedding = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    print("Using HuggingFace embeddings.")

# Create FAISS vector DB
try:
    vector_db = FAISS.from_documents(documents, embedding)
    db_type = "FAISS"
except:
    # fallback to Chroma
    vector_db = Chroma.from_documents(
        documents,
        embedding,
        collection_name="homes_rag",
        persist_directory="./chroma_store"
    )
    db_type = "Chroma"

print("\n=== STEP 2: Vector DB Created:", db_type, "===")

# ========================================================
# STEP 3 — Extract User Preferences (LLM + Pydantic)
# ========================================================

class Prefs(BaseModel):
    budget: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    min_size: Optional[int] = None
    preferences: Optional[str] = None

print("\nDescribe what you want (ex: 3 bedrooms under 350000, 2 bathrooms).")
user_text = input("Your text:\n> ")

extract_prompt = f"""
Extract JSON from this request:

\"\"\"{user_text}\"\"\"

Output only this format:
{{
 "budget": int|null,
 "bedrooms": int|null,
 "bathrooms": int|null,
 "min_size": int|null,
 "preferences": string|null
}}
"""

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": extract_prompt}],
    temperature=0.0
)

json_raw = resp.choices[0].message.content.strip()

prefs_dict = json.loads(json_raw)
prefs = Prefs(**prefs_dict)

# Ask missing fields
missing = [f for f in prefs_dict if prefs_dict[f] is None and f != "preferences"]

if missing:
    print("\nMissing fields:", missing)
    for m in missing:
        val = input(f"Enter {m} (leave blank for none): ")
        if val.strip():
            try:
                setattr(prefs, m, int(val))
            except:
                setattr(prefs, m, None)

print("\n=== STEP 3: Final Preferences ===")
print(prefs.model_dump_json(indent=2))
# ========================================================
# STEP 4 — Semantic Search + Metadata Filtering
# ========================================================

query = ", ".join([
    f"budget {prefs.budget}" if prefs.budget else "",
    f"{prefs.bedrooms} bedrooms" if prefs.bedrooms else "",
    f"{prefs.bathrooms} bathrooms" if prefs.bathrooms else "",
    f"{prefs.min_size} sqft" if prefs.min_size else "",
    prefs.preferences or ""
]).strip(", ")

print("\nSemantic Query:", query)

results = vector_db.similarity_search(query, k=20)

filtered = []
for doc in results:
    md = doc.metadata
    ok = True
    if prefs.budget and md["Price"] > prefs.budget: ok = False
    if prefs.bedrooms and md["Bedrooms"] < prefs.bedrooms: ok = False
    if prefs.bathrooms and md["Bathrooms"] < prefs.bathrooms: ok = False
    if prefs.min_size and md["House Size"] < prefs.min_size: ok = False
    if ok:
        filtered.append(doc)

print("\n=== STEP 4: Filtered Matches ===")
if not filtered:
    print("No matches.")
else:
    for i, d in enumerate(filtered[:5], 1):
        print(f"\nMatch #{i}")
        for k, v in d.metadata.items():
            print(k + ":", v)

# ========================================================
# STEP 5 — Personalized Recommendation
# ========================================================

if filtered:
    best = filtered[0].metadata
    prompt = f"""
Rewrite this property into a short personalized recommendation.
Keep all facts exact. Use 1–2 emojis.

Property:
{json.dumps(best, indent=2)}

Buyer preferences:
{prefs.model_dump_json(indent=2)}

Return only the rewritten description.
"""

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    print("\n=== STEP 5: Personalized Recommendation ===")
    print(textwrap.fill(resp.choices[0].message.content.strip(), width=100))
else:
    print("No property to personalize.")
