import os
import re
import json
import ast
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI(
    title="AI-Powered Furniture Search API",
    version="15.0.0",
    description="Advanced furniture search with LangChain-powered natural language understanding"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()

# Device configuration
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 0 if device_str == 'cuda' else -1
print(f"✅ Using device: {device_str}")

# Similarity threshold for quality matches
SIMILARITY_THRESHOLD = 0.23

print("\n📂 Loading dataset...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  

DEFAULT_DATA_PATH = os.path.join(PARENT_DIR, "cleaned_intern_data.csv")
DEFAULT_MODEL_PATH = os.path.join(PARENT_DIR, "fine_tuned_furniture_model")

print(f"📁 Script directory: {SCRIPT_DIR}")
print(f"📁 Parent directory: {PARENT_DIR}")
print(f"📁 Looking for data at: {DEFAULT_DATA_PATH}")

try:
    DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    print(f"📂 Attempting to load from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=['uniq_id'])
    df['price'] = df['price'].fillna('').astype(str)
    text_columns = ['brand', 'manufacturer', 'country_of_origin', 'material', 'categories', 'title', 'description']
    for col in text_columns:
        if col not in df.columns:
            df[col] = 'N/A'
        df[col] = df[col].fillna('unknown').astype(str).str.lower()
    df['categories_list'] = df['categories'].apply(
        lambda x: ast.literal_eval(x) if x not in ['unknown', '[]'] and x.startswith('[') else []
    )
    df['main_category'] = df['categories_list'].apply(
        lambda cats: cats[-1].lower() if cats else 'unknown'
    )
    
    df['price_numeric'] = pd.to_numeric(df['price_cleaned'], errors='coerce')
    
    df.set_index('uniq_id', inplace=True)
    
    print(f"✅ Dataset loaded: {len(df)} products")
    
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at {DATA_PATH}")
    df = None
    exit(1)

# LOAD AI MODELS

print("\n🤖 Loading AI models...")

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
print(f"🤖 Loading text model from: {MODEL_PATH}")

text_model = SentenceTransformer(MODEL_PATH, device=device_str)
text_embedding_dim = text_model.get_sentence_embedding_dimension()

image_model = SentenceTransformer('clip-ViT-B-32', device=device_str)
image_embedding_dim = 512

print(f"✅ Text embedding dimension: {text_embedding_dim}")
print(f"✅ Image embedding dimension: {image_embedding_dim}")


# LANGCHAIN SETUP WITH GOOGLE FLAN-T5 preffered a larger model like 8b mistral for more json understanding

print("\n🧠 Loading LangChain with Google Flan-T5...")

# Query understanding LLM (Large model for accuracy)
query_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0.0,
        "max_length": 128
    },
    pipeline_kwargs={
        "max_new_tokens": 100  
    },
    device=device
)

# Description generation LLM 
desc_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0.7,  
        "max_length": 64
    },
    device=device
)


# Ultra-compact query analysis prompt (< 200 tokens for Flan-T5-large)
query_analysis_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Extract JSON from furniture query. Use "any" if not mentioned, "none" for no price.

Examples:
"chair" → {{"category":"chair","material":"any","price_min":"none","price_max":"none","brand":"any","country_of_origin":"any","semantic_intent":"chair"}}
"wooden table vietnam under 300" → {{"category":"table","material":"wood","price_min":"none","price_max":"300","brand":"any","country_of_origin":"vietnam","semantic_intent":"wooden table vietnam"}}
"foam chair china under 150" → {{"category":"chair","material":"foam","price_min":"none","price_max":"150","brand":"any","country_of_origin":"china","semantic_intent":"foam chair china"}}
"sofa between 200 500" → {{"category":"sofa","material":"any","price_min":"200","price_max":"500","brand":"any","country_of_origin":"any","semantic_intent":"sofa"}}

Query: {query}
JSON:"""
)

query_chain = LLMChain(llm=query_llm, prompt=query_analysis_prompt)

desc_prompt = PromptTemplate(
    input_variables=["title", "original_description"],
    template="Generate a creative and appealing one-sentence product description for '{title}'. Additional info: '{original_description}'. Make it engaging and highlight key features."
)

desc_chain = LLMChain(llm=desc_llm, prompt=desc_prompt)

print("✅ LangChain chains created successfully")

# PINECONE SETUP

print("\n📊 Connecting to Pinecone...")
PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY",
    "pcsk_5pv4yK_7egZVBomhiC2qKLmyGjFjpivFrfG92HfHkSiZ1Z1PWXvVQv83U1seg9SZ2ZJCx6"
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("product-recommendations")
print("✅ Pinecone connected")

class QueryRequest(BaseModel):
    query: str

class Product(BaseModel):
    id: str
    title: str
    price: str
    image_url: str
    generated_description: str
    material: str
    country_of_origin: str
    brand: str
    category: str
    similarity_score: float

class RecommendationResponse(BaseModel):
    products: List[Product]
    query_understanding: Dict[str, Any]
    filters_applied: Dict[str, Any]
    total_found: int

class AnalyticsResponse(BaseModel):
    brand_counts: Dict[str, int]
    category_counts: Dict[str, int]
    material_counts: Dict[str, int]
    country_counts: Dict[str, int]
    price_stats: Dict[str, float]
    total_products: int

def parse_query_with_langchain(query: str):
    """
    Use LangChain with Google Flan-T5 to understand natural language queries
    and extract structured filters
    """
    try:
        # Use invoke() instead of deprecated run()
        result = query_chain.invoke({"query": query})
        ai_response = result.get('text', '').strip()
        print(f"🤖 LangChain AI Response: {ai_response}")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            print("⚠️ No JSON found in AI response, using fallback")
            parsed = {
                "semantic_intent": query,
                "category": "any",
                "material": "any",
                "price_min": "none",
                "price_max": "none",
                "brand": "any",
                "country_of_origin": "any"
            }
        
        print(f"📋 Parsed Query Understanding: {json.dumps(parsed, indent=2)}")
        
    except Exception as e:
        print(f"⚠️ LangChain parsing failed: {e}. Using fallback.")
        parsed = {
            "semantic_intent": query,
            "category": "any",
            "material": "any",
            "price_min": "none",
            "price_max": "none",
            "brand": "any",
            "country_of_origin": "any"
        }
    
    # Check if query is about non-furniture items
    if parsed.get('category') == 'unknown':
        print("🛑 Query detected as NON-FURNITURE item. Stopping search.")
        return None, None, parsed
    
    # Build Pinecone filters from parsed data
    filters = []
    
    # Mapping of parsed fields to Pinecone metadata fields
    filter_mapping = {
        'category': 'main_category',
        'brand': 'brand',
        'material': 'material',
        'country_of_origin': 'country_of_origin'
    }
    
    for parsed_key, pinecone_field in filter_mapping.items():
        value = parsed.get(parsed_key, 'any')
        if value not in ['any', 'unknown', 'none', '']:
            filters.append({pinecone_field: {"$eq": value.lower()}})
    
    # Handle price filtering (CRITICAL: Use price_numeric)
    price_filter = {}
    try:
        # Min price
        if parsed.get('price_min') not in ['none', '', None]:
            price_val = float(re.sub(r'[^\d.]', '', str(parsed['price_min'])))
            price_filter['$gte'] = price_val
            print(f"💰 Min Price Filter: ${price_val}")
        
        # Max price
        if parsed.get('price_max') not in ['none', '', None]:
            price_val = float(re.sub(r'[^\d.]', '', str(parsed['price_max'])))
            price_filter['$lte'] = price_val
            print(f"💰 Max Price Filter: ${price_val}")
    
    except (ValueError, TypeError) as e:
        print(f"⚠️ Price parsing error: {e}")
    
    # Add price filter if exists
    if price_filter:
        filters.append({"price_numeric": price_filter})
    
    # Build final Pinecone filter
    pinecone_filter = {"$and": filters} if filters else {}
    
    return parsed.get('semantic_intent', query), pinecone_filter, parsed

def extract_price_from_text(text: str) -> str:
    """Extract price value from text string"""
    if pd.isna(text) or text == '':
        return "N/A"
    match = re.search(r'[\$]?[\d,]+\.?\d*', text)
    if match and match.group().strip('$') != '':
        return f"${match.group().replace(',', '')}"
    return "N/A"

def generate_creative_description(title: str, original_desc: str) -> str:
    """Generate creative product description using LangChain"""
    try:
        # Use invoke() instead of deprecated run()
        result = desc_chain.invoke({
            "title": title,
            "original_description": original_desc[:200]  # Limit input length
        })
        generated = result.get('text', '').strip()
        return generated if generated else original_desc[:150]
    except Exception as e:
        print(f"⚠️ Description generation failed: {e}")
        return original_desc[:150]


@app.get("/")
def read_root():
    """Root endpoint - API information"""
    return {
        "message": "AI-Powered Furniture Search API with LangChain",
        "version": "15.0.0",
        "ai_powered": True,
        "langchain_enabled": True,
        "features": [
            "Natural language query understanding with LangChain",
            "Google Flan-T5 language model integration",
            "Price range filtering (under, over, between)",
            "Country of origin filtering",
            "Material filtering",
            "Brand filtering",
            "Category detection",
            "Semantic similarity search",
            "AI-generated product descriptions",
            "Similarity threshold filtering"
        ],
        "example_queries": [
            "wooden table from vietnam under 300",
            "foam chairs from china under 150",
            "leather sofa between 500 and 1000",
            "metal furniture under 200 made in india",
            "ikea chairs",
            "glass coffee table below 200",
            "steel bed frame over 300 from usa"
        ],
        "endpoints": {
            "POST /recommend": "Search for furniture with natural language",
            "GET /analytics": "Get product statistics",
            "GET /health": "Check API health"
        }
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_products(request: QueryRequest):
    """
    Main recommendation endpoint with LangChain-powered AI understanding
    
    Natural language examples:
    - "foam chairs under 150 from china"
    - "wooden table from vietnam under 300"
    - "leather sofa between 500 and 1000"
    - "metal furniture under 200 made in india"
    - "show me glass tables below 200"
    """
    if index is None or df is None:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    print(f"\n{'='*80}")
    print(f"🔍 NEW SEARCH REQUEST: {request.query}")
    print(f"{'='*80}")
    
    # Parse query using LangChain
    semantic_query, pinecone_filter, query_analysis = parse_query_with_langchain(request.query)
    
    # Check if query is about non-furniture
    if semantic_query is None:
        print("❌ Non-furniture query detected. Returning empty results.")
        return RecommendationResponse(
            products=[],
            query_understanding=query_analysis,
            filters_applied={},
            total_found=0
        )
    
    print(f"\n🎯 Semantic Intent: {semantic_query}")
    print(f"⚙️ Pinecone Filters: {json.dumps(pinecone_filter, indent=2)}")
    
    # Generate embedding for semantic search
    text_embedding = text_model.encode(semantic_query, convert_to_numpy=True)
    image_embedding = np.zeros(image_embedding_dim)
    query_embedding = np.concatenate([text_embedding.flatten(), image_embedding]).tolist()
    
    # Search in Pinecone with filters
    query_results = index.query(
        vector=query_embedding,
        top_k=20,
        filter=pinecone_filter,
        include_metadata=True
    )
    
    # Apply similarity threshold
    valid_matches = []
    print(f"\n📊 Similarity Filtering (Threshold: {SIMILARITY_THRESHOLD}):")
    for match in query_results['matches']:
        score = match['score']
        is_valid = score >= SIMILARITY_THRESHOLD
        status = "✅ VALID" if is_valid else "❌ FILTERED"
        print(f"  {status} | ID: {match['id'][:25]}... | Score: {score:.4f}")
        if is_valid:
            valid_matches.append(match)
    
    print(f"\n✅ {len(valid_matches)} products passed similarity threshold")
    
    if not valid_matches:
        print("⚠️ No products met the similarity threshold")
        return RecommendationResponse(
            products=[],
            query_understanding=query_analysis,
            filters_applied=pinecone_filter,
            total_found=0
        )
    
    # Build product response
    products = []
    for match in valid_matches:
        product_id = match['id']
        
        if product_id not in df.index:
            print(f"⚠️ Product {product_id} not found in dataset")
            continue
        
        product_data = df.loc[product_id]
        if isinstance(product_data, pd.DataFrame):
            product_data = product_data.iloc[0]
        
        # Extract image URL
        image_url_str = str(product_data.get("images", "[]"))
        try:
            image_list = ast.literal_eval(image_url_str) if image_url_str.startswith('[') else []
            first_url = image_list[0] if image_list else ""
        except:
            first_url = ""
        
        # Get product details
        title = str(product_data.get("title", "N/A"))
        
        # Use price_cleaned for display (CRITICAL FIX)
        price_val = product_data.get("price_cleaned", "N/A")
        if pd.notna(price_val) and str(price_val) != "N/A":
            price = f"${price_val}"
        else:
            price = "N/A"
        
        desc = str(product_data.get("description", ""))
        
        # Generate creative description using LangChain
        creative_desc = generate_creative_description(title, desc)
        
        # Create product object
        product_obj = Product(
            id=product_id,
            title=title,
            price=price,
            image_url=first_url,
            generated_description=creative_desc,
            material=str(product_data.get("material", "unknown")).title(),
            country_of_origin=str(product_data.get("country_of_origin", "unknown")).title(),
            brand=str(product_data.get("brand", "unknown")).title(),
            category=str(product_data.get("main_category", "unknown")).title(),
            similarity_score=round(float(match['score']), 4)
        )
        products.append(product_obj)
    
    # Sort: products with price first, then without price
    products_with_price = [p for p in products if p.price != "N/A"]
    products_without_price = [p for p in products if p.price == "N/A"]
    final_products = products_with_price + products_without_price
    
    print(f"\n✅ Returning {len(final_products)} products")
    
    return RecommendationResponse(
        products=final_products,
        query_understanding=query_analysis,
        filters_applied=pinecone_filter,
        total_found=len(final_products)
    )

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get comprehensive product analytics and statistics"""
    if df is None:
        raise HTTPException(status_code=503, detail="Product data not available")
    
    # Top brands
    brand_counts = df[df['brand'] != 'unknown']['brand'].value_counts().nlargest(10).to_dict()
    
    # Top categories
    category_counts = df[df['main_category'] != 'unknown']['main_category'].value_counts().nlargest(10).to_dict()
    
    # Top materials
    material_counts = df[df['material'] != 'unknown']['material'].value_counts().nlargest(10).to_dict()
    
    # Top countries
    country_counts = df[df['country_of_origin'] != 'unknown']['country_of_origin'].value_counts().nlargest(10).to_dict()
    
    # Price statistics (using price_numeric)
    numeric_prices = df['price_numeric'].dropna()
    price_stats = {
        'mean': round(numeric_prices.mean(), 2) if not numeric_prices.empty else 0,
        'median': round(numeric_prices.median(), 2) if not numeric_prices.empty else 0,
        'max': round(numeric_prices.max(), 2) if not numeric_prices.empty else 0,
        'min': round(numeric_prices.min(), 2) if not numeric_prices.empty else 0,
        'count': len(numeric_prices)
    }
    
    return AnalyticsResponse(
        brand_counts=brand_counts,
        category_counts=category_counts,
        material_counts=material_counts,
        country_counts=country_counts,
        price_stats=price_stats,
        total_products=len(df)
    )

@app.get("/health")
def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "version": "15.0.0",
        "langchain_enabled": True,
        "models_loaded": {
            "text_model": text_model is not None,
            "image_model": image_model is not None,
            "query_llm": query_llm is not None,
            "desc_llm": desc_llm is not None,
            "query_chain": query_chain is not None,
            "desc_chain": desc_chain is not None
        },
        "services": {
            "pinecone_connected": index is not None,
            "data_loaded": df is not None,
            "total_products": len(df) if df is not None else 0
        },
        "ai_features": {
            "natural_language_understanding": True,
            "langchain_integration": True,
            "google_flan_t5": True,
            "semantic_search": True,
            "smart_filtering": True,
            "description_generation": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("     AI-POWERED FURNITURE SEARCH API WITH LANGCHAIN")
    print("="*80)
    print("\n🚀 Features:")
    print("   ✅ LangChain integration")
    print("   ✅ Google Flan-T5 large language model")
    print("   ✅ Natural language query understanding")
    print("   ✅ Price filtering (under, over, between)")
    print("   ✅ Country of origin filtering")
    print("   ✅ Material filtering")
    print("   ✅ Brand filtering")
    print("   ✅ Semantic similarity search")
    print("   ✅ AI-generated descriptions")
    print("   ✅ Similarity threshold filtering")
    print("\n📡 Starting server on http://0.0.0.0:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
