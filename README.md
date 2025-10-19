# Ikarus: AI-Powered Multi-Modal Furniture Search

![Ikarus Banner](https://placehold.co/1200x300/6366f1/ffffff?text=Ikarus%20AI%20Furniture%20Search&font=inter)

**Ikarus** is a sophisticated, full-stack application that redefines product discovery through a multi-modal, AI-powered search engine. Users can describe the furniture they're looking for in natural language, and Ikarus intelligently understands the query, applies relevant filters, and retrieves the most similar products by analyzing both text descriptions and product images.

The project combines a powerful Python backend (FastAPI, LangChain, Pinecone, Sentence Transformers) with a sleek, modern frontend (React, Material-UI).

## ✨ Key Features

- **🧠 Natural Language Understanding (NLU):** Leverages **LangChain** to parse complex, conversational queries (e.g., *"wooden chairs from vietnam under $200"*). The current implementation uses Google's **Flan-T5-Large** for a balance between performance and VRAM constraints.
- **🎨 Multi-Modal Embeddings:** Generates combined vector embeddings from both product text (fine-tuned `all-MiniLM-L6-v2`) and images (`CLIP-ViT-B-32`) for highly accurate, context-aware search results.
- **⚡ Blazing-Fast Vector Search:** Utilizes **Pinecone**'s serverless vector database for efficient, real-time similarity searches across a large product catalog.
- **🔧 Smart Filtering:** Automatically extracts and applies filters from the user's query, including **price range, category, material, brand, and country of origin**.
- **🤖 AI-Generated Descriptions:** Uses a smaller **Flan-T5-Small** model via LangChain to generate creative and engaging product descriptions on the fly.
- **📊 Interactive UI & Analytics:** A responsive and intuitive frontend built with **React** and **Material-UI**, featuring a chat-style search interface and a comprehensive analytics dashboard.
- **⚙️ End-to-End Pipeline:** Includes scripts for data cleaning, analysis, model fine-tuning, and embedding generation, providing a complete workflow from raw data to a deployed application.

## 🚀 Project Architecture

The application follows a modern, decoupled architecture:

1.  **Frontend (React App):** The user interacts with the chat interface. The natural language query is sent to the backend API.
2.  **Backend (FastAPI):**
    - The API receives the query.
    - **LangChain Chain:** The query is passed to a `LLMChain` which uses the `google/flan-t5-large` model to parse the text into a structured JSON object containing a `semantic_intent` and various filters (price, material, etc.).
    - **Sentence Transformers:** The `semantic_intent` is encoded into a vector embedding using the fine-tuned text model.
    - **Pinecone Query:** The backend queries the Pinecone index using the generated vector and the extracted metadata filters.
    - **Data Retrieval:** The IDs from the Pinecone results are used to look up full product details from the cleaned CSV.
    - **Response Generation:** The final product list, enriched with AI-generated descriptions, is sent back to the frontend.
3.  **Data & ML Pipeline (Offline Scripts):**
    - **`data_analytics.py`**: Cleans and prepares the initial `intern_data_ikarus.csv`.
    - **`multi_model.py`**:
        - Fine-tunes the Sentence Transformer model on product descriptions using triplet loss.
        - Generates multi-modal embeddings for all products.
        - Upserts the final vectors into the Pinecone index.



## 🤖 Model Configuration & Upgrades

This project is configured to be runnable on consumer-grade GPUs with moderate VRAM (e.g., 8-12 GB).

-   **Query Understanding:** `google/flan-t5-large` is used for parsing user queries. It provides a good baseline for extracting structured data from natural language but was primarily chosen to respect VRAM limitations.
-   **Description Generation:** `google/flan-t5-small` is used for its speed in generating short, creative descriptions.

### Upgrading for Enhanced Performance

For significantly better JSON parsing accuracy and more nuanced language understanding, it is **highly recommended to upgrade to a more powerful instruction-tuned model** if your hardware allows (e.g., >16GB VRAM). Models like **`mistralai/Mistral-7B-Instruct-v0.2`** or other 7B/8B parameter models are excellent choices.

To upgrade, you can modify the `model_id` in `backend/main.py`:

```python
# In backend/main.py, find this section:

# Original (Flan-T5)
query_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    # ... other parameters
)

# --- UPGRADE EXAMPLE (Mistral 8B) ---
# For models like Mistral, you might need to install `bitsandbytes` for quantization
# pip install bitsandbytes
query_llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2", # Or other powerful models
    task="text-generation",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 128,
        "torch_dtype": torch.bfloat16,
        # Optional: Use quantization for VRAM savings on larger models
        # "quantization_config": {"load_in_4bit": True},
    },
    device=device
)
