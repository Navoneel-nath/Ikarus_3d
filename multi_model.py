import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pinecone
from pinecone import ServerlessSpec
import requests
from PIL import Image
import torch
import os

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# load dataset
try:
    df = pd.read_csv('/content/cleaned_intern_data.csv')
    print("Cleaned dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'cleaned_intern_data.csv' not found. Please run the Data Analytics notebook first.")
    df = pd.DataFrame()

if not df.empty:
    df.dropna(subset=['categories', 'description'], inplace=True)
    df_sampled = df.groupby('categories').apply(lambda x: x.sample(min(len(x), 50))).reset_index(drop=True)
    print(f"Working with a sampled dataset of {len(df_sampled)} items for triplet generation.")

    train_examples = []
    category_map = df_sampled.groupby('categories')['description'].apply(list).to_dict()
    categories_list = list(category_map.keys())

    print("Generating training triplets...")
    for index, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
        anchor_text = row['description']
        anchor_category = row['categories']

        if len(category_map[anchor_category]) > 1:
            possible_positives = [text for text in category_map[anchor_category] if text != anchor_text]
            if possible_positives:
                positive_text = np.random.choice(possible_positives)
                negative_category = np.random.choice([cat for cat in categories_list if cat != anchor_category])
                negative_text = np.random.choice(category_map[negative_category])
                train_examples.append(InputExample(texts=[anchor_text, positive_text, negative_text]))

    print(f"Generated {len(train_examples)} triplets.")


# fine tuning text model
if not df.empty and train_examples:
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name, device=device)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.TripletLoss(model=model)

    num_epochs = 1
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    print("Starting model fine-tuning...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path='./fine_tuned_furniture_model',
              show_progress_bar=True)
    print("Model fine-tuning complete. Model saved to './fine_tuned_furniture_model'.")
else:
    print("Skipping model fine-tuning because no training triplets were generated.")


# embeddings for multi modal and pinecone setup
if not df.empty:
    # Pinecone 
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5pv4yK_7egZVBomhiC2qKLmyGjFjpivFrfG92HfHkSiZ1Z1PWXvVQv83U1seg9SZ2ZJCx6")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Error initializing Pinecone. Please check your API key. Error: {e}")
        pc = None

    if pc:
        index_name = "product-recommendations"

        # dimensions calculiations
        if index_name in [index_info["name"] for index_info in pc.list_indexes()]:
            print(f"Deleting pre-existing index '{index_name}' to ensure correct dimensions.")
            pc.delete_index(index_name)

        if os.path.exists('./fine_tuned_furniture_model'):
            text_model = SentenceTransformer('./fine_tuned_furniture_model', device=device)
            print("Loaded fine-tuned text model.")
        else:
            text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print("Fine-tuned model not found. Using the base 'all-MiniLM-L6-v2' model.")

        try:
            image_model = SentenceTransformer('clip-ViT-B-32', device=device)
            print("Loaded CLIP image model.")
        except Exception as e:
            print(f"Error loading CLIP image model: {e}")
            image_model = None

        text_embedding_dim = text_model.get_sentence_embedding_dimension()
        image_embedding_dim = image_model.get_sentence_embedding_dimension() if image_model else 0

        if text_embedding_dim is None: text_embedding_dim = 384
        if image_embedding_dim is None: image_embedding_dim = 512

        vector_dim = text_embedding_dim + image_embedding_dim

        print(f"Text embedding dimension: {text_embedding_dim}")
        print(f"Image embedding dimension: {image_embedding_dim}")
        print(f"Combined vector dimension: {vector_dim}")

        if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
            print(f"Creating new Pinecone serverless index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=vector_dim,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        index = pc.Index(index_name)
        print("Pinecone setup complete.")

        # generate embeddings and upload
        batch_size = 64
        print("Generating and uploading embeddings to Pinecone...")
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]

            text_corpus = (batch_df['title'].fillna('') + ". " + batch_df['description'].fillna('')).tolist()
            text_embeddings = text_model.encode(text_corpus, convert_to_tensor=True, show_progress_bar=False)

            if image_model and image_embedding_dim > 0:
                image_embeddings_list = []
                for img_str in batch_df['images'].fillna(''):
                    image_urls = img_str.split(',')
                    if image_urls and image_urls[0]:
                        try:
                            image = Image.open(requests.get(image_urls[0], stream=True, timeout=5).raw).convert("RGB")
                            img_embedding = image_model.encode(image, convert_to_tensor=True, show_progress_bar=False)
                            image_embeddings_list.append(img_embedding)
                        except Exception:
                            image_embeddings_list.append(torch.zeros(image_embedding_dim, device=device))
                    else:
                        image_embeddings_list.append(torch.zeros(image_embedding_dim, device=device))
                image_embeddings = torch.stack(image_embeddings_list)
                combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1).cpu().numpy()
            else:
                combined_embeddings = text_embeddings.cpu().numpy()

            upserts = [
                {"id": row['uniq_id'], "values": combined_embeddings[idx].tolist()}
                for idx, row in enumerate(batch_df.to_dict('records'))
            ]

            if upserts:
                index.upsert(vectors=upserts)

        print("Embedding generation and upload complete.")
        print(index.describe_index_stats())

if not df.empty:
    def search_similarity(query, model, k=3, data_sample=df_sampled):
        query_embedding = model.encode(query)
        corpus_embeddings = model.encode(data_sample['description'].fillna('').tolist())
        similarities = util.cos_sim(query_embedding, corpus_embeddings)
        top_k_indices = torch.topk(similarities, k=k, dim=1).indices[0]
        return data_sample.iloc[top_k_indices]['title'].tolist()

    original_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    if os.path.exists('./fine_tuned_furniture_model'):
        fine_tuned_model = SentenceTransformer('./fine_tuned_furniture_model', device=device)
    else:
        fine_tuned_model = original_model

    search_query = "a comfortable wooden chair for the living room"

    print(f"\n--- Search Results for: '{search_query}' (ORIGINAL Model) ---")
    for res in search_similarity(search_query, original_model):
        print(f"- {res}")

    print(f"\n--- Search Results for: '{search_query}' (FINE-TUNED Model) ---")
    for res in search_similarity(search_query, fine_tuned_model):
        print(f"- {res}")

    print("\n--- Model Training & Evaluation Complete ---")
