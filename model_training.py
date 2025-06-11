import chromadb
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from data_processing import load_coursera_data  # Import data loading function

# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
CHROMA_JOB_SKILLS_COLLECTION = "job_skills_embeddings"
CHROMA_COURSE_SKILLS_COLLECTION = "course_skills_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def get_embedding_model():
    """Loads and returns the sentence transformer model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
    return client

def initialize_chroma_db(df_coursera):
    """Initializes ChromaDB with skill embeddings."""
    chroma_client = get_chroma_client()

    # --- Job Skills ---
    print("Creating/Loading job skills collection...")
    job_skill_collection = chroma_client.get_or_create_collection(
        name=CHROMA_JOB_SKILLS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # --- Coursera Skills ---
    print("Creating/Loading course skills collection...")
    course_skill_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COURSE_SKILLS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return job_skill_collection, course_skill_collection

def train_model_rag(df_coursera, model, course_skill_collection):
    """Trains the Sentence Transformer model using RAG."""
    print("Preparing training data for RAG...")
    train_examples = []
    for index, row in df_coursera.iterrows():
        course_url = row.get('course_url', f"course_idx_{index}")
        course_title = row.get("Title", "N/A")
        canonical_skills = row.get("Canonical_Course_Skills", [])

        for skill in canonical_skills:
            if not skill:
                continue
            # Create InputExample: course_title and skill are related
            train_examples.append(InputExample(texts=[course_title, skill], label=1.0))

    print("Training the Sentence Transformer model with RAG...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)  # type: ignore

    # Check if CUDA is available and move the model to the GPU if it is
    if torch.cuda.is_available():
        model.to('cuda')  # type: ignore
        print("Model moved to GPU for training.")
    else:
        print("CUDA not available, training on CPU.")

    model.fit(  # type: ignore
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,  # Adjust as needed
        warmup_steps=100,  # Adjust as needed
    )
    return model

def populate_chroma_db(df_coursera, model, course_skill_collection):
    """Populates ChromaDB with Coursera skill embeddings (trained model)."""
    print("Populating ChromaDB with Coursera skill embeddings (trained model)...")
    embeddings_batch_course = []
    metadata_batch_course = []
    ids_batch_course = []

    for index, row in df_coursera.iterrows():
        course_url = row.get('course_url', f"course_idx_{index}")
        course_title = row.get("Title", "N/A")
        canonical_skills = row.get("Canonical_Course_Skills", [])

        for skill in canonical_skills:
            if not skill:
                continue
            skill_id = f"course_{str(course_url)}_{skill.replace(' ', '_')}"

            # Move the input to the same device as the model
            if torch.cuda.is_available():
                skill_embedding = model.encode(skill)  # type: ignore
            else:
                skill_embedding = model.encode(skill)  # type: ignore

            embeddings_batch_course.append(skill_embedding.tolist())  # Use the TRAINED model
            metadata_batch_course.append(
                {"course_url": str(course_url), "skill_name": skill, "course_title": course_title, "source": "coursera"})
            ids_batch_course.append(skill_id)

    if ids_batch_course:
        try:
            course_skill_collection.add(embeddings=embeddings_batch_course, metadatas=metadata_batch_course,
                                        ids=ids_batch_course)
            print(f"Added {len(ids_batch_course)} course skill embeddings to ChromaDB.")
        except Exception as e:
            print(f"Error adding course skill embeddings: {e}")

    print("ChromaDB skill population complete.")

if __name__ == "__main__":
    # 1. Load Coursera data
    df_coursera = load_coursera_data()
    if df_coursera is None:
        print("Failed to load Coursera data. Exiting.")
        exit()

    # 2. Initialize Sentence Transformer model
    model = get_embedding_model()
    if model is None:
        print("Failed to load embedding model. Exiting.")
        exit()

    # 3. Initialize ChromaDB
    job_skill_collection, course_skill_collection = initialize_chroma_db(df_coursera)

    # 4. Train the model using RAG
    trained_model = train_model_rag(df_coursera, model, course_skill_collection)

    # 5. Populate ChromaDB with the trained model
    populate_chroma_db(df_coursera, trained_model, course_skill_collection)

    print("Model training and ChromaDB population complete.")