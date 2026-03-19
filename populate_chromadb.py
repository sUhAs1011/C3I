import pandas as pd
from pathlib import Path
import numpy as np
import re
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords')


# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"

# ChromaDB and embedding settings
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
CHROMA_JOB_SKILLS_COLLECTION = "job_skills_embeddings"
CHROMA_COURSE_SKILLS_COLLECTION = "course_skills_embeddings"
CHROMA_JOBS_COLLECTION = "jobs_embeddings"
CHROMA_COURSES_COLLECTION = "courses_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Global model variable to avoid re-loading
embedding_model = None

COURSE_CSVS = [
    "coursera1.csv",
    "coursera2.csv",
    "coursera3.csv",
    "coursera_data_analyst.csv",
    "coursera_hardware.csv",
    "coursera_info_sec.csv",
    "coursera_product.csv",
    "coursera_project.csv",
    "coursera_software.csv",
    "coursera_cyber_security.csv",
    "coursera_ui_ux.csv",
    "coursera_web_developer.csv"
]

JOB_CSVS = [
    "data_science_analytics.csv",
    "engineering_hardware_networks.csv",
    "engineering_software_qa.csv",
    "it_information_security.csv",
    "project_program_management.csv",
    "product_management.csv",
    "research_development.csv",
    "ux_design_architecture.csv",
    "postings.csv"
]

def get_embedding_model():
    """Loads and returns the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            embedding_model = None
    return embedding_model

def get_chroma_client():
    """Initializes and returns the ChromaDB client."""
    print(f"Initializing ChromaDB client at {CHROMA_DATA_PATH}...")
    return chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))

def initialize_chroma_collections():
    """Initialize ChromaDB collections for jobs and courses."""
    client = get_chroma_client()
    
    collections = {
        "jobs": client.get_or_create_collection(
            name=CHROMA_JOBS_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        ),
        "courses": client.get_or_create_collection(
            name=CHROMA_COURSES_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        ),
        "job_skills": client.get_or_create_collection(
            name=CHROMA_JOB_SKILLS_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        ),
        "course_skills": client.get_or_create_collection(
            name=CHROMA_COURSE_SKILLS_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    }
    
    print("ChromaDB collections initialized successfully.")
    return collections

def create_job_embedding_text(job_row):
    """Create a more descriptive and structured text representation of a job for embedding."""
    parts = []
    
    # Title is the most important
    title = ""
    title_cols = ['jobtitle', 'job_title', 'business_title', 'title', 'Title']
    for col in title_cols:
        if col in job_row and pd.notna(job_row[col]):
            title = str(job_row[col]).strip()
            if title:
                parts.append(f"Job Title: {title}")
                break
    
    # Company
    company = ""
    company_cols = ['company', 'Company', 'organization', 'Organization', 'agency', 'Agency']
    for col in company_cols:
        if col in job_row and pd.notna(job_row[col]):
            company = str(job_row[col]).strip()
            if company:
                parts.append(f"Company: {company}")
                break

    # Skills are crucial for matching
    skills = ""
    skills_cols = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
    for col in skills_cols:
        if col in job_row and pd.notna(job_row[col]):
            skills_text = str(job_row[col]).strip()
            if skills_text and skills_text.lower() != 'no skills specified':
                skills = skills_text
                parts.append(f"Key Skills: {skills}")
                break
    
    # Description for context
    description = ""
    desc_cols = ['jobdescription', 'job_description', 'description', 'Description']
    for col in desc_cols:
        if col in job_row and pd.notna(job_row[col]):
            desc_text = str(job_row[col]).strip()
            if desc_text and desc_text.lower() != 'no description available':
                # Truncate to a reasonable length
                description = (desc_text[:400] + '...') if len(desc_text) > 400 else desc_text
                parts.append(f"Description: {description}")
                break

    # Combine into a single, structured string
    if not parts:
        return ""
    
    return ". ".join(parts)

def create_course_embedding_text(course_row):
    """Create a more descriptive and structured text representation of a course for embedding."""
    parts = []
    
    # Course Title
    title = ""
    title_cols = ['title', 'Title', 'course_title', 'Course Title', 'name', 'Name']
    for col in title_cols:
        if col in course_row and pd.notna(course_row[col]):
            title = str(course_row[col]).strip()
            if title:
                parts.append(f"Course: {title}")
                break
    
    # Organization/Provider
    organization = ""
    org_cols = ['organization', 'Organization', 'instructor', 'Instructor', 'provider', 'Provider']
    for col in org_cols:
        if col in course_row and pd.notna(course_row[col]):
            organization = str(course_row[col]).strip()
            if organization:
                parts.append(f"Provider: {organization}")
                break
    
    # Skills are the most important part for matching
    skills = ""
    skills_cols = ['skills', 'Skills', 'skill', 'Skill', 'topics', 'Topics']
    for col in skills_cols:
        if col in course_row and pd.notna(course_row[col]):
            skills_text = str(course_row[col]).strip()
            if skills_text and skills_text.lower() != 'no skills specified':
                skills = skills_text
                parts.append(f"Skills covered: {skills}")
                break
                
    # Description/Metadata for additional context
    description = ""
    desc_cols = ['description', 'Description', 'course_description', 'summary', 'Summary', 'metadata', 'Metadata']
    for col in desc_cols:
        if col in course_row and pd.notna(course_row[col]):
            desc_text = str(course_row[col]).strip()
            if desc_text and desc_text.lower() not in ['no description available', 'nan']:
                description = (desc_text[:400] + '...') if len(desc_text) > 400 else desc_text
                parts.append(f"Details: {description}")
                break

    if not parts:
        return ""
        
    return ". ".join(parts)

def populate_skill_embeddings(df, collection, dataset_name, model, skill_column_names, skill_prefix):
    from utils import extract_skills_from_text
    print(f"📊 Populating {skill_prefix} skill embeddings for {dataset_name} in ChromaDB...")

    # List of datasets that use dot-gt columns for skills
    dot_gt_datasets = {
    "data_science_analytics",
    "engineering_hardware_networks", 
    "engineering_software_qa",
    "it_information_security",
    "project_program_management",
    "product_management",
    "research___development",
    "ux_design___architecture"
    }
    
    # Check if this is a dot-gt dataset before standardizing column names
    is_dot_gt_dataset = dataset_name in dot_gt_datasets
    
    # Store original column names for dot-gt datasets
    original_columns = df.columns.tolist() if is_dot_gt_dataset else None
    
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Special handling for NYC Jobs
    if dataset_name == "nyc_fresh_jobs_postings":
        skill_column_names = ['preferred_skills']

    # Handle dot-gt columns - look for the standardized versions
    if is_dot_gt_dataset:
        # Find dot-gt columns in the standardized column names
        dot_gt_cols = [col for col in df.columns if 'dot_gt' in col]
        if dot_gt_cols:
            skill_column_names = dot_gt_cols
            print(f"    Found dot-gt columns: {dot_gt_cols}")
        else:
            print(f"    Warning: No dot-gt columns found in {dataset_name}")
            print(f"    Available columns: {df.columns.tolist()}")

    # If no skill columns were identified, try a default list
    if not skill_column_names:
        skill_column_names = [
            'skills', 'preferred_skills', 'job_skills', 
            'required_skills', 'topics', 'tags'
        ]

    embeddings = []
    metadatas = []
    ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {dataset_name} {skill_prefix} skills"):
        skills = set()
        
        # Check if we're dealing with text fields that need skill extraction
        is_text_extraction = any(text_field in col.lower() for col in skill_column_names 
                               for text_field in ['description', 'summary', 'content', 'overview', 'metadata'])
        
        if is_text_extraction:
            # Extract skills from text fields
            for col in skill_column_names:
                if col in row and pd.notna(row[col]):
                    text_content = str(row[col]).strip()
                    if text_content and text_content.lower() not in ['no description available', 'nan']:
                        extracted_skills = extract_skills_from_text(text_content)
                        skills.update(extracted_skills)
        else:
            # Direct skill columns
            for col in skill_column_names:
                if col in row and pd.notna(row[col]):
                    val = row[col]
                    if isinstance(val, list):
                        skills.update([str(s).strip() for s in val if str(s).strip()])
                    else:
                        val = str(val).strip()
                        if val and val.lower() != 'no skills specified':
                            split_skills = [s.strip() for s in val.split(',') if s.strip()]
                            skills.update(split_skills)
        
        # Debug: Print first few rows to see what's happening
        if idx < 3 and is_dot_gt_dataset:
            print(f"      Row {idx} - Available skill columns: {skill_column_names}")
            for col in skill_column_names:
                if col in row:
                    print(f"        {col}: '{row[col]}'")
        
        # Fallback: extract from description if no skills found
        if not skills:
            desc_col = None
            for possible_col in ['job_description', 'job_desc', 'description']:
                if possible_col in row and pd.notna(row[possible_col]):
                    desc_col = possible_col
                    break
            if desc_col:
                skills.update(extract_skills_from_text(str(row[desc_col])))

        for skill in skills:
            if skill:
                try:
                    embedding = model.encode(skill)
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding
                    metadata = {
                        "dataset": dataset_name,
                        "skill_name": skill,
                        "source": skill_prefix,
                        "row_index": idx
                    }
                    # Use id_no if present for unique skill ID
                    if 'id_no' in row:
                        skill_id = f"{skill_prefix}{dataset_name}{row['id_no']}{skill.replace(' ', '')}"
                    else:
                        skill_id = f"{skill_prefix}{dataset_name}{idx}{skill.replace(' ', '')}"
                    embeddings.append(embedding.tolist())
                    metadatas.append(metadata)
                    ids.append(skill_id)
                except Exception as e:
                    print(f"      Error encoding skill '{skill}': {e}")

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        try:
            collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"    Added batch {i//batch_size + 1} ({len(batch_ids)} {skill_prefix} skills)")
        except Exception as e:
            print(f"    Error adding batch {i//batch_size + 1}: {e}")

    print(f"✅ Added {len(embeddings)} {skill_prefix} skill embeddings to ChromaDB")

def populate_job_embeddings(job_datasets, collections, model):
    """Populate ChromaDB with job embeddings."""
    print("📊 Populating job embeddings in ChromaDB...")

    embeddings = []
    metadatas = []
    ids = []

    for dataset_name, df in job_datasets.items():
        if df is None or df.empty:
            continue

        print(f"  Processing {dataset_name} ({len(df)} jobs)...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding jobs: {dataset_name}"):
            # Create job embedding text
            job_text = create_job_embedding_text(row)
            if not job_text or job_text.strip() == "":
                continue

            # Generate embedding
            try:
                embedding = model.encode(job_text)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding

                # Create metadata
                metadata = {
                    "dataset": dataset_name,
                    "job_text": job_text[:200],  # Truncate for storage
                    "source": "job"
                }

                # Add any available job-specific metadata
                if 'company' in row and pd.notna(row['company']):
                    metadata['company'] = str(row['company'])
                if 'jobtitle' in row and pd.notna(row['jobtitle']):
                    metadata['job_title'] = str(row['jobtitle'])
                elif 'business_title' in row and pd.notna(row['business_title']):
                    metadata['job_title'] = str(row['business_title'])

                # Use id_no if present for unique job ID
                if 'id_no' in row:
                    job_id = f"job_{dataset_name}_{row['id_no']}"
                else:
                    job_id = f"job_{dataset_name}_{idx}"
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
                ids.append(job_id)

            except Exception as e:
                print(f"    Error processing job {idx}: {e}")
                continue

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            collections["jobs"].add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"    Added batch {i//batch_size + 1} ({len(batch_ids)} jobs)")
        except Exception as e:
            print(f"    Error adding batch {i//batch_size + 1}: {e}")
    
    print(f"✅ Added {len(embeddings)} job embeddings to ChromaDB")

def populate_course_embeddings(course_datasets, collections, model):
    """Populate ChromaDB with course embeddings."""
    print("📚 Populating course embeddings in ChromaDB...")

    embeddings = []
    metadatas = []
    ids = []

    for dataset_name, df in course_datasets.items():
        if df is None or df.empty:
            continue

        print(f"  Processing {dataset_name} ({len(df)} courses)...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding courses: {dataset_name}"):
            # Create course embedding text
            course_text = create_course_embedding_text(row)
            if not course_text or course_text.strip() == "":
                continue

            # --- PHASE 3: Deep Model Resegmentation (Granular Syllabus Segmentation) ---
            # Instead of a single monolithic embedding, we chunk the course data into modular sub-embeddings.
            chunks = []
            
            # 1. The Monolithic Baseline (for backward compatibility)
            chunks.append({
                "text": course_text,
                "type": "monolithic",
                "chunk_id": 0
            })
            
            # 2. Granular Module Embeddings (Breaking down the syllabus/skills)
            # Split by '. ' which delineates Title, Provider, Skills, and Details
            parts = course_text.split('. ')
            for p_idx, part in enumerate(parts):
                if len(part.strip()) > 15: # Ignore tiny fragments
                    # Prepend the title for context so the sub-embedding isn't orphaned
                    title_context = row.get('title', row.get('course_title', 'Unknown Course'))
                    chunks.append({
                        "text": f"[{title_context}] Sub-module: {part.strip()}",
                        "type": "granular_module",
                        "chunk_id": p_idx + 1
                    })

            for chunk in chunks:
                try:
                    embedding = model.encode(chunk["text"])
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding

                    # Create metadata
                    metadata = {
                        "dataset": dataset_name,
                        "course_text": chunk["text"][:200],  # Truncate for storage
                        "source": "course",
                        "chunk_type": chunk["type"],
                        "chunk_id": chunk["chunk_id"]
                    }

                    # Add any available course-specific metadata
                    if 'organization' in row and pd.notna(row['organization']):
                        metadata['organization'] = str(row['organization'])
                    if 'title' in row and pd.notna(row['title']):
                        metadata['course_title'] = str(row['title'])
                    elif 'course_title' in row and pd.notna(row['course_title']):
                        metadata['course_title'] = str(row['course_title'])
                    if 'rating' in row and pd.notna(row['rating']):
                        metadata['rating'] = str(row['rating'])
                    if 'difficulty' in row and pd.notna(row['difficulty']):
                        metadata['difficulty'] = str(row['difficulty'])

                    # Use id_no if present for unique course ID, plus chunk_id
                    base_id = row['id_no'] if 'id_no' in row else idx
                    course_id = f"course_{dataset_name}_{base_id}_chunk_{chunk['chunk_id']}"
                    
                    embeddings.append(embedding.tolist())
                    metadatas.append(metadata)
                    ids.append(course_id)

                except Exception as e:
                    print(f"    Error processing course {idx} chunk {chunk['chunk_id']}: {e}")
                    continue

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            collections["courses"].add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"    Added batch {i//batch_size + 1} ({len(batch_ids)} courses)")
        except Exception as e:
            print(f"    Error adding batch {i//batch_size + 1}: {e}")
    
    print(f"✅ Added {len(embeddings)} course embeddings to ChromaDB")

def map_jobs_to_courses(job_datasets, course_datasets, model, top_n=3, output_path=None, batch_size=1000):
    """
    For each job posting, find the top N most relevant courses using sentence-transformers and cosine similarity.
    Uses batch processing to avoid memory issues with large datasets.
    Save the mapping as a CSV or JSON if output_path is provided.
    """
    print("\n🔗 Mapping jobs to relevant courses using semantic similarity...")
    from sklearn.metrics.pairwise import cosine_similarity
    import json
    
    # Prepare course texts (only once)
    course_rows = []
    course_texts = []
    course_ids = []
    for dataset_name, df in course_datasets.items():
        if df is None or df.empty:
            continue
        for idx, row in df.iterrows():
            # Use id_no if present for unique course ID, otherwise use idx
            if 'id_no' in row:
                course_id = f"course_{dataset_name}_{row['id_no']}"
            else:
                course_id = f"course_{dataset_name}_{idx}"
            # Use the same logic as create_course_embedding_text
            course_text = create_course_embedding_text(row)
            if course_text and course_text.strip():
                course_rows.append(row)
                course_texts.append(course_text)
                course_ids.append(course_id)
    print(f"  Total courses: {len(course_texts)}")

    # Build a mapping from course_id to course info (title, organization, etc.)
    course_id_to_info = {}
    for dataset_name, df in course_datasets.items():
        if df is None or df.empty:
            continue
        for idx, row in df.iterrows():
            # Use id_no if present for unique course ID, otherwise use idx
            if 'id_no' in row:
                course_id = f"course_{dataset_name}_{row['id_no']}"
            else:
                course_id = f"course_{dataset_name}_{idx}"
            title = None
            for col in ['title', 'Title', 'course_title', 'Course Title', 'name', 'Name']:
                if col in row and pd.notna(row[col]):
                    title = str(row[col])
                    break
            org = None
            for col in ['organization', 'Organization', 'provider', 'Provider']:
                if col in row and pd.notna(row[col]):
                    org = str(row[col])
                    break
            course_id_to_info[course_id] = {
                "title": title if title else "Unknown Title",
                "organization": org if org else "Unknown Organization"
            }

    # Encode course embeddings once
    print("  Encoding course texts...")
    course_embeds = model.encode(course_texts, show_progress_bar=True, batch_size=64)
    print(f"  Course embeddings shape: {course_embeds.shape}")

    # Prepare job texts
    job_rows = []
    job_texts = []
    job_ids = []
    for dataset_name, df in job_datasets.items():
        if df is None or df.empty:
            continue
        for idx, row in df.iterrows():
            # Use id_no if present for unique job ID
            if 'id_no' in row:
                job_id = f"job_{dataset_name}_{row['id_no']}"
            else:
                job_id = f"job_{dataset_name}_{idx}"
            # Use the same logic as create_job_embedding_text
            job_text = create_job_embedding_text(row)
            if job_text and job_text.strip():
                job_rows.append(row)
                job_texts.append(job_text)
                job_ids.append(job_id)
    print(f"  Total jobs: {len(job_texts)}")

    # Process jobs in batches to avoid memory issues
    job_to_courses = []
    total_batches = (len(job_texts) + batch_size - 1) // batch_size
    
    print(f"  Processing jobs in {total_batches} batches of {batch_size}...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(job_texts))
        
        print(f"    Processing batch {batch_idx + 1}/{total_batches} (jobs {start_idx}-{end_idx-1})...")
        
        # Get batch of job texts
        batch_job_texts = job_texts[start_idx:end_idx]
        batch_job_ids = job_ids[start_idx:end_idx]
        
        # Encode batch of job embeddings
        batch_job_embeds = model.encode(batch_job_texts, show_progress_bar=False, batch_size=64)
        
        # Compute cosine similarity for this batch
        batch_sim_matrix = cosine_similarity(batch_job_embeds, course_embeds)
        batch_sim_matrix = 1 / (1 + np.exp(-batch_sim_matrix))  # Apply sigmoid function
        
        # For each job in this batch, get top N courses
        for i, job_id in enumerate(batch_job_ids):
            top_idx = np.argsort(batch_sim_matrix[i])[::-1][:top_n]
            top_courses = []
            for j in top_idx:
                course_id = course_ids[j]
                sim_score = float(batch_sim_matrix[i][j])
                course_info = course_id_to_info.get(course_id, {})
                top_courses.append({
                    "course_id": course_id,
                    "similarity": sim_score,
                    "title": course_info.get("title", "Unknown Title"),
                    "organization": course_info.get("organization", "Unknown Organization")
                })
            
            current_job_row = job_rows[start_idx + i]
            title, company = get_job_title_and_company(current_job_row)
            required_skills = extract_job_skills(current_job_row)

            job_to_courses.append({
                "job_id": job_id,
                "job_title": title if title else "Unknown Title",
                "company": company if company else "Unknown Company",
                "required_skills": required_skills,
                "top_courses": top_courses
            })
        
        # Clear batch variables to free memory
        del batch_job_embeds, batch_sim_matrix
        import gc
        gc.collect()

    # Optionally save to file
    if output_path:
        print(f"  Saving mapping to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(job_to_courses, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Job-to-course mapping saved to: {output_path}")
    else:
        print("  ✓ Job-to-course mapping complete (not saved to file)")
    
    return job_to_courses

def create_positive_pairs_from_mapping(job_to_course_mapping, threshold=0.4):
    """
    Create positive pairs using semantic similarity mapping.
    Only pairs with similarity >= threshold are included.
    """
    positive_pairs = []
    for job_id, course_list in job_to_course_mapping.items():
        for course_id, sim_score in course_list:
            if sim_score >= threshold:
                positive_pairs.append((job_id, course_id, sim_score))
    return positive_pairs

def get_job_title_and_company(row):
    # Try all possible title columns
    title = None
    for col in ['jobtitle', 'job_title', 'business_title', 'title', 'Title']:
        if col in row and pd.notna(row[col]):
            title = str(row[col])
            break
    company = None
    for col in ['company', 'Company', 'organization', 'Organization', 'agency', 'Agency']:
        if col in row and pd.notna(row[col]):
            company = str(row[col])
            break
    return title, company

def extract_job_skills(row):
    """Extract skills from a job row."""
    skills = set()
    
    # Common skill column names
    skill_columns = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
    
    for col in skill_columns:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, list):
                skills.update([str(s).strip() for s in val if str(s).strip()])
            else:
                val = str(val).strip()
                if val and val.lower() != 'no skills specified':
                    split_skills = [s.strip() for s in val.split(',') if s.strip()]
                    skills.update(split_skills)
    
    # Fallback: extract from description if no skills found
    if not skills:
        desc_col = None
        for possible_col in ['job_description', 'job_desc', 'description']:
            if possible_col in row and pd.notna(row[possible_col]):
                desc_col = possible_col
                break
        if desc_col:
            from utils import extract_skills_from_text
            skills.update(extract_skills_from_text(str(row[desc_col])))
    
    return list(skills)

def main():
    print("🚀 Starting ChromaDB population from raw data...")
    
    # Load the embedding model
    model = get_embedding_model()
    if model is None:
        print("❌ Failed to load embedding model. Exiting.")
        return
    
    # Initialize ChromaDB collections
    collections = initialize_chroma_collections()
    
    # Load raw datasets
    processed_datasets = {}
    print("Loading raw datasets from job_dataset directory...")
    
    for fname in JOB_CSVS + COURSE_CSVS:
        key = fname.replace('.csv', '').replace(' ', '').replace('-', '').lower()
        path = JOB_DATASET_DIR / fname
        if path.exists():
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            last_err = None
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except Exception as e:
                    last_err = e
            else:
                print(f"  Error loading {path}: {last_err}")
                continue
            # Ensure skills columns are treated as strings or lists of strings
            for col in ['skills', 'preferred_skills', 'job_skills', 'required_skills', 'topics', 'canonical_course_skills']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and (x.startswith("[") or x.startswith("{")) else x)
                    df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))
                    df[col] = df[col].fillna('No skills specified')
            processed_datasets[key] = df
            print(f"  Loaded {path} (Shape: {df.shape})")
        else:
            print(f"  Warning: File not found for {fname} at {path}. Skipping.")

    if not processed_datasets:
        print("No raw data found to populate ChromaDB. Exiting.")
        return

    # Separate job and course datasets
    job_datasets = {name: df for name, df in processed_datasets.items() if name.startswith(tuple(j.replace('.csv', '').replace(' ', '').replace('-', '').lower() for j in JOB_CSVS))}
    course_datasets = {name: df for name, df in processed_datasets.items() if name.startswith(tuple(c.replace('.csv', '').replace(' ', '').replace('-', '').lower() for c in COURSE_CSVS))}

    # Populate embeddings for full job/course descriptions
    populate_job_embeddings(job_datasets, collections, model)
    populate_course_embeddings(course_datasets, collections, model)

    # Populate embeddings for individual skills
    job_skill_cols = ['skills', 'preferred_skills', 'job_skills', 'required_skills']
    for name, df in job_datasets.items():
        populate_skill_embeddings(df, collections["job_skills"], name, model, job_skill_cols, "job")

    # Check what columns are available in course datasets and use appropriate skill columns
    for name, df in course_datasets.items():
        if df is None or df.empty:
            continue
            
        print(f"  Checking columns for {name}: {df.columns.tolist()}")
        
        # Look for skill-related columns in the original column names (before standardization)
        skill_columns = []
        
        # Common skill column names in course datasets
        possible_skill_cols = [
            'Skills', 'skills', 'Skill', 'skill', 
            'Topics', 'topics', 'Topic', 'topic',
            'Metadata', 'metadata', 'Description', 'description',
            'Course Skills', 'course_skills', 'Course_Skills',
            'Review counts', 'review_counts', 'Ratings', 'ratings'
        ]
        
        for col in df.columns:
            if any(skill_name in col for skill_name in possible_skill_cols):
                skill_columns.append(col)
        
        if skill_columns:
            print(f"    Found skill columns for {name}: {skill_columns}")
            populate_skill_embeddings(df, collections["course_skills"], name, model, skill_columns, "course")
        else:
            print(f"    No skill columns found for {name}. Available columns: {df.columns.tolist()}")
            # Try to extract skills from description or other text fields
            text_columns = [col for col in df.columns if any(text_field in col.lower() for text_field in ['description', 'summary', 'content', 'overview', 'metadata'])]
            if text_columns:
                print(f"    Will extract skills from text columns: {text_columns}")
                populate_skill_embeddings(df, collections["course_skills"], name, model, text_columns, "course")
            else:
                print(f"    Skipping {name} - no suitable columns for skill extraction")

    # --- Map jobs to relevant courses using semantic similarity ---
    output_mapping_path = JOB_DATASET_DIR / "job_to_course_mapping.json"
    
    # Use smaller batch size if you have memory constraints
    # You can reduce this to 500, 250, or even 100 if needed
    batch_size = 100  # Reduced from 1000 to be more conservative
    map_jobs_to_courses(job_datasets, course_datasets, model, top_n=3, output_path=output_mapping_path, batch_size=batch_size)

    print("✅ ChromaDB population complete.")

if __name__ == "__main__":
    main()