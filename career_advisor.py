import pandas as pd
from pathlib import Path
import re
import glob
import os

# For Vector DB integration
import chromadb
from sentence_transformers import SentenceTransformer # type: ignore

# Define paths to the data files
# Assuming the script is run from the workspace root where 'archive (3)' directory exists.
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"  # New base directory for job and course data
COURSERA_CSV = JOB_DATASET_DIR / "coursera_course_dataset_v3.csv"
# LINKEDIN_JOBS_CSV = DATA_DIR / "linkedin_job_postings.csv" # Not directly used in this simplified version
# JOB_SKILLS_CSV = DATA_DIR / "job_skills.csv" # No longer directly used
# JOB_SUMMARY_CSV = DATA_DIR / "job_summary.csv" # Not directly used
RESUME_DIR = BASE_DIR / "resume" # Directory where user might upload resumes

# SQLite Database
# DB_PATH = BASE_DIR / "job_data.db" # Removed SQLite

# ChromaDB Setup
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
CHROMA_JOB_SKILLS_COLLECTION = "job_skills_embeddings"
CHROMA_COURSE_SKILLS_COLLECTION = "course_skills_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A good general-purpose model

# Global variable for the embedding model to avoid reloading it multiple times
embedding_model = None

def get_embedding_model():
    """Loads and returns the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Please ensure sentence-transformers is installed and the model name is correct.")
            raise
    return embedding_model

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
    return client

# --- Skill Normalization ---
# Basic map, can be expanded significantly (e.g., using LLM suggestions or larger dictionaries)
SKILL_NORMALIZATION_MAP = {
    # Programming Languages & Scripting
    "python programming": "python", "pyhton": "python", # Common typo
    "java programming": "java", "core java": "java",
    "javascripting": "javascript", "javasript": "javascript", "js": "javascript", "es6": "javascript",
    "c#": "c-sharp", "c sharp": "c-sharp",
    "c++": "c-plus-plus", "cpp": "c-plus-plus",
    "php scripting": "php",
    "ruby on rails": "ruby", "ror": "ruby",
    "golang": "go",
    "swift programming": "swift",
    "kotlin programming": "kotlin",
    "scala programming": "scala",
    "r programming": "r",
    "typescript": "typescript", "ts": "typescript",
    "visual basic": "vb.net", "vb": "vb.net",
    "shell scripting": "shell script", "bash scripting": "shell script", "bash": "shell script",
    "powershell": "powershell",

    # Web Frameworks & Libraries
    "reactjs": "react", "react.js": "react",
    "angularjs": "angular", "angular.js": "angular", "angular 2+": "angular",
    "vue.js": "vue", "vuejs": "vue",
    "node.js": "node", "nodejs": "node",
    "express.js": "express", "expressjs": "express",
    "django framework": "django",
    "flask framework": "flask",
    "spring framework": "spring", "spring boot": "spring",
    ".net core": "dotnet-core", ".net": "dotnet-core", "asp.net": "asp-dotnet",
    "jquery": "jquery",
    "bootstrap framework": "bootstrap",

    # Databases & Data Storage
    "sql server": "sql", "ms sql": "sql", "mssql": "sql",
    "mysql": "sql",
    "postgresql": "sql", "postgres": "sql", "postgressql": "sql",
    "oracle db": "oracle", "oracle database": "oracle",
    "sqlite": "sqlite",
    "mongodb": "mongodb", "mongo": "mongodb",
    "cassandra": "cassandra",
    "redis": "redis",
    "elasticsearch": "elasticsearch", "elastic search": "elasticsearch",
    "dynamodb": "dynamodb",
    "firebase": "firebase",
    "data warehousing": "data warehouse",
    "etl processes": "etl", "extract transform load": "etl",

    # Cloud Platforms & Services
    "aws": "amazon web services", "amazon web services (aws)": "amazon web services",
    "amazon s3": "amazon s3", "s3": "amazon s3",
    "amazon ec2": "amazon ec2", "ec2": "amazon ec2",
    "aws lambda": "aws lambda",
    "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "google compute engine": "google compute engine",
    "google cloud functions": "google cloud functions",
    "azure": "microsoft azure", "ms azure": "microsoft azure",
    "azure functions": "azure functions",
    "azure blob storage": "azure blob storage",
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "docker": "docker",
    "serverless architecture": "serverless",
    "iaas": "iaas", "paas": "paas", "saas": "saas",

    # ML/AI & Data Science
    "machine learning (ml)": "machine learning", "ml": "machine learning",
    "deep learning (dl)": "deep learning", "dl": "deep learning", "neural networks": "deep learning",
    "artificial intelligence (ai)": "artificial intelligence", "ai": "artificial intelligence",
    "natural language processing": "nlp", "nlp techniques": "nlp",
    "computer vision": "computer vision", "cv": "computer vision",
    "data analysis": "data analysis", "data analytics": "data analysis",
    "data visualization": "data visualization", "dataviz": "data visualization",
    "tableau software": "tableau",
    "power bi": "powerbi", "microsoft power bi": "powerbi",
    "pandas library": "pandas",
    "numpy library": "numpy",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "tensorflow framework": "tensorflow", "keras": "tensorflow", # Keras often used with TF
    "pytorch framework": "pytorch",
    "big data technologies": "big data", "hadoop": "big data", "spark": "big data", "apache spark": "big data",
    "statistics": "statistical analysis", "statistical modeling": "statistical analysis",

    # DevOps & Tools
    "git version control": "git", "github": "git", "gitlab": "git", "bitbucket": "git",
    "ci/cd": "cicd", "continuous integration": "cicd", "continuous deployment": "cicd",
    "jenkins": "jenkins",
    "ansible": "ansible",
    "terraform": "terraform",
    "jira": "jira", "atlassian jira": "jira",
    "agile methodologies": "agile", "scrum master": "agile", "scrum": "agile", "kanban": "agile",

    # Operating Systems
    "linux os": "linux", "unix environment": "unix",
    "windows server": "windows server",
    "macos": "macos",

    # Software Engineering & Design
    "software development life cycle": "sdlc",
    "object-oriented programming": "oop", "object oriented design": "oop",
    "rest apis": "restful apis", "restful services": "restful apis", "api design": "restful apis",
    "microservices architecture": "microservices",
    "design patterns": "design patterns",
    "software testing": "qa testing", "quality assurance": "qa testing", "automated testing": "qa testing",
    "unit testing": "unit testing", "integration testing": "integration testing",

    # Business & Soft Skills
    "excel": "microsoft excel", "ms excel": "microsoft excel",
    "spreadsheets": "spreadsheet software",
    "microsoft office suite": "microsoft office", "ms office": "microsoft office",
    "powerpoint": "powerpoint", "ms powerpoint": "powerpoint",
    "communication skills": "communication", "verbal communication": "communication", "written communication": "communication",
    "team work": "teamwork", "collaboration skills": "teamwork", "team player": "teamwork",
    "problem-solving": "problem solving", "analytical skills": "problem solving", "critical thinking": "critical thinking",
    "project management": "project management", "project planning": "project management",
    "leadership skills": "leadership", "team leadership": "leadership",
    "time management": "time management",
    "customer service": "customer support", "client facing skills": "customer support",
    "technical support": "technical support",
    "data entry": "data entry",
    "sales skills": "sales", "business development": "sales",
    "marketing skills": "marketing", "digital marketing": "marketing", "seo": "seo", "sem": "sem",
    "content creation": "content writing", "copywriting": "content writing",
    "graphic design": "graphic design", "adobe photoshop": "adobe creative suite", "adobe illustrator": "adobe creative suite", "adobe indesign": "adobe creative suite",
    # Add more as needed
}

def normalize_skill(skill_text, normalization_map):
    """Normalizes a single skill string."""
    if not isinstance(skill_text, str):
        return "" # Or handle as an error/log
    cleaned_skill = skill_text.lower().strip()
    return normalization_map.get(cleaned_skill, cleaned_skill) # Return mapped value or original cleaned skill

def normalize_skill_list(skills, normalization_map):
    """Normalizes a list of skill strings and returns a list of unique canonical skills."""
    if not skills:
        return []
    
    # Handle cases where 'skills' might be a single string instead of a list (e.g. from bad CSV parsing before splitting)
    if isinstance(skills, str):
        # This assumes skills in a string are comma-separated if not already a list
        # This is a fallback; ideally, the input 'skills' should already be a list of strings.
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    normalized_set = set()
    for skill in skills:
        if isinstance(skill, str): # Ensure each item in the list is a string
            normalized_set.add(normalize_skill(skill, normalization_map))
        # else: log or handle non-string item
    return sorted(list(normalized_set))

def initialize_vector_database(df_coursera_for_skills):
    """
    Initializes ChromaDB and populates it with skill embeddings from job postings and Coursera data.
    This function should be run once to set up the ChromaDB collections.
    df_coursera_for_skills: DataFrame containing processed Coursera data, including 'Canonical_Course_Skills'.
    """
    print(f"Initializing ChromaDB at {CHROMA_DATA_PATH}...")
    
    model = get_embedding_model()
    chroma_client = get_chroma_client()
    
    chunk_size = 10 # For reading job postings

    # -- ChromaDB Initialization for Job Skills --
    print(f"Populating ChromaDB with job skill embeddings from CSV files in {JOB_DATASET_DIR}...")
    job_skill_collection = chroma_client.get_or_create_collection(
        name=CHROMA_JOB_SKILLS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB collection '{CHROMA_JOB_SKILLS_COLLECTION}' ready.")
    
    processed_job_skill_ids_in_chroma = set() # To avoid trying to add duplicates in this session if script is re-run partially
                                          # For true idempotency, would need to check existing IDs in Chroma or clear collection.

    # Iterate through CSV files in the job_dataset directory
    for job_csv in glob.glob(os.path.join(str(JOB_DATASET_DIR), "*.csv")):
        job_csv_path = Path(job_csv)
        if job_csv_path == COURSERA_CSV:
            print(f"Skipping Coursera data file: {job_csv_path}")
            continue  # Skip the Coursera data file

        print(f"Processing job postings from: {job_csv_path}")
        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(job_csv_path, chunksize=chunk_size, on_bad_lines='skip')):
                print(f"Processing chunk {chunk_idx + 1} of {job_csv_path} for job skills...")
                embeddings_batch = []
                metadata_batch = []
                ids_batch = []
                
                for _, row in chunk.iterrows():
                    job_link = row.get('title href')  # Changed to 'title href'
                    raw_skills_str = row.get('job-desc') # Changed to 'job-desc'

                    if pd.isna(job_link) or pd.isna(raw_skills_str) or not isinstance(raw_skills_str, str):
                        continue
                    
                    canonical_skills = normalize_skill_list(raw_skills_str.split(','), SKILL_NORMALIZATION_MAP)
                    
                    for skill in canonical_skills:
                        if not skill: continue # Skip empty skill strings after normalization
                        skill_id = f"job_{job_link}_{skill.replace(' ', '_').replace('/', '_')}" # Make ID more robust
                        
                        # This check is for current session; a true check would query Chroma if skill_id exists
                        if skill_id not in processed_job_skill_ids_in_chroma:
                            embeddings_batch.append(model.encode(skill).tolist()) # type: ignore
                            metadata_batch.append({"job_link": str(job_link), "skill_name": skill, "source": "job_posting", "file": job_csv_path.name})
                            ids_batch.append(skill_id)
                            processed_job_skill_ids_in_chroma.add(skill_id)

                if ids_batch:
                    try:
                        job_skill_collection.add(embeddings=embeddings_batch, metadatas=metadata_batch, ids=ids_batch)
                        print(f"Added {len(ids_batch)} job skill embeddings to ChromaDB from chunk {chunk_idx + 1} of {job_csv_path}.")
                    except Exception as e:
                        print(f"Error adding job skill embeddings to ChromaDB for chunk {chunk_idx+1} of {job_csv_path}: {e}")
            print(f"Finished populating ChromaDB for job skills from {job_csv_path}.")
        except Exception as e:
            print(f"Error processing {job_csv_path}: {e}")

    # -- ChromaDB Population for Coursera Skills --
    print("Populating ChromaDB with Coursera skill embeddings...")
    if df_coursera_for_skills is None or df_coursera_for_skills.empty:
        print("Coursera DataFrame is empty or None. Cannot populate Coursera skills in ChromaDB.")
        return

    course_skill_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COURSE_SKILLS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB collection '{CHROMA_COURSE_SKILLS_COLLECTION}' ready.")

    embeddings_batch_course = []
    metadata_batch_course = []
    ids_batch_course = []
    processed_course_skill_ids_in_chroma = set()

    for index, row in df_coursera_for_skills.iterrows():
        course_url = row.get('course_url', f"course_idx_{index}")
        course_title = row.get("Title", "N/A")
        canonical_skills = row.get("Canonical_Course_Skills", [])
        
        if not isinstance(canonical_skills, list): canonical_skills = []
        
        for skill in canonical_skills:
            if not skill: continue
            skill_id = f"course_{str(course_url)}_{skill.replace(' ', '_').replace('/', '_')}"

            if skill_id not in processed_course_skill_ids_in_chroma:
                embeddings_batch_course.append(model.encode(skill).tolist()) # type: ignore
                metadata_batch_course.append({"course_url": str(course_url), "skill_name": skill, "course_title": course_title, "source": "coursera"})
                ids_batch_course.append(skill_id)
                processed_course_skill_ids_in_chroma.add(skill_id)
            
            if len(ids_batch_course) >= 1000: # Batch add
                try:
                    course_skill_collection.add(embeddings=embeddings_batch_course, metadatas=metadata_batch_course, ids=ids_batch_course)
                    print(f"Added {len(ids_batch_course)} course skill embeddings to ChromaDB.")
                    embeddings_batch_course, metadata_batch_course, ids_batch_course = [], [], []
                except Exception as e:
                    print(f"Error adding batch of course skill embeddings: {e}")
                    embeddings_batch_course, metadata_batch_course, ids_batch_course = [], [], []
    
    if ids_batch_course: # Add any remaining
        try:
            course_skill_collection.add(embeddings=embeddings_batch_course, metadatas=metadata_batch_course, ids=ids_batch_course)
            print(f"Added final {len(ids_batch_course)} course skill embeddings to ChromaDB.")
        except Exception as e:
            print(f"Error adding final batch of course skill embeddings: {e}")
    
    print("ChromaDB skill population complete.")

def load_coursera_data():
    """
    Loads and preprocesses the Coursera dataset from CSV.
    This DataFrame is used for ChromaDB population and for course detail lookup.
    """
    print(f"Loading and processing Coursera data from {COURSERA_CSV}...")
    if not COURSERA_CSV.exists():
        print(f"{COURSERA_CSV} not found.")
        return None
    try:
        df_coursera = pd.read_csv(COURSERA_CSV)
        if "Skills" in df_coursera.columns:
            df_coursera["Processed_Skills"] = df_coursera["Skills"].astype(str).apply(
                lambda x: [skill.strip().lower() for skill in x.split(",") if skill.strip()]
            )
            df_coursera["Canonical_Course_Skills"] = df_coursera["Processed_Skills"].apply(
                lambda skill_list: normalize_skill_list(skill_list, SKILL_NORMALIZATION_MAP)
            )
        else:
            print("Warning: 'Skills' column not found in Coursera data. Adding empty skill lists.")
            df_coursera["Processed_Skills"] = pd.Series([[] for _ in range(len(df_coursera))])
            df_coursera["Canonical_Course_Skills"] = pd.Series([[] for _ in range(len(df_coursera))])

        if df_coursera.columns[0].strip() == "": # Remove unnamed index column if it exists
             df_coursera = df_coursera.iloc[: , 1:]
        print("Coursera data loaded and processed into DataFrame.")
        return df_coursera
    except Exception as e:
        print(f"Error loading or processing Coursera data: {e}")
        return None

# --- Core Logic Functions ---
def extract_skills_from_resume(resume_path):
    """
    Extracts and normalizes skills from a given resume PDF (Placeholder).
    """
    print(f"Extracting skills from resume: {resume_path} (Placeholder)")
    # TODO: Implement PDF parsing and actual LLM/NLP skill extraction
    raw_skills = []
    if "software_engineer_resume.pdf" in str(resume_path):
        raw_skills = ["python", "Java", "SQL", "Django", "Machine Learning", "Communication", "problem-solving", "AWS"]
    elif "data_analyst_resume.pdf" in str(resume_path):
        raw_skills = ["SQL", "Excel", "Tableau", "Python", "Data Analysis", "Statistics", "ms excel", "Power BI"]
    else:
        raw_skills = ["placeholder skill 1", "Placeholder Skill 2", "js", "project management"]
    
    normalized_resume_skills = normalize_skill_list(raw_skills, SKILL_NORMALIZATION_MAP)
    print(f"Normalized resume skills: {normalized_resume_skills}")
    return normalized_resume_skills

def find_target_job_skills_placeholder(job_title_query):
    """
    Placeholder to represent getting skills for a target job role.
    In a full system, this might involve an LLM or a predefined skill set for roles.
    For this simplified version, we return a fixed list based on a keyword.
    """
    print(f"Getting placeholder target job skills for: '{job_title_query}'")
    if "software engineer" in job_title_query.lower():
        return normalize_skill_list(["Python", "Java", "SQL", "APIs", "Problem Solving", "Teamwork", "Agile", "Cloud Computing", "Docker"], SKILL_NORMALIZATION_MAP)
    elif "data analyst" in job_title_query.lower():
        return normalize_skill_list(["SQL", "Microsoft Excel", "Tableau", "Python", "Data Visualization", "Statistical Analysis", "R Programming"], SKILL_NORMALIZATION_MAP)
    return normalize_skill_list(["Communication", "Problem Solving", "Teamwork"], SKILL_NORMALIZATION_MAP)

def analyze_skill_gap(resume_skills, job_skills):
    """
    Compares resume skills with job skills to find the gap.
    Both inputs are expected to be lists of canonical skill strings.
    """
    print("Analyzing skill gap...")
    resume_skills_set = set(s.lower().strip() for s in resume_skills)
    job_skills_set = set(s.lower().strip() for s in job_skills) # Ensure job_skills are also cleaned if coming from varied sources
    
    missing_skills = sorted(list(job_skills_set - resume_skills_set))
    print(f"Resume Skills (canonical): {sorted(list(resume_skills_set))}")
    print(f"Target Job Skills (canonical): {sorted(list(job_skills_set))}")
    print(f"Missing Skills (canonical string): {missing_skills}")
    return missing_skills

def recommend_courses_with_vector_db(skills_to_find_courses_for, df_coursera_details, top_n_courses_per_skill=2, overall_top_n=5):
    """
    Recommends Coursera courses based on a list of skills using ChromaDB for semantic search.
    df_coursera_details is used to retrieve full course details after finding matches.
    """
    print(f"Recommending courses using Vector DB for skills: {skills_to_find_courses_for}...")
    if df_coursera_details is None or df_coursera_details.empty:
        print("Coursera DataFrame not available for recommendations.")
        return []
    if not skills_to_find_courses_for:
        print("No skills provided to find courses for. No recommendations needed.")
        return []

    all_recommendations_data = []
    try:
        model = get_embedding_model()
        chroma_client = get_chroma_client()
        course_skill_collection = chroma_client.get_collection(name=CHROMA_COURSE_SKILLS_COLLECTION)
    except Exception as e:
        print(f"Error setting up ChromaDB for course recommendation: {e}.")
        return [] # Cannot proceed without vector DB

    for skill_needed_str in skills_to_find_courses_for:
        print(f"Searching courses in ChromaDB for skill: '{skill_needed_str}'")
        if not skill_needed_str: continue # Skip if skill is empty

        skill_embedding = model.encode(skill_needed_str).tolist() # type: ignore
        
        try:
            query_results = course_skill_collection.query(
                query_embeddings=[skill_embedding],
                n_results=top_n_courses_per_skill * 2, # Fetch more to allow for deduplication across skills
                include=["metadatas", "distances"]
            )
        except Exception as e:
            print(f"Error querying ChromaDB for skill '{skill_needed_str}': {e}")
            continue 

        if query_results and query_results['ids'] and query_results['ids'][0]:
            for i, meta in enumerate(query_results['metadatas'][0]):
                course_url = meta.get("course_url")
                matched_skill_in_course = meta.get("skill_name")
                distance = query_results['distances'][0][i]
                course_title = meta.get("course_title")

                if course_url:
                    # Full details lookup from the DataFrame (already loaded)
                    course_details_row = df_coursera_details[df_coursera_details['course_url'] == course_url]
                    if not course_details_row.empty:
                        details = course_details_row.iloc[0]
                        all_recommendations_data.append({
                            "queried_skill": skill_needed_str,
                            "matched_skill_in_course": matched_skill_in_course,
                            "similarity_distance": distance,
                            "course_title": details.get("Title", course_title), # Prefer Title from DF
                            "organization": details.get("Organization", "N/A"),
                            "course_url": course_url,
                            "rating": details.get("Ratings", "N/A"),
                            "difficulty": details.get("Difficulty", "N/A")
                        })
    
    # Deduplicate courses based on URL and sort by distance
    unique_recommendations = []
    seen_urls = set()
    # Sort by distance first to prioritize closer matches when deduplicating
    all_recommendations_data.sort(key=lambda x: x['similarity_distance'])
    
    for rec in all_recommendations_data:
        if rec["course_url"] not in seen_urls:
            unique_recommendations.append(rec)
            seen_urls.add(rec["course_url"])
            if len(unique_recommendations) >= overall_top_n: # Overall limit
                break
                
    print(f"Found {len(unique_recommendations)} unique course recommendations via Vector DB.")
    return unique_recommendations

def main():
    """
    Main function to run the simplified career advisor application.
    Focuses on populating ChromaDB with skills and recommending courses based on resume skills.
    """
    # Step 1: Load Coursera Data into a DataFrame
    # This DataFrame will be used for populating ChromaDB and for looking up course details later.
    df_coursera = load_coursera_data()
    if df_coursera is None:
        print("Exiting because Coursera data could not be loaded.")
        return

    # Step 2: Initialize Vector Database (ChromaDB) with skills from jobs and Coursera
    # This should ideally be run once, or if CHROMA_DATA_PATH does not exist / needs refresh.
    # For this script, we'll make it conditional or you can run it explicitly.
    
    # A simple check: if the Chroma data path doesn't exist, assume we need to initialize.
    # More robust would be to check collection counts or have a specific flag.
    if not CHROMA_DATA_PATH.exists() or input(f"ChromaDB path '{CHROMA_DATA_PATH}' seems to exist. Re-initialize/populate ChromaDB? (y/N): ").lower() == 'y':
        print("Attempting to initialize and populate ChromaDB with skills...")
        try:
            initialize_vector_database(df_coursera) # Pass the loaded coursera df
        except Exception as e:
            print(f"ChromaDB initialization failed: {e}")
            # Depending on severity, you might want to exit or continue without vector features
            # For now, we'll try to continue if df_coursera is loaded, but recommendations might fail.
    else:
        print("Skipping ChromaDB initialization/population as path exists and user did not opt-in.")

    print("\n--- Welcome to the Simplified Career Advisor! ---")

    # Step 3: Get User Resume and Extract Skills (Placeholder)
    sample_resume_path = RESUME_DIR / "software_engineer_resume.pdf" 
    if not RESUME_DIR.exists(): RESUME_DIR.mkdir(parents=True, exist_ok=True)
    if not sample_resume_path.exists():
         with open(sample_resume_path, 'w') as f:
             f.write("Dummy resume: Python, Java, SQL, problem-solving, AWS, Docker.")
    
    user_resume_canonical_skills = extract_skills_from_resume(sample_resume_path)
    if not user_resume_canonical_skills:
        print("Could not extract skills from resume. Exiting.")
        return
    print(f"\nSkills extracted from your resume (canonical): {user_resume_canonical_skills}")

    # Step 4: Define Target Job Skills (Simplified - No SQLite job search)
    target_job_query = input("Enter your target job title (e.g., 'software engineer', 'data analyst'): ")
    target_job_canonical_skills = find_target_job_skills_placeholder(target_job_query)
    print(f"Skills for target job '{target_job_query}' (canonical): {target_job_canonical_skills}")

    # Step 5: Analyze Skill Gap
    missing_canonical_skills = analyze_skill_gap(user_resume_canonical_skills, target_job_canonical_skills)

    # Step 6: Recommend Courses based on Missing Skills using Vector DB
    if missing_canonical_skills:
        print(f"\nAttempting to find courses for missing skills: {missing_canonical_skills}")
        course_recommendations = recommend_courses_with_vector_db(missing_canonical_skills, df_coursera)
        
        if course_recommendations:
            print("\n--- Recommended Courses (from Vector DB) ---")
            for rec in course_recommendations:
                print(f"For skill: '{rec['queried_skill']}' (matched '{rec.get('matched_skill_in_course', 'N/A')}' in course; distance {rec.get('similarity_distance', -1):.4f})")
                print(f"  Course: {rec['course_title']} by {rec['organization']}")
                print(f"  Difficulty: {rec['difficulty']}, Rating: {rec['rating']}")
                print(f"  Link: {rec['course_url']}\n")
        else:
            print("No specific courses found for the missing skills via Vector DB, or Coursera data is unavailable.")
    elif target_job_canonical_skills: # Only print if there were target skills to compare against
        print("\nCongratulations! Your resume skills seem to cover the requirements for this job based on our canonical skill comparison.")
    else:
        print("\nNo target job skills defined for comparison. Showing courses for all your resume skills instead:")
        # If no missing skills because no target job skills, recommend based on all resume skills
        course_recommendations_for_resume = recommend_courses_with_vector_db(user_resume_canonical_skills, df_coursera)
        if course_recommendations_for_resume:
            print("\n--- Courses related to your resume skills (from Vector DB) ---")
            for rec in course_recommendations_for_resume:
                print(f"Related to your skill: '{rec['queried_skill']}' (matched '{rec.get('matched_skill_in_course', 'N/A')}' in course; distance {rec.get('similarity_distance', -1):.4f})")
                print(f"  Course: {rec['course_title']} by {rec['organization']}")
                print(f"  Difficulty: {rec['difficulty']}, Rating: {rec['rating']}")
                print(f"  Link: {rec['course_url']}\n")

if __name__ == "__main__":
    main()
