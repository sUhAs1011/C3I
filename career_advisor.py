import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from pathlib import Path
import re
try:
    import spacy
except ModuleNotFoundError:
    print("Error: spacy module not found. Please install it using: pip install spacy")
    # Optionally, you can exit the program or disable spacy-related functionality
    spacy = None  # Set spacy to None to prevent further errors
except Exception as e:
    print(f"An unexpected error occurred while importing spacy: {e}")
    spacy = None

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
try:
    import fitz  # PyMuPDF - Import here, outside the function
except ModuleNotFoundError:
    print("Error: fitz module not found. Please install it using: pip install pymupdf")
    fitz = None
except Exception as e:
    print(f"An unexpected error occurred while importing fitz: {e}")
    fitz = None

import chromadb
import numpy as np  # Import numpy
import webbrowser  # Import webbrowser
from tkinter import ttk  # Import ttk

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# For Vector DB integration
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:
    print("Error: sentence_transformers module not found. Please install it using: pip install sentence_transformers")
    SentenceTransformer = None
except Exception as e:
    print(f"An unexpected error occurred while importing sentence_transformers: {e}")
    SentenceTransformer = None

from utils import SKILL_NORMALIZATION_MAP, normalize_skill, normalize_skill_list

# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"
COURSERA_CSV = JOB_DATASET_DIR / "coursera_course_dataset_v3.csv"
RESUME_DIR = BASE_DIR / "resume"
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
CHROMA_JOB_SKILLS_COLLECTION = "job_skills_embeddings"
CHROMA_COURSE_SKILLS_COLLECTION = "course_skills_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Global Variables ---
embedding_model = None
df_coursera = None  # Store Coursera data
job_skill_collection = None  # Store ChromaDB job skills collection
course_skill_collection = None  # Store ChromaDB course skills collection
nlp = None # spacy model

# --- Helper Functions ---
def get_embedding_model():
    """Loads and returns the sentence transformer model."""
    global embedding_model
    if SentenceTransformer is None:
        print("The sentence_transformers library is not installed.")
        return None
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            messagebox.showerror("Error", "Failed to load embedding model. Ensure sentence-transformers is installed.")
            return None
    return embedding_model

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
    return client

def initialize_vector_database(df_coursera_for_skills):
    """Initializes ChromaDB with skill embeddings."""
    global job_skill_collection, course_skill_collection
    print(f"Initializing ChromaDB at {CHROMA_DATA_PATH}...")

    model = get_embedding_model()
    if model is None:
        return # Exit if embedding model failed to load

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

    # --- Populate with Coursera Skills ---
    print("Populating ChromaDB with Coursera skill embeddings...")
    embeddings_batch_course = []
    metadata_batch_course = []
    ids_batch_course = []

    for index, row in df_coursera_for_skills.iterrows():
        course_url = row.get('course_url', f"course_idx_{index}")
        course_title = row.get("Title", "N/A")
        canonical_skills = row.get("Canonical_Course_Skills", [])

        for skill in canonical_skills:
            if not skill:
                continue
            skill_id = f"course_{str(course_url)}_{skill.replace(' ', '_')}"
            embeddings_batch_course.append(model.encode(skill).tolist())  # type: ignore
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

def load_coursera_data():
    """Loads and preprocesses the Coursera dataset."""
    global df_coursera
    print(f"Loading Coursera data from {COURSERA_CSV}...")
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
        print("Coursera data loaded and processed into DataFrame.")
        return True
    except Exception as e:
        print(f"Error loading or processing Coursera data: {e}")
        messagebox.showerror("Error", "Failed to load Coursera data.")
        return False

def get_spacy_model():
    """Loads and returns the spaCy model."""
    global nlp
    if nlp is None:
        print("Loading spaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")  # Or a larger model like "en_core_web_lg"
            print("spaCy model loaded.")
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            messagebox.showerror("Error", "Failed to load spaCy model.  Make sure you have downloaded it (python -m spacy download en_core_web_sm)")
            return None
    return nlp

def extract_text_from_pdf(pdf_path):
    try:
        if fitz is None:
            print("PyMuPDF (fitz) is not installed. Cannot extract text from PDF.")
            return None
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return None
    except fitz.FileDataError:
        print(f"Error: Could not open {pdf_path} - file might be corrupted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def extract_skills_from_text_spacy(text, nlp):
    """Extracts skills from text using spaCy's NER and keyword matching."""
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECHNOLOGY", "SKILL"]:  # Example entity types
            skills.append(ent.text)

    # Add keyword matching for common skills
    keywords = ["Python", "Java", "SQL", "AWS", "Machine Learning", "Deep Learning"]
    for keyword in keywords:
        if keyword in text:
            skills.append(keyword)

    return skills

def extract_skills_from_resume(resume_path):
    """Extracts and normalizes skills from a resume PDF using spaCy."""
    print(f"Extracting skills from resume: {resume_path}")
    try:
        # 1. Extract text from PDF
        text = extract_text_from_pdf(resume_path)
        if not text:
            messagebox.showerror("Error", "Could not extract text from resume.")
            return []

        # 2. Load spaCy model
        nlp = get_spacy_model()
        if nlp is None:
            return []

        # 3. Extract skills from text using spaCy
        skills = extract_skills_from_text_spacy(text, nlp)

        # 4. Clean up the skills (remove extra spaces, convert to lowercase)
        cleaned_skills = [skill.strip().lower() for skill in skills if skill.strip()]

        # 5. Remove duplicates
        cleaned_skills = list(set(cleaned_skills))

        # 6. Normalize the skills
        normalized_resume_skills = normalize_skill_list(cleaned_skills, SKILL_NORMALIZATION_MAP)
        print(f"Normalized resume skills: {normalized_resume_skills}")
        return normalized_resume_skills

    except Exception as e:
        print(f"Error extracting skills from resume: {e}")
        messagebox.showerror("Error", f"Could not extract skills from resume: {e}")
        return []

def find_target_job_skills(job_title_query):
    """Retrieves job skills from ChromaDB based on the job title."""
    global job_skill_collection
    print(f"Getting job skills for target role: '{job_title_query}' by querying ChromaDB...")
    model = get_embedding_model()
    if model is None:
        return []

    chroma_client = get_chroma_client()
    job_skill_collection = chroma_client.get_collection(name=CHROMA_JOB_SKILLS_COLLECTION)

    try:
        results = job_skill_collection.query(
            query_embeddings=model.encode([job_title_query]).tolist(),
            n_results=10,
            include=["metadatas"]
        )

        job_skills = set()
        if results and results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                skill = metadata.get('skill_name')
                if skill:
                    job_skills.add(skill)

        final_job_skills = sorted(list(job_skills))
        print(f"Found {len(final_job_skills)} job skills for '{job_title_query}': {final_job_skills}")
        return final_job_skills

    except Exception as e:
        print(f"Error querying ChromaDB for job skills: {e}")
        messagebox.showerror("Error", "Failed to retrieve job skills from ChromaDB.")
        return []

def analyze_skill_gap(resume_skills, job_skills):
    """Compares resume skills with job skills to find the gap."""
    print("Analyzing skill gap...")
    resume_skills_set = set(s.lower().strip() for s in resume_skills)
    job_skills_set = set(s.lower().strip() for s in job_skills)

    missing_skills = sorted(list(job_skills_set - resume_skills_set))
    print(f"Missing Skills (canonical string): {missing_skills}")
    return missing_skills

def recommend_courses(missing_skills, job_skills):
    """Recommends Coursera courses based on a list of skills."""
    global course_skill_collection, df_coursera
    print(f"Recommending courses for skills: {missing_skills}...")

    skill_weights = {}
    for skill in job_skills:
        skill_weights[skill] = 1.0  # Default weight
    # Adjust weights based on importance
    if "python" in skill_weights:
        skill_weights["python"] = 1.2

    if not missing_skills:
        print("No missing skills provided. No recommendations needed.")
        return []

    all_recommendations_data = []
    try:
        model = get_embedding_model()
        chroma_client = get_chroma_client()
        course_skill_collection = chroma_client.get_collection(name=CHROMA_COURSE_SKILLS_COLLECTION)
    except Exception as e:
        print(f"Error setting up ChromaDB for course recommendation: {e}.")
        messagebox.showerror("Error", "Failed to set up ChromaDB for course recommendation.")
        return []

    for skill_needed_str in missing_skills:
        print(f"Searching courses in ChromaDB for skill: '{skill_needed_str}'")
        skill_embedding = model.encode(skill_needed_str).tolist()  # type: ignore

        try:
            query_results = course_skill_collection.query(
                query_embeddings=[skill_embedding],
                n_results=5, # Increased n_results to get more options
                include=["metadatas"]
            )
        except Exception as e:
            print(f"Error querying ChromaDB for skill '{skill_needed_str}': {e}")
            continue

        if query_results and query_results['ids'] and query_results['ids'][0]:
            for meta in query_results['metadatas'][0]:
                course_url = meta.get("course_url")
                matched_skill_in_course = meta.get("skill_name")
                course_title = meta.get("course_title")

                # Add a check to ensure the matched skill is relevant
                if skill_needed_str.lower() not in matched_skill_in_course.lower() and matched_skill_in_course.lower() not in skill_needed_str.lower():
                    print(f"Skipping course '{course_title}' because matched skill '{matched_skill_in_course}' is not relevant to '{skill_needed_str}'")
                    continue

                if course_url:
                    course_details_row = df_coursera[df_coursera['course_url'] == course_url]
                    if not course_details_row.empty:
                        details = course_details_row.iloc[0]
                        all_recommendations_data.append({
                            "queried_skill": skill_needed_str,
                            "matched_skill_in_course": matched_skill_in_course,
                            "course_title": details.get("Title", course_title),
                            "organization": details.get("Organization", "N/A"),
                            "course_url": course_url,
                            "rating": details.get("Ratings", "N/A"),
                            "difficulty": details.get("Difficulty", "N/A")
                        })

    print(f"Found {len(all_recommendations_data)} course recommendations.")
    return all_recommendations_data

# --- GUI Functions ---
def browse_resume():
    """Opens a file dialog to select a resume PDF."""
    filename = filedialog.askopenfilename(initialdir=RESUME_DIR, title="Select a resume",
                                           filetypes=(("PDF files", "*.pdf"), ("all files", "*.*")))
    resume_path.set(filename)

def analyze_skills():
    """Analyzes skills and recommends courses based on user input."""
    resume_file = resume_path.get()
    job_title = job_title_entry.get()

    if not resume_file:
        messagebox.showerror("Error", "Please select a resume file.")
        return
    if not job_title:
        messagebox.showerror("Error", "Please enter a job title.")
        return

    # Create a progress bar
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="indeterminate")
    progress_bar.grid(row=5, column=0, columnspan=3, pady=10)
    progress_bar.start()

    # --- Extract Skills from Resume ---
    resume_skills = extract_skills_from_resume(resume_file)

    # Stop the progress bar
    progress_bar.stop()
    progress_bar.destroy()

    if not resume_skills:
        messagebox.showerror("Error", "Could not extract skills from resume.")
        return

    # --- Find Target Job Skills ---
    job_skills = find_target_job_skills(job_title)
    if not job_skills:
        messagebox.showerror("Error", "Could not find job skills for the given title.")
        return

    # --- Analyze Skill Gap ---
    missing_skills = analyze_skill_gap(resume_skills, job_skills)

    # --- Recommend Courses ---
    course_recommendations = recommend_courses(missing_skills, job_skills)

    # --- Display Results ---
    display_results(course_recommendations, resume_skills, job_skills)

def display_results(course_recommendations, resume_skills=None, job_skills=None):
    """Displays the course recommendations in the GUI."""
    result_text.delete("1.0", tk.END)  # Clear previous results

    if course_recommendations:
        result_text.insert(tk.END, "--- Recommended Courses ---\n")
        for rec in course_recommendations:
            result_text.insert(tk.END, f"For skill: '{rec['queried_skill']}'\n")
            result_text.insert(tk.END, f"  Course: {rec['course_title']} by {rec['organization']}\n")
            result_text.insert(tk.END, f"  Rating: {rec['rating']}, Difficulty: {rec['difficulty']}\n")

            # Create a clickable link
            url = rec['course_url']
            link = tk.Label(result_text, text="Link", fg="blue", cursor="hand2")
            link.bind("<Button-1>", lambda event, url=url: webbrowser.open_new(url))
            result_text.window_create(tk.END, window=link)
            result_text.insert(tk.END, "\n\n")
    else:
        if resume_skills is not None and job_skills is not None:
            result_text.insert(tk.END, "Your resume is well-aligned with the required job skills. No specific courses are recommended at this time.")
        else:
            result_text.insert(tk.END, "No courses found for the missing skills.")

def initialize_data():
    """Loads Coursera data and initializes ChromaDB."""
    global df_coursera

    global nlp
    nlp = get_spacy_model() # load spacy model

    if not load_coursera_data():
        return False

    if not CHROMA_DATA_PATH.exists():
        print("ChromaDB path does not exist. Initializing...")
        initialize_vector_database(df_coursera)
    else:
        print("ChromaDB path exists. Skipping initialization.")
    return True

# --- GUI Setup ---
root = tk.Tk()
root.title("Career Advisor")

# Configure background color for the root window
root.configure(bg="#E0F2F7")  # Light sky blue

# Get screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Find the center point
center_x = int(screen_width/2 - 600/2)  # Assuming a window width of 600
center_y = int(screen_height/2 - 400/2) # Assuming a window height of 400

# Set the position of the window to the center of the screen
root.geometry(f'600x400+{center_x}+{center_y}')

# --- Variables ---
resume_path = tk.StringVar()

# --- Style for Widgets ---
label_bg = "#E0F2F7"  # Light sky blue for labels
entry_bg = "#F0FAFF"  # Lighter shade for entry fields
button_bg = "#A8D5E2" # Sky blue for buttons
button_fg = "black"    # Button text color

# --- Widgets ---
# Resume Selection
resume_label = tk.Label(root, text="Resume:", bg=label_bg)
resume_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
resume_entry = tk.Entry(root, textvariable=resume_path, width=50, bg=entry_bg)
resume_entry.grid(row=0, column=1, padx=10, pady=5)
browse_button = tk.Button(root, text="Browse", command=browse_resume, bg=button_bg, fg=button_fg)
browse_button.grid(row=0, column=2, padx=10, pady=5)

# Job Title Input
job_title_label = tk.Label(root, text="Job Title:", bg=label_bg)
job_title_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
job_title_entry = tk.Entry(root, width=50, bg=entry_bg)
job_title_entry.grid(row=1, column=1, padx=10, pady=5)

# Analyze Button
analyze_button = tk.Button(root, text="Analyze Skills", command=analyze_skills, bg=button_bg, fg=button_fg)
analyze_button.grid(row=2, column=1, pady=10)
analyze_button_tooltip = ttk.Label(root,
                                    text="Click here to analyze your skills and get course recommendations.",
                                    background=label_bg)
analyze_button_tooltip.grid(row=2, column = 0)

# Results Display
result_label = tk.Label(root, text="Recommended Courses:", bg=label_bg)
result_label.grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
result_text = tk.Text(root, height=15, width=70, bg=entry_bg)
result_text.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

# --- Initialize Data ---
import threading

def load_data_async():
    if not load_coursera_data():
        root.destroy()
    else:
        initialize_vector_database(df_coursera)  # Initialize ChromaDB in the background

threading.Thread(target=load_data_async).start()
root.mainloop()
