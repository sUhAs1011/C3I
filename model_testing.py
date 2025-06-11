import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from data_processing import load_coursera_data, get_spacy_model, normalize_skill_list
from utils import SKILL_NORMALIZATION_MAP
import torch
import ssl
import webbrowser
import fitz  # PyMuPDF
import threading

# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
CHROMA_JOB_SKILLS_COLLECTION = "job_skills_embeddings"
CHROMA_COURSE_SKILLS_COLLECTION = "course_skills_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RESUME_DIR = BASE_DIR  # You might want to set a default resume directory

# --- Global Variables ---
df_coursera = None
nlp = None
model = None  # Embedding model

def get_embedding_model():
    """Loads and returns the sentence transformer model."""
    global model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        # Bypass SSL certificate verification (use with caution!)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # Apply the SSL context globally
        ssl._create_default_https_context = lambda: context

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        messagebox.showerror("Error", f"Failed to load embedding model: {e}")
        return None

def get_chroma_client():
    """Initializes and returns a ChromaDB client."""
    client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
    return client

def recommend_courses(missing_skills, job_skills, course_skill_collection, model, df_coursera):
    """Recommends Coursera courses based on a list of skills."""
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

    for skill_needed_str in missing_skills:
        print(f"Searching courses in ChromaDB for skill: '{skill_needed_str}'")

        # Move the input to the same device as the model
        if torch.cuda.is_available():
            skill_embedding = model.encode(skill_needed_str)  # type: ignore
        else:
            skill_embedding = model.encode(skill_needed_str)  # type: ignore

        try:
            query_results = course_skill_collection.query(
                query_embeddings=[skill_embedding.tolist()],
                n_results=5,  # Increased n_results to get more options
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
                    print(
                        f"Skipping course '{course_title}' because matched skill '{matched_skill_in_course}' is not relevant to '{skill_needed_str}'")
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

def extract_skills_from_text_spacy(text, nlp):
    """Extracts skills from text using spaCy's NER and keyword matching."""
    skills = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECHNOLOGY", "SKILL"]:  # Example entity types
            skills.append(ent.text)

    # Add keyword matching for common skills
    keywords = ["Python", "Java", "SQL", "AWS", "Machine Learning", "Deep Learning"]
    for keyword in keywords:
        if keyword in text:
            skills.append(keyword)

    return skills

def test_model(resume_text, job_title, df_coursera):
    """Tests the model by extracting skills from a resume and recommending courses."""
    # 1. Load spaCy model
    nlp = get_spacy_model()
    if nlp is None:
        print("Failed to load spaCy model. Exiting.")
        return None

    # 2. Extract skills from resume text
    resume_skills = extract_skills_from_text_spacy(resume_text, nlp)
    cleaned_skills = [skill.strip().lower() for skill in resume_skills if skill.strip()]
    cleaned_skills = list(set(cleaned_skills))
    normalized_resume_skills = normalize_skill_list(cleaned_skills, SKILL_NORMALIZATION_MAP)
    print(f"Normalized resume skills: {normalized_resume_skills}")

    # 3. Load embedding model
    # model = get_embedding_model() # Load embedding model
    global model
    if model is None:
        print("Failed to load embedding model. Exiting.")
        return None

    # 4. Get ChromaDB client and collections
    chroma_client = get_chroma_client()
    job_skill_collection = chroma_client.get_collection(name=CHROMA_JOB_SKILLS_COLLECTION)
    course_skill_collection = chroma_client.get_collection(name=CHROMA_COURSE_SKILLS_COLLECTION)

   # 5. Find target job skills (replace with your logic to fetch job skills)
    results = job_skill_collection.query(
        query_embeddings=model.encode([job_title]).tolist(),
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
    print(f"Found {len(final_job_skills)} job skills for '{job_title}': {final_job_skills}")

    # 6. Analyze skill gap
    resume_skills_set = set(s.lower().strip() for s in normalized_resume_skills)
    job_skills_set = set(s.lower().strip() for s in final_job_skills)
    missing_skills = sorted(list(job_skills_set - resume_skills_set))
    print(f"Missing Skills (canonical string): {missing_skills}")

    # 7. Recommend courses
    course_recommendations = recommend_courses(missing_skills, final_job_skills, course_skill_collection, model, df_coursera)

    return course_recommendations

# --- GUI Functions ---
def browse_resume():
    """Opens a file dialog to select a resume PDF and extracts text."""
    filename = filedialog.askopenfilename(initialdir=RESUME_DIR, title="Select a resume",
                                           filetypes=(("PDF files", "*.pdf"), ("all files", "*.*")))
    resume_path.set(filename)
    # Extract text from PDF
    try:
        text = extract_text_from_pdf(filename)
        resume_text.set(text)
    except Exception as e:
        messagebox.showerror("Error", f"Could not read resume file: {e}")
        resume_text.set("")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        messagebox.showerror("Error", f"Error extracting text from PDF: {e}")
    return text

def analyze_skills_threaded():
    """Starts the skill analysis in a separate thread."""
    threading.Thread(target=analyze_skills).start()

def analyze_skills():
    """Analyzes skills and recommends courses based on user input."""
    job_title = job_title_entry.get()
    resume_content = resume_text.get()

    if not resume_content:
        messagebox.showerror("Error", "Please select a resume file.")
        return
    if not job_title:
        messagebox.showerror("Error", "Please enter a job title.")
        return

    # Create a progress bar
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="indeterminate")
    progress_bar.grid(row=5, column=0, columnspan=3, pady=10)
    progress_bar.start()

    # --- Run the Model ---
    course_recommendations = test_model(resume_content, job_title, df_coursera)

    # Stop the progress bar
    progress_bar.stop()
    progress_bar.destroy()

    # --- Display Results ---
    display_results(course_recommendations)

def display_results(course_recommendations):
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
        result_text.insert(tk.END, "No courses found for the missing skills.")

def initialize_data():
    """Loads Coursera data and initializes ChromaDB."""
    global df_coursera, model

    global nlp
    nlp = get_spacy_model() # load spacy model

    df = load_coursera_data()
    if df is None:
        return False
    else:
        df_coursera = df

    # Load embedding model in initialize data
    model = get_embedding_model()
    if model is None:
        return False

    return True

# --- GUI Setup ---
root = tk.Tk()
root.title("Career Advisor")

# --- Variables ---
resume_path = tk.StringVar()
resume_text = tk.StringVar()

# --- Widgets ---
resume_label = tk.Label(root, text="Resume:")
resume_label.grid(row=0, column=0, padx=10, pady=5)
resume_entry = tk.Entry(root, textvariable=resume_path, width=50)
resume_entry.grid(row=0, column=1, padx=10, pady=5)
browse_button = tk.Button(root, text="Browse", command=browse_resume)
browse_button.grid(row=0, column=2, padx=10, pady=5)

job_title_label = tk.Label(root, text="Job Title:")
job_title_label.grid(row=1, column=0, padx=10, pady=5)
job_title_entry = tk.Entry(root, width=50)
job_title_entry.grid(row=1, column=1, padx=10, pady=5)

analyze_button = tk.Button(root, text="Analyze Skills", command=analyze_skills_threaded)
analyze_button.grid(row=2, column=1, pady=10)

result_label = tk.Label(root, text="Recommended Courses:")
result_label.grid(row=3, column=0, padx=10, pady=5)
result_text = tk.Text(root, height=15, width=70)
result_text.grid(row=3, column=1, padx=10, pady=5)

# --- Initialize Data ---
if initialize_data():
    print("Data initialized successfully.")
else:
    messagebox.showerror("Error", "Failed to initialize data. Check console for details.")
    root.destroy()

root.mainloop()