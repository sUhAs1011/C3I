import pandas as pd
from pathlib import Path
from utils import SKILL_NORMALIZATION_MAP, normalize_skill_list
import spacy

# --- Define Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
JOB_DATASET_DIR = BASE_DIR / "job_dataset"
COURSERA_CSV = JOB_DATASET_DIR / "coursera_course_dataset_v3.csv"

def load_coursera_data():
    """Loads and preprocesses the Coursera dataset."""
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
        return df_coursera
    except Exception as e:
        print(f"Error loading or processing Coursera data: {e}")
        return None

def load_job_posting_data(job_posting_csv):
    """Loads and preprocesses job posting data from a CSV file."""
    print(f"Loading job posting data from {job_posting_csv}...")
    try:
        df_jobs = pd.read_csv(job_posting_csv)
        # Add your job posting data preprocessing steps here
        # Example:
        # df_jobs["Processed_Skills"] = df_jobs["Skills"].astype(str).apply(
        #     lambda x: [skill.strip().lower() for skill in x.split(",") if skill.strip()]
        # )
        # df_jobs["Canonical_Job_Skills"] = df_jobs["Processed_Skills"].apply(
        #     lambda skill_list: normalize_skill_list(skill_list, SKILL_NORMALIZATION_MAP)
        # )
        print("Job posting data loaded and processed.")
        return df_jobs
    except Exception as e:
        print(f"Error loading or processing job posting data: {e}")
        return None

def get_spacy_model():
    """Loads and returns the spaCy model."""
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")  # Or a larger model like "en_core_web_lg"
        print("spaCy model loaded.")
        return nlp
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        return None

if __name__ == "__main__":
    coursera_df = load_coursera_data()
    if coursera_df is not None:
        print("Coursera data loaded successfully.")
        # You can add code here to print some statistics about the data, e.g.,
        # print(coursera_df.head())
        # print(coursera_df.describe())
    else:
        print("Failed to load Coursera data.")

    # Example usage with job posting data (replace with your actual file)
    # job_data_file = JOB_DATASET_DIR / "your_job_postings.csv"
    # job_df = load_job_posting_data(job_data_file)
    # if job_df is not None:
    #     print("Job data loaded successfully.")
    # else:
    #     print("Failed to load job data.")