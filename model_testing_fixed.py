import streamlit as st
import pandas as pd
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import fitz  # PyMuPDF
import docx
import re
import os
import json
import warnings
import cv2
import pytesseract
from PIL import Image
import io

warnings.filterwarnings('ignore')

# --- Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DATA_PATH = BASE_DIR / "chroma_data"
MODEL_SAVE_PATH = BASE_DIR / "trained_model"
DSSM_MODEL_PATH = MODEL_SAVE_PATH / "dssm_best_model.pth"
JOB_EMBEDDINGS_COLLECTION = "jobs_embeddings"
COURSE_EMBEDDINGS_COLLECTION = "courses_embeddings"
MAPPING_FILE_PATH = BASE_DIR / "job_dataset" / "job_to_course_mapping.json"

# --- Model and App Configuration ---
DSSM_CONFIG = {
    'query_dim': 384,
    'doc_dim': 384,
    "hidden_dims": [512, 256, 128],  # Must match the upgraded model_training.py
    'dropout': 0.2,                  # Must match the upgraded training code
}

# --- Utility Functions ---

SDG_DESCRIPTIONS = [
    "SDG 1: No Poverty", "SDG 2: Zero Hunger", "SDG 3: Good Health and Well-being",
    "SDG 4: Quality Education", "SDG 5: Gender Equality", "SDG 6: Clean Water and Sanitation",
    "SDG 7: Affordable and Clean Energy", "SDG 8: Decent Work and Economic Growth",
    "SDG 9: Industry, Innovation and Infrastructure", "SDG 10: Reduced Inequality",
    "SDG 11: Sustainable Cities and Communities", "SDG 12: Responsible Consumption and Production",
    "SDG 13: Climate Action", "SDG 14: Life Below Water", "SDG 15: Life on Land",
    "SDG 16: Peace, Justice and Strong Institutions", "SDG 17: Partnerships for the Goals"
]

@st.cache_resource
def get_sdg_embeddings(_model):
    return _model.encode(SDG_DESCRIPTIONS)

if 'rejected_courses' not in st.session_state:
    st.session_state.rejected_courses = set()
if 'rejected_course_embeddings' not in st.session_state:
    st.session_state.rejected_course_embeddings = []

def generate_explanation_ollama(skill_gap, course_title, job_title, course_skills=None):
    """Generates an explanation using a local Ollama instance."""
    
    skills_context = f" The course specifically teaches these skills: {course_skills}." if course_skills else ""
    
    if course_title.startswith("the sequence:"):
        prompt = (
            f"Act as a strict Senior Technical Architect. The user is transitioning to a '{job_title}' role. "
            f"They currently lack these technical skills: {', '.join(skill_gap)}. "
            f"Explain in 3-4 dense, highly technical sentences why this specific learning roadmap: '{course_title.replace('the sequence:', '')}' "
            "is the optimal sequence. \n"
            "CRITICAL RULES: \n"
            "1. NO introductory or conversational filler (Do NOT say 'I'd be happy to help' or 'To become a...'). \n"
            "2. Jump instantly into the technical analysis of the syllabus. \n"
            "3. Explicitly describe how the architectural or programmatic concepts in the earlier courses provide necessary technical prerequisites for the later courses."
        )
    else:
        prompt = (
            f"Act as a strict Senior Technical Architect. The user is transitioning to '{job_title}'. "
            f"They currently lack these technical skills: {', '.join(skill_gap)}. "
            f"Explain in 2-3 dense, highly technical sentences why the course '{course_title}' "
            f"will directly bridge this gap.{skills_context} \n"
            "CRITICAL RULES: \n"
            "1. NO introductory or conversational filler (Do NOT say 'I'd be happy to help'). \n"
            "2. Jump instantly into the technical analysis. \n"
            "3. Focus purely on the frameworks, APIs, tools, and technical architecture."
        )
    
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }, timeout=20)
        
        if response.status_code == 200:
            return response.json().get('response', 'Explanation generated but empty.')
        else:
            try:
                err_msg = response.json().get('error', response.text)
            except:
                err_msg = response.text
            return f"Error from Ollama (Status {response.status_code}): {err_msg}"
    except requests.exceptions.RequestException:
        return "Explanation unavailable (Is local Ollama running? Try 'ollama run llama3')"

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        document = docx.Document(file)
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def extract_text_with_ocr(image):
    """Extract text from image using OCR."""
    try:
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return None, "Tesseract OCR is not installed. Please install it or use text-based documents instead."
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get better text recognition
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract for OCR
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        
        return text.strip(), None
    except Exception as e:
        return None, f"Error in OCR processing: {e}"

def extract_text_from_pdf_with_ocr(file):
    """Extract text from PDF using both text extraction and OCR for images."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # First try to extract text normally
            page_text = page.get_text()
            
            # If no text found, try OCR on the page image
            if not page_text.strip():
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Use OCR on the image
                ocr_result = extract_text_with_ocr(img)
                if ocr_result and isinstance(ocr_result, tuple):
                    ocr_text, error = ocr_result
                    if ocr_text:
                        page_text = ocr_text
                    elif error:
                        st.warning(f"OCR warning: {error}")
                elif ocr_result:  # Handle old format for backward compatibility
                    page_text = ocr_result
            
            text += page_text + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF with OCR: {e}")
        return None

def extract_text_from_image(file):
    """Extract text from image file using OCR."""
    try:
        image = Image.open(file)
        ocr_result = extract_text_with_ocr(image)
        if ocr_result and isinstance(ocr_result, tuple):
            text, error = ocr_result
            if error:
                st.warning(f"OCR warning: {error}")
            return text
        return ocr_result
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

def validate_job_search(job_title, client, embedding_model):
    """Simple validation: Accept any job title that finds results in the database."""
    if not job_title or len(job_title.strip()) < 2:
        return False, "Job title too short. Please enter a valid job title."
    
    try:
        # Try to find jobs in the database
        collection = client.get_collection(name=JOB_EMBEDDINGS_COLLECTION)
        query_embedding = embedding_model.encode([job_title])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Check top 5 results
            include=["metadatas"]
        )
        
        if not results or not results['ids'][0]:
            return False, f"No jobs found for '{job_title}'. Please try a different job title."
        
        # Count how many jobs were found
        job_count = len(results['ids'][0])
        
        # If we found any jobs, accept the job title
        if job_count > 0:
            return True, f"Found {job_count} jobs for '{job_title}'"
        
        return False, f"'{job_title}' doesn't match any jobs in our database. Please try a professional job title."
        
    except Exception as e:
        return False, f"Error validating job title: {str(e)}"

def is_resume_content(text):
    """
    Validates if the extracted text actually contains resume-like content.
    """
    if not text or len(text.strip()) < 100:
        return False, "Text too short to be a resume"
    
    # Convert to lowercase for easier pattern matching
    text_lower = text.lower()
    
    # Resume indicators - look for common resume sections and keywords
    resume_indicators = [
        'resume', 'cv', 'curriculum vitae', 'professional summary', 'work experience',
        'employment history', 'education', 'skills', 'qualifications', 'certifications',
        'professional experience', 'career objective', 'summary of qualifications',
        'technical skills', 'professional skills', 'work history', 'job experience',
        'professional background', 'career summary', 'employment record',
        'experience', 'work', 'employment', 'career', 'professional', 'job', 'position',
        'role', 'responsibilities', 'achievements', 'accomplishments', 'duties'
    ]
    
    # Check if any resume indicators are present
    has_resume_indicators = any(indicator in text_lower for indicator in resume_indicators)
    
    # Look for common resume patterns - more efficient matching
    contact_patterns = ['email', 'phone', 'address', 'linkedin', 'contact']
    work_patterns = ['experience', 'employment', 'work', 'position', 'role', 'responsibilities', 'job', 'career']
    education_patterns = ['education', 'degree', 'university', 'college', 'school', 'diploma']
    
    has_contact_info = any(pattern in text_lower for pattern in contact_patterns)
    has_work_section = any(pattern in text_lower for pattern in work_patterns)
    has_education = any(pattern in text_lower for pattern in education_patterns)
    
    # Check for professional formatting (dates, company names)
    has_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))  # Years like 2020, 2021
    has_company_names = bool(re.search(r'\b(inc|corp|llc|ltd|company|corporation|technologies|solutions|systems|group|team)\b', text_lower))
    
    # Calculate a normalized score (0-10) using weighted features
    # Additional scoring criteria for more granular assessment
    has_skills_section = any(pattern in text_lower for pattern in ['skills', 'technical skills', 'professional skills', 'competencies', 'expertise'])
    has_certifications = any(pattern in text_lower for pattern in ['certification', 'certified', 'certificate', 'license', 'accreditation'])
    has_projects = any(pattern in text_lower for pattern in ['project', 'portfolio', 'achievement', 'accomplishment', 'deliverable'])
    has_achievements = any(pattern in text_lower for pattern in ['achievement', 'accomplishment', 'result', 'outcome', 'impact', 'contribution'])
    has_responsibilities = any(pattern in text_lower for pattern in ['responsibility', 'duty', 'task', 'function', 'role', 'position'])

    # Declarative weighted feature list
    weighted_features = [
        (has_resume_indicators, 2),
        (has_contact_info, 1),
        (has_work_section, 2),
        (has_education, 1),
        (has_dates, 1),
        (has_company_names, 1),
        (has_skills_section, 1),
        (has_certifications, 1),
        (has_projects, 1),
        (has_achievements, 1),
        (has_responsibilities, 1),
    ]

    raw_score = sum(weight for present, weight in weighted_features if present)
    max_raw_score = sum(weight for _, weight in weighted_features)
    score = int(round(10 * raw_score / max_raw_score)) if max_raw_score > 0 else 0
    
    # Check for irrelevant content that suggests it's not a resume
    # Only flag very specific non-resume content patterns
    irrelevant_patterns = [
        'recipe', 'cooking', 'food', 'restaurant', 'menu', 'ingredients', 'instructions',
        'novel', 'story', 'fiction', 'chapter', 'book', 'literature',
        'research paper', 'academic paper', 'thesis', 'dissertation',
        'invoice', 'receipt', 'bill', 'financial statement'
        # Removed overly broad patterns like 'form', 'application', 'contract', 'legal document'
        # as these can appear in legitimate resumes
    ]
    
    # Only flag if multiple very specific irrelevant patterns are found
    irrelevant_count = sum(1 for pattern in irrelevant_patterns if pattern in text_lower)
    has_irrelevant_content = irrelevant_count >= 3  # Increased threshold to 3 for more leniency
    
    # Final validation - be more lenient with irrelevant content if we have strong resume indicators
    if has_irrelevant_content and score < 7:  # Only reject if low score AND has irrelevant content
        return False, f"Document appears to contain non-resume content (found {irrelevant_count} irrelevant patterns) and lacks sufficient resume indicators (score: {score}/10)"
    
    # Additional check: ensure there's enough professional content - more efficient
    professional_words = ['experience', 'skills', 'education', 'work', 'employment', 'career', 'professional', 'job', 'position', 'role', 'project', 'technology', 'development', 'management', 'analysis']
    
    # More efficient counting using set intersection
    text_words = set(text_lower.split())
    professional_word_count = len(text_words.intersection(set(professional_words)))
    
    # More lenient professional content requirement
    if professional_word_count < 2:  # Reduced from 3 to 2
        return False, f"Document lacks sufficient professional content (found {professional_word_count} professional terms). Please upload a proper resume."
    
    # Score requirement for 10-point scale - reduced threshold for better acceptance
    if score >= 4:  # Reduced from 5 to 4 for more leniency
        return True, f"Resume validation passed (score: {score}/10, professional terms: {professional_word_count})"
    else:
        return False, f"Document doesn't appear to be a resume (score: {score}/10). Please upload a proper resume document."

def extract_skills_from_resume(text):
    """
    Enhanced skill extractor that captures more comprehensive skills from resume text.
    """
    if not text:
        return []
    
    # Method 1: Extract specific technical skills using regex
    technical_pattern = r"""
        \b(
            # Programming languages
            Python|Java|C\+\+|C\#|JavaScript|TypeScript|PHP|Ruby|Go|Swift|Kotlin|Scala|R|
            # Frameworks and libraries
            React|Angular|Vue\.js|Node\.js|Django|Flask|FastAPI|Spring|Express|Laravel|Streamlit|OpenCV|
            # Databases
            SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Oracle|Redis|Cassandra|Elasticsearch|
            # Cloud platforms
            AWS|Azure|GCP|Docker|Kubernetes|Terraform|Ansible|Jenkins|
            # ML/AI
            TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy|Matplotlib|Seaborn|
            Machine\sLearning|Deep\sLearning|AI|NLP|Computer\sVision|Sentence\sTransformers|OCR|Tesseract|
            # Tools and methodologies
            Git|CI/CD|DevOps|Agile|Scrum|RESTful\sAPIs|GraphQL|Microservices|Socket\sProgramming|
            # Data and analytics
            Data\sAnalysis|Data\sVisualization|Big\sData|Hadoop|Spark|Tableau|Power\sBI|
            # Other technical skills
            Linux|Windows|MacOS|Network\sEngineering|Cyber\sSecurity|Cloud\sComputing|
            UI/UX|Frontend|Backend|Full\sStack|Mobile\sDevelopment|API\sDevelopment|
            IPFS|Blockchain|UDP|TCP|SSL|TLS|Arduino|GSM|ChromaDB|Chroma|DSSM|Ollama
        )\b
    """
    
    technical_skills = re.findall(technical_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    # Method 2: Extract business and domain skills
    business_pattern = r"""
        \b(
            # Business skills
            Marketing|Sales|Finance|Accounting|HR|Human\sResources|Operations|Supply\sChain|
            Customer\sService|Business\sDevelopment|Strategy|Consulting|Project\sManagement|
            Product\sManagement|Business\sAnalysis|Market\sResearch|Competitive\sAnalysis|
            # Domain expertise
            E-commerce|Retail|Healthcare|Finance|Banking|Insurance|Education|Government|Compliance|Regulation|
            Manufacturing|Logistics|Transportation|Real\sEstate|Media|Entertainment|
            # Soft skills
            Leadership|Communication|Teamwork|Problem\sSolving|Analytical|Critical\sThinking|
            Time\sManagement|Organization|Planning|Coordination|Collaboration|
            # Web and digital
            Web\sDevelopment|Web\sApplication|Digital\sMarketing|SEO|SEM|Social\sMedia|
            Content\sCreation|Email\sMarketing|Affiliate\sMarketing|PPC|Google\sAds|
            # Operations and processes
            Operations|Process\sImprovement|Quality\sAssurance|Six\sSigma|Lean|Kaizen|
            Workflow|Automation|Efficiency|Optimization|Standardization|
            # Qualifications and certifications
            Certification|Qualification|AWS\sCertified|Azure\sCertified|Google\sCloud|CISSP|CEH|CompTIA
        )\b
    """
    
    business_skills = re.findall(business_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    # Method 3: Extract multi-word skills and phrases
    # Look for patterns like "X Y" where both words are capitalized
    multi_word_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    multi_word_skills = re.findall(multi_word_pattern, text)
    
    # Filter multi-word skills to keep only relevant ones
    relevant_multi_words = []
    for skill in multi_word_skills:
        skill_lower = skill.lower()
        
        # Skip if it contains irrelevant terms
        irrelevant_terms = [
            'alarm', 'burglar', 'sem', 'using', 'powered', 'skill', 'gap',
            'analysis', 'education', 'healthcare', 'system', 'storage',
            'best', 'practice', 'role', 'rotational', 'shift', 'strict', 'adherence',
            'position', 'rotation', 'compliance', 'policy', 'procedure', 'guideline',
            'standard', 'protocol', 'requirement', 'mandatory', 'obligatory',
            'compulsory', 'essential', 'necessary', 'important', 'critical',
            'vital', 'crucial', 'primary', 'secondary', 'tertiary', 'main',
            'major', 'minor', 'senior', 'junior', 'entry', 'level', 'mid',
            'lead', 'principal', 'associate', 'assistant', 'coordinator',
            'specialist', 'expert', 'consultant', 'advisor', 'analyst',
            'technician', 'operator', 'administrator', 'supervisor', 'manager',
            'director', 'executive', 'officer', 'representative', 'agent',
            'member', 'participant', 'contributor', 'stakeholder', 'partner',
            'collaborator', 'colleague', 'peer', 'subordinate', 'superior',
            'report', 'direct', 'indirect', 'matrix', 'functional', 'line',
            'staff', 'support', 'service', 'maintenance', 'operation',
            'production', 'manufacturing', 'assembly', 'quality', 'control',
            'assurance', 'testing', 'validation', 'verification', 'inspection',
            'audit', 'review', 'assessment', 'evaluation', 'appraisal',
            'feedback', 'input', 'output', 'result', 'outcome', 'impact',
            'effect', 'influence', 'contribution', 'value', 'benefit',
            'advantage', 'disadvantage', 'pro', 'con', 'positive', 'negative',
            'good', 'bad', 'excellent', 'poor', 'average', 'above', 'below',
            'high', 'low', 'medium', 'moderate', 'extreme', 'intense',
            'mild', 'strong', 'weak', 'powerful', 'effective', 'efficient',
            'productive', 'successful', 'unsuccessful', 'failed', 'succeeded',
            'achieved', 'accomplished', 'completed', 'finished', 'done',
            'ongoing', 'continuous', 'regular', 'periodic', 'occasional',
            'frequent', 'rare', 'common', 'uncommon', 'typical', 'atypical',
            'normal', 'abnormal', 'standard', 'non-standard', 'custom',
            'default', 'optional', 'mandatory', 'required', 'necessary',
            'essential', 'important', 'critical', 'vital', 'crucial',
            'primary', 'secondary', 'tertiary', 'main', 'major', 'minor'
        ]
        if any(term in skill_lower for term in irrelevant_terms):
            continue
            
        # Keep if it contains relevant keywords
        relevant_keywords = [
            'development', 'management', 'analysis', 'design', 'engineering',
            'marketing', 'operations', 'application', 'web', 'digital',
            'business', 'project', 'product', 'customer', 'service',
            'data', 'software', 'network', 'security', 'cloud', 'database', 
            'api', 'mobile', 'frontend', 'backend', 'machine', 'learning',
            'deep', 'artificial', 'intelligence', 'nlp', 'computer', 'vision',
            'blockchain', 'transformer', 'ipfs', 'arduino', 'udp', 'ssl', 'tls',
            'ocr', 'chroma', 'dssm', 'ollama', 'streamlit', 'mongodb', 'socket'
        ]
        if any(keyword in skill_lower for keyword in relevant_keywords):
            relevant_multi_words.append(skill)
    
    # Combine all skills
    all_skills = technical_skills + business_skills + relevant_multi_words
    
    # Enhanced filtering to remove non-skill terms
    filtered_skills = []
    for skill in all_skills:
        skill_lower = skill.lower().strip()
        
        # Skip if it's too short
        if len(skill_lower) <= 2:
            continue
            
        # Skip if it's a number or date
        if skill_lower.isdigit() or re.match(r'^\d{4}$', skill_lower):  # Years like 2023
            continue
            
        # Skip if it's a month name
        months = ['january', 'february', 'march', 'april', 'may', 'june', 
                 'july', 'august', 'september', 'october', 'november', 'december',
                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if skill_lower in months:
            continue
            
        # Skip if it's a degree level (but keep the actual degree field)
        degree_levels = ['bachelor', 'master', 'phd', 'associate', 'diploma']
        if skill_lower in degree_levels:
            continue
            
        # Skip if it's a generic job title
        generic_job_titles = ['manager', 'director', 'engineer', 'developer', 'analyst', 
                             'specialist', 'coordinator', 'assistant', 'associate', 'lead']
        if skill_lower in generic_job_titles:
            continue
            
        # Skip if it's a company type
        company_types = ['inc', 'corp', 'llc', 'ltd', 'company', 'corporation']
        if skill_lower in company_types:
            continue
            
        # Skip if it's a location
        locations = ['city', 'state', 'country', 'region', 'area', 'zone']
        if skill_lower in locations:
            continue
            
        # Skip if it's a time period
        time_periods = ['years', 'months', 'weeks', 'days', 'hours', 'minutes']
        if skill_lower in time_periods:
            continue
            
        # Skip if it's a common resume section header
        section_headers = ['summary', 'objective', 'profile', 'background', 'overview',
                          'highlights', 'achievements', 'accomplishments', 'responsibilities',
                          'education', 'experience', 'skills', 'certifications', 'references']
        if skill_lower in section_headers:
            continue
            
        # Skip if it's a personal pronoun or common word
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                       'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                       'after', 'above', 'below', 'between', 'among', 'within', 'without']
        if skill_lower in common_words:
            continue
            
        # Skip if it's a single letter or very short abbreviation
        if len(skill_lower) <= 1 or (len(skill_lower) <= 3 and skill_lower.isupper()):
            continue
            
        # Keep the skill if it passes all filters
        filtered_skills.append(skill)
    
    # Final cleanup: remove timeline phrases, months, and non-skill multi-word noise
    banned_substrings = [
        'graduat',  # graduating, graduate, graduation
        'intern',   # intern, internship
        'fresher', 'semester', 'term', 'session'
    ]
    months_full = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    months_abbr = ['jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months_set = set(months_full + months_abbr)

    # Allow common multi-word skill phrases; others beyond 3 words will be dropped
    allowed_multiword_phrases = set([
        'machine learning', 'deep learning', 'computer vision', 'data structures',
        'big data', 'cloud computing', 'software engineering', 'computer networks',
        'web technologies', 'data analytics', 'project management', 'business analysis',
        'product management', 'supply chain',
        'sentence transformers', 'deep structured semantic model', 'socket programming',
        'chroma db'
    ])

    cleaned_skills = []
    seen_normalized = set()

    for skill in filtered_skills:
        original = skill.strip()
        if not original:
            continue
        lower = original.lower()

        # Drop if contains any banned substring (timeline, graduating, intern)
        if any(bad in lower for bad in banned_substrings):
            continue

        # Tokenize to detect months/years
        tokens = re.split(r"[^a-zA-Z0-9+#./]+", lower)
        if any(tok in months_set for tok in tokens):
            # If a month token appears, treat as timeline phrase → drop
            continue
        if any(re.fullmatch(r'(19|20)\d{2}', tok or '') for tok in tokens):
            # Contains a year → likely timeline
            continue

        # Limit overly long multi-word phrases unless explicitly allowed
        word_count = len([t for t in tokens if t])
        if word_count > 3 and lower not in allowed_multiword_phrases:
            continue

        # Normalize for deduping (case-insensitive, collapse spaces/punct to single space)
        norm = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9+#./ ]+", " ", lower)).strip()
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        cleaned_skills.append(original)

    return cleaned_skills

# --- DSSM Model Definition ---
class FeatureAttention(nn.Module):
    """Self-attention mechanism to weight important semantic features."""
    def __init__(self, in_features):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, in_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights

class DSSMModel(nn.Module):
    """Deep Structured Semantic Model (DSSM) with Attention. Mirrors the training script's model."""
    
    def __init__(self, query_dim, doc_dim, hidden_dims, dropout=0.1):
        super(DSSMModel, self).__init__()
        self.query_tower = self._build_tower(query_dim, hidden_dims, dropout)
        self.query_attention = FeatureAttention(hidden_dims[-1])
        
        self.doc_tower = self._build_tower(doc_dim, hidden_dims, dropout)
        self.doc_attention = FeatureAttention(hidden_dims[-1])
        
    def _build_tower(self, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        return nn.Sequential(*layers)
    
    def forward(self, query_emb, doc_emb):
        query_features = self.query_tower(query_emb)
        query_features = self.query_attention(query_features)
        
        doc_features = self.doc_tower(doc_emb)
        doc_features = self.doc_attention(doc_features)
        
        return query_features, doc_features

# --- Backend Loading Functions ---

@st.cache_resource
def get_chroma_client():
    """Establishes a connection to the ChromaDB persistent client."""
    if not CHROMA_DATA_PATH.exists():
        st.error(f"ChromaDB data not found at {CHROMA_DATA_PATH}. Please run populate_chromadb.py.")
        return None
    return chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))

@st.cache_resource
def load_dssm_model():
    """Loads the pre-trained DSSM model from disk."""
    if not DSSM_MODEL_PATH.exists():
        st.error(f"DSSM model not found at {DSSM_MODEL_PATH}. Please run model_training.py.")
        return None
    
    model = DSSMModel(
        query_dim=DSSM_CONFIG['query_dim'],
        doc_dim=DSSM_CONFIG['doc_dim'],
        hidden_dims=DSSM_CONFIG['hidden_dims'],
        dropout=DSSM_CONFIG['dropout']
    )
    try:
        # Load on CPU, as we are doing inference
        model.load_state_dict(torch.load(DSSM_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading DSSM model: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    """Loads the sentence transformer model for creating embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_job_course_mapping():
    """Loads the job-to-course mapping from JSON file."""
    if not MAPPING_FILE_PATH.exists():
        st.warning(f"Job-to-course mapping file not found at {MAPPING_FILE_PATH}")
        return {}
    
    try:
        with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        st.success(f"Loaded {len(mapping)} job-course mappings from JSON file")
        return mapping
    except Exception as e:
        st.error(f"Error loading mapping file: {e}")
        return {}

# --- Core Logic Functions ---

def find_target_job(job_title, client, embedding_model):
    """Finds the most relevant job in ChromaDB based on a title query."""
    if not client:
        return None, "ChromaDB client not available."
    
    try:
        collection = client.get_collection(name=JOB_EMBEDDINGS_COLLECTION)
        query_embedding = embedding_model.encode([job_title])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "documents"]
        )
        
        if not results or not results['ids'][0]:
            return None, "No matching job found for this title."
            
        job_metadata = results['metadatas'][0][0]
        job_description = results['documents'][0][0]
        
        return job_metadata, job_description
    except Exception as e:
        return None, f"Error querying ChromaDB for jobs: {e}"


def get_skills_from_job_metadata(metadata):
    """
    Robustly extract skills from job metadata. Handles:
      - direct skills fields (list or comma-separated string)
      - various 'Key Skills' or 'Requirements' patterns inside job_text/description
      - fallback: heuristic extraction using existing resume skill extractor
    """
    # Normalize keys for case-insensitive lookup
    metadata_lower = {k.lower(): v for k, v in metadata.items()}

    # 1) Direct skill fields (many datasets store lists or comma-strings)
    possible_skill_keys = ["key skills", "skills", "job_skills", "required_skills", "preferred_skills", "skills_covered"]
    for key in possible_skill_keys:
        val = None
        # try lower-case key first on normalized dict
        if key in metadata_lower:
            val = metadata_lower[key]
        # also try original-case keys if provided
        elif key in metadata:
            val = metadata[key]
        if val:
            # lists
            if isinstance(val, (list, tuple, set)):
                skills = sorted(list(set([str(s).strip().lower() for s in val if str(s).strip()])))
                return skills
            # string: comma separated or newline separated
            if isinstance(val, str):
                # try to split on commas/semicolons/newlines
                parts = re.split(r'[,\n;|•]', val)
                parts = [p.strip().lower() for p in parts if p.strip()]
                if parts:
                    return sorted(list(set(parts)))

    # 2) Look inside job_text / description for common "Key Skills" patterns
    job_text = ""
    for k in ["job_text", "jobdescription", "job_description", "description", "details", "summary"]:
        if k in metadata_lower and metadata_lower[k]:
            job_text = str(metadata_lower[k])
            break
        if k in metadata and metadata[k]:
            job_text = str(metadata[k])
            break

    if job_text:
        # try several regexes to capture lists after common headings
        regexes = [
            r"key\s+skills\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)",
            r"skills\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)",
            r"requirements\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)",
            r"technical\s+skills\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)",
            r"must\s+have\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)",
            r"responsibilities\s*[:\-]\s*(?P<skills>.+?)(?:\n|$)"
        ]
        for rx in regexes:
            m = re.search(rx, job_text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                skills_part = m.group("skills")
                # stop at next heading or period if it's long
                skills_part = re.split(r'[\.\n\r]{1,}', skills_part)[0]
                parts = re.split(r'[,;|/•\-]+', skills_part)
                parts = [p.strip().lower() for p in parts if p.strip()]
                if parts:
                    return sorted(list(set(parts)))

        # If no heading matched, attempt to extract multi-word skill-like tokens with the resume skill extractor
        try:
            # reuse your existing extractor function which returns cleaned skill strings
            inferred_skills = extract_skills_from_resume(job_text)
            if inferred_skills:
                # normalize to lowercase
                inferred_skills = sorted(list(set([s.strip().lower() for s in inferred_skills if s.strip()])))
                return inferred_skills
        except Exception:
            pass

    # 3) Try JSON mapping fallback (if you have a precomputed mapping file)
    try:
        json_mapping = load_job_course_mapping()
        if json_mapping:
            job_title = metadata.get('job_title', metadata.get('Job Title', ''))
            if not job_title and 'job_text' in metadata_lower:
                job_text_local = metadata_lower['job_text']
                # try to pull title heuristically
                m_title = re.search(r'job\s*title\s*[:\-]\s*(.+?)(?:\n|$)', str(job_text_local), flags=re.IGNORECASE)
                if m_title:
                    job_title = m_title.group(1).strip()

            for job_entry in json_mapping:
                if job_entry.get('job_title', '').strip().lower() == str(job_title).strip().lower():
                    required_skills = job_entry.get('required_skills', [])
                    if required_skills:
                        return sorted(list(set([str(s).strip().lower() for s in required_skills if str(s).strip()])))
    except Exception:
        pass

    # 4) Last resort: return empty list so the caller knows there are no extracted skills
    return []

def calculate_dynamic_market_weight(skill, client, embedding_model, sample_size=5000):
    """
    Calculates the market demand of a skill mathematically by measuring its semantic
    centrality to the job market vector space (Approach C: DSSM-Derived Implicit Demand).
    """
    try:
        # 1. Encode the skill concept
        query_text = f"Required skill: {skill}"
        skill_embedding = torch.tensor(embedding_model.encode([query_text]), dtype=torch.float32)
        
        # 2. Get job embeddings from ChromaDB
        job_collection = client.get_collection(name=JOB_EMBEDDINGS_COLLECTION)
        job_data = job_collection.get(include=["embeddings"])
        
        if not job_data or not job_data.get('embeddings'):
            return 1.0
            
        embeddings_list = job_data['embeddings']
        
        # Sample for performance if the dataset is extremely large
        if len(embeddings_list) > sample_size:
            import random
            embeddings_list = random.sample(embeddings_list, sample_size)
            
        all_job_embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        
        # 3. Calculate cosine similarity to all sampled jobs
        similarities = F.cosine_similarity(skill_embedding, all_job_embeddings)
        
        # 4. Compute mean similarity (Centrality in the job market latent space)
        mean_sim = torch.mean(similarities).item()
        
        # 5. Min-Max Normalization to multiplier range [1.0, 2.0]
        # Cross-domain cosine similarities for all-MiniLM-L6-v2 typically fall between 0.1 and 0.4
        min_bound = 0.10
        max_bound = 0.40
        
        clamped_sim = max(min_bound, min(mean_sim, max_bound))
        multiplier = 1.0 + ((clamped_sim - min_bound) / (max_bound - min_bound))
        
        return round(multiplier, 2)
        
    except Exception as e:
        print(f"Error calculating dynamic market weight for {skill}: {e}")
        return 1.0

def find_course_recommendations(skill_gap, dssm_model, client, embedding_model, top_n=3, use_json_mapping=True):
    """Finds the best course recommendations for a list of missing skills."""
    if not all([client, dssm_model, embedding_model]):
        return {}, "A required model or client is missing."

    # Try to use JSON mapping first if available and requested
    if use_json_mapping:
        json_mapping = load_job_course_mapping()
        if json_mapping:
            st.info("Using pre-computed job-course mappings for recommendations...")
            return find_course_recommendations_from_json(skill_gap, json_mapping, client, top_n)
    
    # Fallback to DSSM model-based recommendations
    st.info("Using DSSM model for course recommendations...")
    return find_course_recommendations_from_dssm(skill_gap, dssm_model, client, embedding_model, top_n)

def find_course_recommendations_from_json(skill_gap, json_mapping, client, top_n=3):
    """Find course recommendations using the pre-computed JSON mapping."""
    recommendations = {}
    
    # Get all courses from ChromaDB for metadata lookup
    course_collection = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
    all_courses = course_collection.get(include=["metadatas"])
    
    # Create a mapping from course_id to metadata
    course_id_to_meta = {}
    for i, course_id in enumerate(all_courses['ids']):
        course_id_to_meta[course_id] = all_courses['metadatas'][i]
    
    for skill in skill_gap:
        skill_recs = []
        seen_courses = set()  # Track unique courses to avoid duplicates
        
        # Search through all job entries in the mapping
        for job_entry in json_mapping:
            job_title = job_entry.get('job_title', '').lower()
            job_skills = job_entry.get('required_skills', [])
            
            # Check if this job is related to the skill
            skill_lower = skill.lower()
            job_related = (
                skill_lower in job_title or 
                any(skill_lower in str(s).lower() for s in job_skills) or
                any(skill_lower in course_info.get('title', '').lower() 
                    for course_info in job_entry.get('top_courses', []))
            )
            
            if job_related:
                # Get courses for this job
                for course_info in job_entry.get('top_courses', []):
                    course_id = course_info.get('course_id')
                    course_title = course_info.get('title', 'Unknown Course')
                    similarity = course_info.get('similarity', 0.0)
                    
                    # Skip if we've already seen this course
                    if course_title in seen_courses:
                        continue
                    seen_courses.add(course_title)
                    
                    # Check if course metadata is available
                    if course_id in course_id_to_meta:
                        meta = course_id_to_meta[course_id]
                        organization = meta.get('organization', 'Coursera')
                        # Normalize unknowns to Coursera
                        if organization in ('Unknown Organization', None, ''):
                            organization = meta.get('Organization', meta.get('organization_name', 'Coursera')) or 'Coursera'
                    else:
                        # Default to Coursera for course recommendations
                        organization = 'Coursera'
                    
                    skill_recs.append({
                        "title": course_title,
                        "organization": organization,
                        "similarity": similarity,
                        "source": "JSON Mapping"
                    })
        
        # Sort by similarity and take top N
        skill_recs.sort(key=lambda x: x['similarity'], reverse=True)
        recommendations[skill] = skill_recs[:top_n]
    
    return recommendations, None

def find_course_recommendations_from_dssm(skill_gap, dssm_model, client, embedding_model, top_n=5):
    """Find course recommendations using the DSSM model."""
    course_collection = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
    all_courses = course_collection.get(include=["metadatas", "embeddings"])

    recommendations = {}
    
    # Pre-calculate semantic SDG embeddings
    sdg_embeddings = get_sdg_embeddings(embedding_model)
    sdg_tensor = torch.tensor(sdg_embeddings, dtype=torch.float32)
    
    # Create embeddings for all courses once
    course_ids = all_courses['ids']
    course_embeddings = torch.tensor(all_courses['embeddings'], dtype=torch.float32)
    
    with st.spinner("Searching for the best courses using DSSM model with SDG & Negative Vectors..."):
        for skill in skill_gap:
            # Use the skill as the "job query"
            query_text = f"A course about {skill}"
            query_embedding = torch.tensor(embedding_model.encode([query_text]), dtype=torch.float32)
            
            # --- Phase 2: Apply Market Demand Proxy ---
            if getattr(st.session_state, 'use_market_proxy', True):
                applicable_mult = calculate_dynamic_market_weight(skill, client, embedding_model)
                st.toast(f"📈 Market Demand (Proxy) for '{skill}': {applicable_mult}x multiplier", icon="📊")
                # Mathematically boost the vector magnitude if high demand
                query_embedding = query_embedding * applicable_mult
            
            # --- Apply User Self-Correction Layer (Negative Vectors) ---
            if st.session_state.rejected_course_embeddings:
                alpha = 0.5  # Rejection penalty weight
                for rej_emb in st.session_state.rejected_course_embeddings:
                    # Mathematically push query away from rejected feature space
                    query_embedding -= alpha * torch.tensor(rej_emb, dtype=torch.float32).unsqueeze(0)
            
            # Pass through the DSSM model
            with torch.no_grad():
                dssm_model.eval()
                # The query tower expects a batch, so we need to repeat the query embedding
                repeated_query_emb = query_embedding.repeat(course_embeddings.shape[0], 1)
                
                job_features, course_features = dssm_model(repeated_query_emb, course_embeddings)
                
                # Calculate cosine similarity on the DSSM output
                similarities = F.cosine_similarity(job_features, course_features)
            
            # Get top N recommendations
            top_indices = torch.topk(similarities, k=top_n).indices.tolist()
            
            skill_recs = []
            for idx in top_indices:
                meta = all_courses['metadatas'][idx]
                org = meta.get('organization')
                if org in ('Unknown Organization', 'Unknown', None, ''):
                    org = meta.get('Organization', meta.get('organization_name', 'Coursera')) or 'Coursera'
                
                # --- Apply SDG Integration Layer (Semantic Similarity) ---
                course_emb = all_courses['embeddings'][idx]
                c_tensor = torch.tensor(course_emb, dtype=torch.float32).unsqueeze(0)
                sdg_sims = F.cosine_similarity(c_tensor, sdg_tensor)
                best_sdg_idx = torch.argmax(sdg_sims).item()
                best_sdg = SDG_DESCRIPTIONS[best_sdg_idx]
                
                rec = {
                    "title": meta.get('course_title', 'Unknown Course'),
                    "organization": org,
                    "similarity": similarities[idx].item(),
                    "source": "DSSM Model",
                    "embedding": course_emb,  # Store for negative vectors if rejected
                    "sdg": best_sdg
                }
                skill_recs.append(rec)
            
            recommendations[skill] = skill_recs
            
    return recommendations, None

# --- Main Application ---
st.set_page_config(page_title="Career Copilot", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Career Copilot: Your Skill & Course Advisor")
st.markdown("""
Welcome to Career Copilot! This tool helps you bridge the gap between your current skills and your dream job.

**How it works:**
1.  **Upload your resume** (PDF, DOCX, or scanned image) - Must be a professional resume document
2.  **Enter a job title** you're interested in
3.  We'll analyze your skills, identify gaps, and recommend relevant courses

**📋 Resume Requirements:**
- Must contain **work experience** or employment history
- Should include **education** background and **skills**
- Must be a **completed resume**, not a template or form
- Should have **professional content** (career-related, not personal documents)

**💡 Tips for Success:**
- Upload your **actual completed resume** with real work experience
- Ensure it contains **specific skills** and **job responsibilities**
- Avoid blank templates or application forms
- Use PDF or DOCX format for best results

**Important:** Only upload actual resume documents containing work experience, education, and skills. Other documents will be rejected.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Step 1: Your Information")
    
    # Resume Upload
    st.subheader("📄 Resume Upload")
    resume_file = st.file_uploader(
        "Upload Your Resume", 
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        help="Upload a professional resume document (PDF, DOCX, or scanned image). The document should contain work experience, education, and skills."
    )
    
    # OCR Toggle
    try:
        pytesseract.get_tesseract_version()
        ocr_available = True
        ocr_help = "Enable OCR to extract text from scanned PDFs or images. Recommended for better skill detection."
    except Exception:
        ocr_available = False
        ocr_help = "⚠️ OCR not available - Tesseract not installed. Install Tesseract for image text extraction."
    
    use_ocr = st.checkbox("Use OCR for better text extraction", value=ocr_available, 
                          help=ocr_help, disabled=not ocr_available)
    
    if not ocr_available:
        st.warning("""
        **OCR Not Available**: Tesseract OCR is not installed on your system.
        
        **To enable OCR:**
        - **Windows**: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
        - **Mac**: `brew install tesseract`
        - **Linux**: `sudo apt-get install tesseract-ocr`
        
        **Alternative**: Use PDF or DOCX files with embedded text instead of scanned images.
        """)
    
    # Resume validation info
    st.info("""
    **Resume Requirements:**
    - Must contain work experience or employment history
    - Should include education background
    - Must have skills and qualifications
    - Should contain contact information
    - Minimum 100 characters of text
    
    **Common Resume Sections:**
    - Professional Summary/Objective
    - Work Experience/Employment History
    - Education & Certifications
    - Skills (Technical & Soft Skills)
    - Projects & Achievements
    
    **File Format Support:**
    - **PDF/DOCX**: Best support, text extraction works reliably
    - **Images (PNG/JPG)**: Requires Tesseract OCR installation
    - **Scanned PDFs**: OCR recommended for better accuracy
    """)
    
    # Additional help for validation issues
    st.warning("""
    **If your resume is rejected:**
    1. Ensure it contains actual work experience and skills
    2. Check that it's not a template or form
    3. Make sure it has professional content (not recipes, stories, etc.)
    4. Try expanding the Debug section below to see scoring details
    """)
    
    # Job Title Input
    job_title_input = st.text_input("Enter Desired Job Title", placeholder="e.g., Senior Data Scientist")

    # Recommendation Method Toggle
    st.header("Step 2: Recommendation Method")
    use_json_mapping = st.checkbox("Use Pre-computed Mappings (Faster)", value=True, 
                                  help="Use the job-to-course mapping JSON file for faster recommendations. Uncheck to use the DSSM model.")

    # Phase 2: User Profiling (Sidebar)
    st.sidebar.header("🎓 Learning Persona Checkout")
    st.sidebar.markdown("Help us tailor your learning path.")
    st.session_state.learning_style = st.sidebar.selectbox("How do you learn best?", 
                                         ["Visual (Videos & Lectures)", 
                                          "Theoretical (Deep Reading)", 
                                          "Practical (Project-Based)"])
                                          
    st.sidebar.header("📈 Market Proxy")
    st.session_state.use_market_proxy = st.sidebar.checkbox("Apply Market Demand Weighting", value=True,
                                          help="Boosts recommendations for highly sought-after skills.")

    # Analyze Button
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False
        
    if st.button("Analyze & Recommend Courses", use_container_width=True):
        st.session_state.analysis_triggered = True

# --- Main Content Area ---
if st.session_state.analysis_triggered and resume_file and job_title_input:
    # Load models and client first for validation
    client = get_chroma_client()
    embedding_model = get_embedding_model()
    
    if not client or not embedding_model:
        st.error("Could not load necessary models. Please check the console for errors.")
        st.stop()
    
    # Validate job title using actual database search
    is_valid, validation_message = validate_job_search(job_title_input, client, embedding_model)
    
    if not is_valid:
        st.error("❌ *Job Title Validation Failed*")
        st.warning(validation_message)
        st.markdown("""
        *Try these examples:*
        - Data Scientist, Machine Learning Engineer
        - Software Developer, Full Stack Engineer  
        - DevOps Engineer, Cloud Architect
        - Product Manager, Business Analyst
        - UX Designer, Frontend Developer
        """)
        st.stop()
    
    # 1. Process Resume
    st.header("📄 Your Resume Analysis")
    
    # Check file size - resumes should typically be at least a few KB
    if resume_file.size < 1024:  # Less than 1KB
        st.warning("⚠️ **File Size Warning**: This file is very small and may not contain a complete resume.")
        st.info("Typical resume files are 5KB - 2MB. Very small files might be incomplete or contain minimal content.")
    
    resume_text = ""
    
    with st.spinner("Extracting text from your resume..."):
        if resume_file.type == "application/pdf":
            if use_ocr and ocr_available:
                resume_text = extract_text_from_pdf_with_ocr(resume_file)
                if resume_text:
                    st.success("✅ Text extracted using OCR for better accuracy")
                else:
                    st.warning("⚠️ OCR extraction failed, falling back to text extraction")
                    resume_text = extract_text_from_pdf(resume_file)
                    if resume_text:
                        st.success("✅ Text extracted from PDF (fallback method)")
            else:
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    st.success("✅ Text extracted from PDF")
                    
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
            if resume_text:
                st.success("✅ Text extracted from DOCX")
                
        elif resume_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            if use_ocr and ocr_available:
                resume_text = extract_text_from_image(resume_file)
                if resume_text:
                    st.success("✅ Text extracted from image using OCR")
                else:
                    st.error("❌ Failed to extract text from image")
                    st.info("**Possible solutions:**")
                    st.info("1. Install Tesseract OCR (see instructions above)")
                    st.info("2. Use a PDF or DOCX file instead")
                    st.info("3. Ensure the image contains clear, readable text")
                    st.stop()
            else:
                st.error("❌ Cannot process image files without OCR")
                st.info("**To process images:**")
                st.info("1. Install Tesseract OCR (see instructions above)")
                st.info("2. Or convert your image to PDF/DOCX format")
                st.info("3. Or use a text-based document instead")
                st.stop()

    if resume_text:
        # Validate that the extracted text is actually resume content
        is_valid_resume, validation_message = is_resume_content(resume_text)
        
        # Debug mode - show detailed validation results
        with st.expander("🔍 Debug: Resume Validation Details"):
            st.write("**Validation Results:**")
            __fixed_re_bool_0 = bool(re.search(r"\\b(19|20)\\d{2}\\b", resume_text))
            st.write("- Contains dates: " + str(__fixed_re_bool_0))
            __fixed_re_bool_1 = bool(re.search(r"\\b(19|20)\\d{2}\\b", resume_text))
            st.write("- Contains dates: " + str(__fixed_re_bool_1))
            text_lower = resume_text.lower() if isinstance(resume_text, str) else ""
            has_resume_indicators = any(indicator in text_lower for indicator in [
            'resume', 'cv', 'curriculum vitae', 'professional summary', 'work experience',
            'employment history', 'education', 'skills', 'certifications', 'projects', 'achievements'])

            has_contact_info = any(pattern in text_lower for pattern in ['email', 'phone', 'address', 'linkedin', 'contact'])
            has_work_section = any(pattern in text_lower for pattern in ['experience', 'employment', 'work', 'position', 'role', 'responsibilities', 'job', 'career'])
            has_education = any(pattern in text_lower for pattern in ['education', 'degree', 'university', 'college', 'school', 'diploma'])
            has_dates = bool(re.search(r'\b(19|20)\d{2}\b', resume_text if isinstance(resume_text, str) else ""))

            has_company_names = bool(re.search(r'\b(inc|corp|llc|ltd|company|corporation|technologies|solutions|systems|group|team)\b', text_lower))

            has_skills_section = any(pattern in text_lower for pattern in ['skills', 'technical skills', 'professional skills', 'competencies', 'expertise'])
            has_certifications = any(pattern in text_lower for pattern in ['certification', 'certified', 'certificate', 'license', 'accreditation'])
            has_projects = any(pattern in text_lower for pattern in ['project', 'portfolio', 'achievement', 'accomplishment', 'deliverable'])
            has_achievements = any(pattern in text_lower for pattern in ['achievement', 'accomplishment', 'result', 'outcome', 'impact', 'contribution'])
            has_responsibilities = any(pattern in text_lower for pattern in ['responsibility', 'duty', 'task', 'function', 'role', 'position'])
            score = 0
            if has_contact_info: 
                score += 1
                st.write(f"- Contact info: +1 point (Total: {score})")
            if has_work_section: 
                score += 2
                st.write(f"- Work section: +2 points (Total: {score})")
            if has_education: 
                score += 1
                st.write(f"- Education: +1 point (Total: {score})")
            if has_dates: 
                score += 1
                st.write(f"- Dates: +1 point (Total: {score})")
            if has_company_names: 
                score += 1
                st.write(f"- Company names: +1 point (Total: {score})")
            
            # Additional scoring criteria
            has_skills_section = any(pattern in text_lower for pattern in ['skills', 'technical skills', 'professional skills', 'competencies', 'expertise'])
            if has_skills_section: 
                score += 1
                st.write(f"- Skills section: +1 point (Total: {score})")
            
            has_certifications = any(pattern in text_lower for pattern in ['certification', 'certified', 'certificate', 'license', 'accreditation'])
            if has_certifications: 
                score += 1
                st.write(f"- Certifications: +1 point (Total: {score})")
            
            has_projects = any(pattern in text_lower for pattern in ['project', 'portfolio', 'achievement', 'accomplishment', 'deliverable'])
            if has_projects: 
                score += 1
                st.write(f"- Project experience: +1 point (Total: {score})")
            
            # New scoring criteria
            has_achievements = any(pattern in text_lower for pattern in ['achievement', 'accomplishment', 'result', 'outcome', 'impact', 'contribution'])
            if has_achievements: 
                score += 1
                st.write(f"- Achievements: +1 point (Total: {score})")
            
            has_responsibilities = any(pattern in text_lower for pattern in ['responsibility', 'duty', 'task', 'function', 'role', 'position'])
            if has_responsibilities: 
                score += 1
                st.write(f"- Responsibilities: +1 point (Total: {score})")
            
            st.write(f"**Final Score: {score}/10**")
            
            # Show what patterns were flagged as irrelevant
            irrelevant_patterns = [
                'recipe', 'cooking', 'food', 'restaurant', 'menu', 'ingredients', 'instructions',
                'novel', 'story', 'fiction', 'chapter', 'book', 'literature',
                'research paper', 'academic paper', 'thesis', 'dissertation',
                'invoice', 'receipt', 'bill', 'financial statement'
            ]
            found_irrelevant = [pattern for pattern in irrelevant_patterns if pattern in text_lower]
            if found_irrelevant:
                st.write(f"**⚠️ Irrelevant patterns found:** {found_irrelevant}")
                st.write("These patterns suggest non-resume content, but may be false positives.")
            else:
                st.write("**✅ No irrelevant patterns found**")
        
        if not is_valid_resume:
            st.error("❌ *Resume Validation Failed*")
            st.warning(validation_message)
            
            # Provide more helpful guidance
            st.markdown("""
            **🔍 What went wrong?**
            The document you uploaded doesn't appear to be a professional resume.
            
            **📋 What we're looking for:**
            - **Work Experience**: Job titles, companies, dates, responsibilities
            - **Education**: Degrees, schools, graduation dates
            - **Skills**: Technical skills, software, tools, languages
            - **Professional Content**: Career-related information, not personal documents
            
            **💡 How to fix this:**
            1. **Upload your actual resume** (not a template or form)
            2. **Ensure it contains work history** and professional skills
            3. **Check the Debug section above** to see detailed scoring
            4. **Try a different resume file** if you have multiple versions
            
            **📄 Good resume examples:**
            - Professional CV with work experience
            - Resume with skills and employment history
            - Document showing your career background
            """)
            
            # Show common issues
            with st.expander("🚫 Common Issues & Solutions"):
                st.markdown("""
                **❌ Document Types That Won't Work:**
                - Job application forms
                - Blank resume templates
                - Personal letters or stories
                - Recipes or cooking instructions
                - Academic papers or research
                - Financial documents or invoices
                
                **✅ Document Types That Work:**
                - Completed professional resumes
                - CVs with work experience
                - Career summaries with skills
                - Professional profiles
                """)
            
            st.stop()
        
        # Show validation success
        st.success(f"✅ {validation_message}")
        
        # Show extracted text for debugging
        with st.expander("🔍 Debug: Extracted Text (First 500 chars)"):
            st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        
        user_skills = extract_skills_from_resume(resume_text)
        
        if user_skills:
            st.subheader("Your Extracted Skills:")
            # Display skills in numbered list format
            skills_text = ""
            for i, skill in enumerate(user_skills, 1):
                skills_text += f"{i}. {skill}\n"
            st.info(skills_text)
        else:
            st.warning("Could not extract any skills from your resume. Please ensure it contains relevant technical and professional skills.")

    else:
        st.error("Failed to read the resume file. Please try a different file.")
        st.markdown("""
        **Common issues:**
        - File is corrupted or password-protected
        - File is too large (>50MB)
        - File contains only images without text
        - File is not a resume document
        """)

    # Placeholder for next steps
    if resume_text and user_skills:
        st.header("🔍 Job & Skill Gap Analysis")
        
        # Load DSSM model (client and embedding_model already loaded)
        dssm_model = load_dssm_model()

        if dssm_model:
            with st.spinner(f"Searching for '{job_title_input}' in our database..."):
                target_job_meta, job_desc = find_target_job(job_title_input, client, embedding_model)
            
            if target_job_meta:
                # Additional check for job relevance
                job_title_found = target_job_meta.get('Job Title', target_job_meta.get('job_title', ''))
                job_text = target_job_meta.get('job_text', '')
                
                # Check if the found job is actually relevant
                job_title_lower = job_title_input.lower()
                found_job_lower = job_title_found.lower()
                
                # Define clearly irrelevant terms
                irrelevant_terms = ['burger', 'pizza', 'food', 'restaurant', 'cooking', 'culinary', 'chef', 'waiter', 'server', 'cashier', 'janitor', 'cleaner', 'driver', 'delivery']
                
                # Check if the search term is clearly irrelevant
                if any(term in job_title_lower for term in irrelevant_terms):
                    st.error("❌ *Invalid Job Title*")
                    st.warning(f"'{job_title_input}' is not a professional job title. Please enter a valid job title like:")
                    st.markdown("""
                    - *Data Scientist* or *Data Analyst*
                    - *Software Engineer* or *Developer*
                    - *Product Manager* or *Business Analyst*
                    - *DevOps Engineer* or *Cloud Architect*
                    """)
                    st.stop()
                
                # Check if the found job is relevant
                relevant_keywords = ['analyst', 'data', 'scientist', 'engineer', 'developer', 'manager', 'specialist', 'consultant', 'architect', 'designer', 'administrator', 'coordinator', 'director', 'lead']
                
                job_is_relevant = (
                    any(keyword in job_title_lower for keyword in relevant_keywords) or
                    any(keyword in found_job_lower for keyword in relevant_keywords) or
                    'data' in job_title_lower or 'data' in found_job_lower
                )
                
                if not job_is_relevant:
                    st.error("❌ *No Relevant Jobs Found*")
                    st.warning(f"'{job_title_input}' doesn't match any professional jobs in our database. Please try a different job title.")
                    st.stop()
                
                st.subheader(f"Best Match Found: {job_title_found}")
                
                # --- DEBUGGING: Show full metadata ---
                with st.expander("🔍 Debug: Full Job Metadata"):
                    st.json(target_job_meta)
                # --- END DEBUGGING ---
                
                # Omit job description display per request

                required_skills = get_skills_from_job_metadata(target_job_meta)
                
                # --- DEBUGGING: Show skill extraction process ---
                st.write(f"*Debug: Extracted skills count: {len(required_skills)}*")
                if required_skills:
                    st.write(f"*Debug: Skills found: {required_skills}*")
                else:
                    st.write("*Debug: No skills extracted - checking metadata structure...*")
                    metadata_lower = {k.lower(): v for k, v in target_job_meta.items()}
                    st.write(f"*Debug: Available keys (lowercase): {list(metadata_lower.keys())}*")
                    
                    # Check for job_text field specifically
                    job_text = metadata_lower.get("job_text", "")
                    if job_text:
                        st.write(f"*Debug: job_text field found (first 200 chars): {str(job_text)[:200]}...*")
                    else:
                        st.write("*Debug: No job_text field found*")
                # --- END DEBUGGING ---
                
                if required_skills:
                    st.subheader("Required Skills for this Role:")
                    st.info(", ".join(required_skills))

                    # --- PHASE 3: Contextual User Resegmentation ---
                    st.subheader("Contextual Resume Analysis")
                    with st.spinner("Segmenting resume for this specific role..."):
                        if required_skills and user_skills:
                            # Encode required and user skills
                            req_emb = torch.tensor(embedding_model.encode(required_skills), dtype=torch.float32)
                            user_emb = torch.tensor(embedding_model.encode(user_skills), dtype=torch.float32)
                            
                            # Encode job title for broader context
                            job_title_str = target_job_meta.get('job_title', target_job_meta.get('Job Title', job_title_found))
                            title_emb = torch.tensor(embedding_model.encode([job_title_str]), dtype=torch.float32)
                            
                            # Calculate similarity to required skills AND job title
                            # user_emb is [U, 384], req_emb is [R, 384]
                            sim_matrix = F.cosine_similarity(user_emb.unsqueeze(1), req_emb.unsqueeze(0), dim=2)
                            max_sims, _ = torch.max(sim_matrix, dim=1)
                            title_sims = F.cosine_similarity(user_emb, title_emb, dim=1)
                            
                            relevant_user_skills = []
                            masked_user_skills = []
                            
                            for i, skill in enumerate(user_skills):
                                is_relevant = False
                                skill_lower = skill.lower()
                                
                                # 1. Semantic similarity to any specified required skill
                                if max_sims[i].item() >= 0.28: 
                                    is_relevant = True
                                # 2. Broad semantic similarity to the Job Title itself
                                elif title_sims[i].item() >= 0.30:
                                    is_relevant = True
                                # 3. Explicit keyword overlap fallback (Using Word Boundaries to prevent partial matches)
                                else:
                                    import re
                                    for req in required_skills:
                                        req_l = req.lower()
                                        if skill_lower == req_l:
                                            is_relevant = True
                                            break
                                        if re.search(r'\b' + re.escape(skill_lower) + r'\b', req_l):
                                            is_relevant = True
                                            break
                                        if re.search(r'\b' + re.escape(req_l) + r'\b', skill_lower):
                                            is_relevant = True
                                            break
                                    
                                if is_relevant:
                                    relevant_user_skills.append(skill)
                                else:
                                    masked_user_skills.append(skill)
                                    
                            # Override user_skills for adaptive downstream gap analysis
                            user_skills = relevant_user_skills
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"🎯 **Highlighted Relevant ({len(relevant_user_skills)})**")
                                st.caption("These skills from your resume apply to this job:")
                                st.write(", ".join(relevant_user_skills) if relevant_user_skills else "None")
                            with col2:
                                st.info(f"👻 **Masked Irrelevant ({len(masked_user_skills)})**")
                                st.caption("These skills were dynamically hidden to reduce noise:")
                                st.write(", ".join(masked_user_skills) if masked_user_skills else "None")

                    # Calculate skill gap
                                skill_gap = [skill for skill in required_skills if skill not in user_skills]

                    if not skill_gap:
                        st.balloons()
                        st.success("🎉 *Congratulations!* Your skills are a great match for this role. No immediate skill gap found.")
                    else:
                        st.subheader("Your Skill Gap:")
                        st.warning(", ".join(skill_gap))

                        st.header("📚 Recommended Courses to Bridge the Gap")
                        
                        # Get job-specific recommendations instead of skill-specific ones
                        job_title = target_job_meta.get('job_title', target_job_meta.get('Job Title', job_title_input))
                        
                        if use_json_mapping:
                            json_mapping = load_job_course_mapping()
                            if json_mapping:
                                # Find this specific job in the JSON mapping
                                job_recommendations = []
                                for job_entry in json_mapping:
                                    if job_entry.get('job_title', '').lower() == job_title.lower():
                                        job_recommendations = job_entry.get('top_courses', [])
                                        break
                                
                                if job_recommendations:
                                    st.success("✅ *Most Relevant Courses for This Role:*")
                                    
                                    # Filter rejected courses
                                    valid_job_recs = [r for r in job_recommendations if r.get('title', 'Unknown Course') not in st.session_state.rejected_courses]
                                    
                                    if not valid_job_recs:
                                         st.info("You've rejected all top recommendations for this role.")
                                         
                                    # --- Phase 2: Skill Roadmap Flowchart ---
                                    sorted_recs = sorted(valid_job_recs[:5], key=lambda x: 1 if any(w in x.get('title','').lower() for w in ['intro','basic','beginner','found']) else (3 if any(w in x.get('title','').lower() for w in ['advanced','pro','master', 'specialization']) else 2))
                                    
                                    for i, course_info in enumerate(sorted_recs, 1):  # Show top 5 sorted by difficulty
                                        course_title = course_info.get('title', 'Unknown Course')
                                        organization = course_info.get('organization', 'Coursera')
                                        if organization in ('Unknown Organization', 'Unknown', None, ''):
                                            organization = 'Coursera'
                                        similarity = course_info.get('similarity', 0.0)
                                        st.markdown(f"{i}. *{course_title}* by {organization}")
                                        st.markdown(f"   📊 Relevance: {similarity:.3f} 🔄 JSON")
                                        
                                        # --- Phase 2: Personality-to-Pedagogy Logic ---
                                        user_style = getattr(st.session_state, 'learning_style', "Visual (Videos & Lectures)")
                                        title_lower = course_title.lower()
                                        pedagogy_match = True
                                        if "Practical" in user_style and not any(w in title_lower for w in ["project", "build", "lab", "hands-on", "workshop"]):
                                            pedagogy_match = False
                                        elif "Theoretical" in user_style and any(w in title_lower for w in ["project", "hands-on", "workshop", "applied"]):
                                            pedagogy_match = False
                                            
                                        ped_badge = "✅ Pedagogy Match" if pedagogy_match else "⚠️ Alternate Style"
                                        st.markdown(f"   🧠 **Learning Style**: {ped_badge} ({user_style.split(' ')[0]})")
                                        
                                        # SDG Badges (Semantic Match)
                                        course_emb = embedding_model.encode([course_title])
                                        c_tensor = torch.tensor(course_emb, dtype=torch.float32)
                                        sdg_embeddings = get_sdg_embeddings(embedding_model)
                                        sdg_tensor = torch.tensor(sdg_embeddings, dtype=torch.float32)
                                        
                                        sdg_sims = F.cosine_similarity(c_tensor, sdg_tensor)
                                        best_sdg_idx = torch.argmax(sdg_sims).item()
                                        best_sdg = SDG_DESCRIPTIONS[best_sdg_idx]
                                        
                                        st.markdown(f"   🌍 **SDG Alignment**: {best_sdg}")
                                        
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            with st.expander("✨ Why this course? (Ollama XAI)"):
                                                with st.spinner("Generating explanation..."):
                                                    # Retrieve course skills from ChromaDB for better context
                                                    course_skills = ""
                                                    try:
                                                        c_coll = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
                                                        c_res = c_coll.get(where={"course_title": course_title}, include=["metadatas"])
                                                        if c_res and c_res.get('metadatas') and len(c_res['metadatas']) > 0:
                                                            course_skills = c_res['metadatas'][0].get('skills', '')
                                                    except Exception:
                                                        pass
                                                        
                                                    explanation = generate_explanation_ollama(skill_gap, course_title, job_title, course_skills)
                                                    st.write(explanation)
                                        with col2:
                                            if st.button("🚫 Reject", key=f"reject_json_{course_title}"):
                                                st.session_state.rejected_courses.add(course_title)
                                                try:
                                                    st.rerun()
                                                except AttributeError:
                                                    st.experimental_rerun()
                                                    
                                        st.markdown("---")
                                        
                                    # --- Phase 2: Skill Roadmap Flowchart ---
                                    if len(sorted_recs) > 1:
                                        with st.expander("🗺️ View Suggested Learning Roadmap & Explanation", expanded=True):
                                            dot = "digraph Roadmap {\n"
                                            dot += "  rankdir=LR;\n"
                                            dot += "  node [shape=box, style=filled, fillcolor=lightblue, fontname=\"Helvetica\"];\n"
                                            nodes = []
                                            for idx, r in enumerate(sorted_recs):
                                                c_title = r.get('title', 'Course').replace('"', "'").replace("\n", " ")
                                                wrapped_title = '\\n'.join([c_title[i:i+30] for i in range(0, len(c_title), 30)])
                                                node_id = f"C{idx}"
                                                dot += f"  {node_id} [label=\"{wrapped_title}\"];\n"
                                                nodes.append(node_id)
                                            for idx in range(len(nodes)-1):
                                                dot += f"  {nodes[idx]} -> {nodes[idx+1]};\n"
                                            dot += "}\n"
                                            st.graphviz_chart(dot)
                                            
                                            st.markdown("### Why this Sequence?")
                                            with st.spinner("Generating roadmap rationale..."):
                                                course_titles = [r.get('title', 'Course') for r in sorted_recs]
                                                roadmap_explanation = generate_explanation_ollama(skill_gap, f"the sequence: {' -> '.join(course_titles)}", job_title)
                                                st.write(roadmap_explanation)
                                                
                                else:
                                    st.info("No specific course recommendations found for this role.")
                            else:
                                st.error("Could not load course recommendations.")
                        else:
                            # Fallback to skill-based recommendations
                            recommendations, error = find_course_recommendations(skill_gap, dssm_model, client, embedding_model, use_json_mapping=False)
                            
                            if error:
                                st.error(error)
                            elif recommendations:
                                st.success("✅ *Recommended Courses for Missing Skills:*")
                                for skill, recs in recommendations.items():
                                    st.subheader(f"Courses for: {skill}")
                                    
                                    # Filter rejected courses
                                    valid_recs = [r for r in recs if r.get('title', 'Unknown Course') not in st.session_state.rejected_courses]
                                    if not valid_recs:
                                         st.info("You've rejected all top recommendations for this skill.")
                                         continue
                                         
                                    for i, rec in enumerate(sorted_recs, 1):  # Show top 3 per skill
                                        source_badge = "🤖 DSSM"
                                        st.markdown(f"{i}. *{rec['title']}* by {rec['organization']}")
                                        st.markdown(f"   📊 Similarity: {rec['similarity']:.3f} {source_badge}")
                                        
                                        # --- Phase 2: Personality-to-Pedagogy Logic ---
                                        user_style = getattr(st.session_state, 'learning_style', "Visual (Videos & Lectures)")
                                        title_lower = rec['title'].lower()
                                        pedagogy_match = True
                                        if "Practical" in user_style and not any(w in title_lower for w in ["project", "build", "lab", "hands-on", "workshop"]):
                                            pedagogy_match = False
                                        elif "Theoretical" in user_style and any(w in title_lower for w in ["project", "hands-on", "workshop", "applied"]):
                                            pedagogy_match = False
                                            
                                        ped_badge = "✅ Pedagogy Match" if pedagogy_match else "⚠️ Alternate Style"
                                        st.markdown(f"   🧠 **Learning Style**: {ped_badge} ({user_style.split(' ')[0]})")
                                        
                                        if "sdg" in rec:
                                            st.markdown(f"   🌍 **SDG Match**: {rec['sdg']}")
                                            
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            with st.expander("✨ Why this course? (Ollama XAI)"):
                                                with st.spinner("Generating explanation..."):
                                                    # Retrieve course skills from ChromaDB for better context
                                                    c_title = rec['title']
                                                    course_skills = ""
                                                    try:
                                                        c_coll = client.get_collection(name=COURSE_EMBEDDINGS_COLLECTION)
                                                        c_res = c_coll.get(where={"course_title": c_title}, include=["metadatas"])
                                                        if c_res and c_res.get('metadatas') and len(c_res['metadatas']) > 0:
                                                            course_skills = c_res['metadatas'][0].get('skills', '')
                                                    except Exception:
                                                        pass
                                                        
                                                    explanation = generate_explanation_ollama([skill], c_title, target_job_meta.get('job_title', job_title_input), course_skills)
                                                    st.write(explanation)
                                        with col2:
                                            if st.button("🚫 Reject", key=f"reject_dssm_{skill}_{rec['title']}"):
                                                st.session_state.rejected_courses.add(rec['title'])
                                                if "embedding" in rec:
                                                    st.session_state.rejected_course_embeddings.append(rec["embedding"])
                                                try:
                                                    st.rerun()
                                                except AttributeError:
                                                    st.experimental_rerun()
                                        st.markdown("---")
                            else:
                                st.info("No specific course recommendations found for the identified skill gap.")
                            
                            # Fallback: Show general course recommendations for this job from JSON mapping
                            if use_json_mapping:
                                json_mapping = load_job_course_mapping()
                                if json_mapping:
                                    # Find this job in JSON mapping
                                    job_title = target_job_meta.get('job_title', target_job_meta.get('Job Title', ''))
                                    if not job_title and 'job_text' in target_job_meta:
                                        job_text = target_job_meta['job_text']
                                        if 'Job Title:' in job_text:
                                            job_title = job_text.split('Job Title:')[1].split('.')[0].strip()
                                    
                                    for job_entry in json_mapping:
                                        if job_entry.get('job_title', '').lower() == job_title.lower():
                                            top_courses = job_entry.get('top_courses', [])
                                            if top_courses:
                                                st.subheader("📚 General Course Recommendations for this Role:")
                                                for i, course_info in enumerate(top_courses[:3], 1):
                                                    course_title = course_info.get('title', 'Unknown Course')
                                                    organization = course_info.get('organization', 'Coursera')
                                                    if organization in ('Unknown Organization', 'Unknown', None, ''):
                                                        organization = 'Coursera'
                                                    similarity = course_info.get('similarity', 0.0)
                                                    st.markdown(f"{i}. *{course_title}* by {organization} (Similarity: {similarity:.3f})")
                                            break

                else:
                    st.warning("Could not determine the required skills for this job title.")
            else:
                st.error("Could not find a matching job in the database. Please try a different title.")
        else:
            st.error("Could not load necessary models or connect to the database. Please check the console for errors.")

    elif st.session_state.analysis_triggered:
        st.warning("Please upload your resume and enter a job title to begin.")

st.markdown("---")
st.markdown("Powered by DSSM and Streamlit")

