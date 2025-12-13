# AI Powered Skill Gap Analysis Reskilling For Employment Trends

During my internship at the Centre of Cognitive Computing and Computational Intelligence (C3I) in the Summer of 2025, I designed and deployed an end-to-end career advisory system using dual-tower Deep Structured Semantic Models and ChromaDB achieving highly accurate job–course matching.

![poster](https://github.com/user-attachments/assets/6f8fb9ab-c453-4db7-9544-944519869cd7)

### Key Features 

- Utilized all-MiniLM-L6-v2 to generate and push refined job and course embeddings into ChromaDB for efficient semantic search.

- Employed a Deep Structured Semantic Model (DSSM) for training to learn enhanced semantic relationships.

- Developed a Streamlit web application as a user-friendly frontend interface, facilitating interactive skill gap analysis and course recommendations.

- Provided intelligent course suggestions directly addressing identified skill gaps relevant to a specific job position, leveraging both pre-computed mappings and the trained DSSM.

### Objective

- Develop a model that extracts skills from user resumes and identifies gaps between current capabilities and target job requirements.

- Create a deep learning-based recommendation system using DSSM that suggests relevant courses based on identified skill gaps and job requirements.

- Build a robust system that can extract skills from various resume formats (PDF, DOCX, images) using OCR and NLP techniques, filtering out irrelevant content.

- Develop an interactive Streamlit application that provides instant career guidance, skill analysis, and course recommendations with a user-friendly interface.



### Implementation

- `eda_analysis`: This script analyzes raw job and course data, identifying missing values and key characteristics. It performs cross-dataset skill analysis to find common skills and highlight gaps between job requirements and course offerings.

- `data_processing`: This script cleans and standardizes raw data, handling missing values and normalizing skill names to a canonical form. It also creates a combined text field for each job and course, which is essential for later embedding.
  
- `populate_chromadb`: Generates This script converts the preprocessed text into vector embeddings using `all-MiniLM-L6-v2` and populates a ChromaDB vector database with these embeddings. It also performs an initial job-to-course similarity mapping, saving the results to a JSON file.

- `model_training`: This script trains a `Deep Structured Semantic Model (DSSM)` to refine job-course similarity. It uses embeddings from ChromaDB and generates a dataset of positive and negative pairs for training. The training process incorporates an `Exponential Moving Average (EMA)` and early stopping to save the best-performing model.

- `model_testing`: This script is a Streamlit web application that acts as the user interface. It analyzes a user's resume, identifies skill gaps for a desired job, and recommends relevant courses by leveraging either the pre-computed mappings or the trained DSSM model for deeper semantic matching.

- `utils`: The utils.py script serves as a central toolkit for the project, containing reusable helper functions that standardize and preprocess text and skills. It sets up `Natural Language Processing (NLP)` components from nltk and spacy for tasks like lemmatization and stop word removal. The script's core functionality revolves around a large `SKILL_NORMALIZATION_MAP` that maps common skill variations to a single, canonical form, ensuring consistency across all job and course data. This script is used by both data_processing.py and testing.py to clean and normalize text, extract skills, and perform semantic comparisons, ensuring that data is consistently formatted before it's used for model training or user interaction. 

### Architecture Diagram
![WhatsApp Image 2025-08-25 at 17 08 29_c9deb4c6](https://github.com/user-attachments/assets/17cf07cb-a8ee-498a-a080-66e6e984af04)

 ### Model Training Results 

<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/66b9e7ca-1061-4538-a946-2f74e3366e01" />

<img width="1869" height="531" alt="image" src="https://github.com/user-attachments/assets/0c66d0af-7d88-4256-b7d0-fdb47fe0691a" />

<img width="751" height="249" alt="image" src="https://github.com/user-attachments/assets/25ef7862-cebd-44f3-8821-925d897031a6" />


### Dataset Info
<img width="867" height="290" alt="image" src="https://github.com/user-attachments/assets/36217c5e-c20a-4422-b30e-78a515bf21f0" />



### Streamlit Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0cfb4da4-e838-4a93-9880-327f3215675d" />

### Skill Extraction From Resume Using Tesseract OCR
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b0bc39df-bfda-40b6-be08-b494695c73cf" />

### Course Recommendation in case of Skill Gap
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4edd745c-ae41-4941-be49-0d2405dc180e" />

### Invalid Document Upload
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/18b42bb9-178e-433d-876b-c2709a4e3ac8" />

### Irrelevant Job Post
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d7ccb700-1720-4773-8b7b-67d042d235f8" />









