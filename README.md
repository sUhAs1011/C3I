# 🎯 AI-Powered Skill Gap Analysis & Reskilling for Employment Trends

An AI-driven career recommendation platform that analyzes a candidate's resume, identifies skill gaps for a target job role, and recommends personalized learning pathways using semantic search and deep learning.

Developed during my **Summer 2025 Internship** at the **Centre of Cognitive Computing and Computational Intelligence (C3I)**, this project combines **Natural Language Processing (NLP)**, **Deep Structured Semantic Models (DSSM)**, **Vector Databases**, and **Resume Parsing** to build an end-to-end intelligent career advisory system.

---

# 📌 Project Overview

Traditional job recommendation systems rely on keyword matching, often missing semantic relationships between skills, courses, and job descriptions.

This project addresses that limitation by combining:

- Semantic Embeddings
- Deep Structured Semantic Models (DSSM)
- ChromaDB Vector Search
- OCR-based Resume Parsing
- NLP Skill Extraction

to provide personalized career guidance and intelligent course recommendations.

---

# 🚀 Features

- 📄 Resume parsing from PDF, DOCX, and images
- 🔍 OCR-powered text extraction using Tesseract
- 🧠 NLP-based skill extraction and normalization
- 🎯 AI-driven skill gap identification
- 📚 Personalized course recommendations
- 🔎 Semantic search using ChromaDB
- 🤖 DSSM-based job-course matching
- 🌐 Interactive Streamlit web application
- 🛣️ Learning roadmap generation

---

# 🏗️ System Architecture

The system follows a multi-stage AI pipeline that transforms raw resumes into personalized career recommendations.

```text
Resume
   │
   ▼
OCR & Text Extraction
(Tesseract)
   │
   ▼
NLP Skill Extraction
   │
   ▼
Skill Normalization
   │
   ▼
Job Embeddings
Course Embeddings
(all-MiniLM-L6-v2)
   │
   ▼
ChromaDB
(Vector Database)
   │
   ▼
Deep Structured
Semantic Model (DSSM)
   │
   ▼
Skill Gap Analysis
   │
   ▼
Course Recommendation
   │
   ▼
Learning Roadmap
```

---

# 🧠 AI Pipeline

## 📄 Resume Processing

Supports multiple resume formats:

- PDF
- DOCX
- Images

Text is extracted using **Tesseract OCR** before NLP preprocessing.

---

## 📝 Skill Extraction

Natural Language Processing techniques are used to:

- Tokenize text
- Remove stop words
- Lemmatize skills
- Normalize skill variations

A custom **Skill Normalization Map** ensures consistent representation across resumes, jobs, and courses.

---

## 🔍 Semantic Embedding Generation

The project uses

**all-MiniLM-L6-v2**

to generate dense embeddings for

- Job descriptions
- Courses
- Skills

These embeddings are stored inside **ChromaDB** for efficient semantic retrieval.

---

## 🤖 Deep Structured Semantic Model (DSSM)

A Dual-Tower DSSM is trained to learn semantic relationships between

- Jobs
- Required skills
- Courses

Training includes

- Positive & negative pair generation
- Early stopping
- Exponential Moving Average (EMA)

allowing the model to outperform traditional cosine similarity.

---

## 📚 Intelligent Recommendation Engine

Once a target job is selected, the system

- Extracts missing skills
- Performs semantic retrieval
- Ranks courses using DSSM
- Generates personalized learning recommendations

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **Frontend** | Streamlit |
| **Deep Learning** | DSSM (Dual Tower Network) |
| **Embeddings** | all-MiniLM-L6-v2 |
| **Vector Database** | ChromaDB |
| **NLP** | SpaCy, NLTK |
| **OCR** | Tesseract OCR |
| **Machine Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy |

---

# 📂 Project Modules

## 📊 Exploratory Data Analysis

- Dataset exploration
- Missing value analysis
- Skill distribution
- Cross-dataset comparison

---

## ⚙️ Data Processing

- Cleaning
- Skill normalization
- Canonical mapping
- Text preprocessing

---

## 🗄️ ChromaDB Population

- Generate embeddings
- Store vectors
- Initial job-course mapping

---

## 🏋️ Model Training

- DSSM Training
- Positive/Negative Sampling
- EMA
- Early Stopping

---

## 🧪 Model Testing

Interactive Streamlit application that

- Parses resumes
- Detects skill gaps
- Recommends courses
- Generates learning roadmap

---

## 🔧 Utility Module

Contains reusable NLP utilities including

- Text preprocessing
- Skill extraction
- Lemmatization
- Skill normalization
- Semantic comparison

---

# 📊 Architecture Diagram

<p align="center">
<img src="https://github.com/user-attachments/assets/17cf07cb-a8ee-498a-a080-66e6e984af04"/>
</p>

---

# 📈 Model Performance

The DSSM training demonstrated effective convergence while learning meaningful semantic representations between job descriptions and educational resources.

### Training Curve

<p align="center">
<img src="https://github.com/user-attachments/assets/66b9e7ca-1061-4538-a946-2f74e3366e01"/>
</p>

### Evaluation Results

<p align="center">
<img src="https://github.com/user-attachments/assets/0c66d0af-7d88-4256-b7d0-fdb47fe0691a"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/25ef7862-cebd-44f3-8821-925d897031a6"/>
</p>

---

# 📊 Dataset

The project utilizes datasets containing

- Job descriptions
- Required skills
- Online courses
- Skill mappings

<p align="center">
<img src="https://github.com/user-attachments/assets/36217c5e-c20a-4422-b30e-78a515bf21f0"/>
</p>

---

# 🖥️ Application Demo

## Streamlit Interface

<img src="https://github.com/user-attachments/assets/3ba1dae3-8721-4084-8633-887f686c6757"/>

---

## Resume Skill Extraction

<img src="https://github.com/user-attachments/assets/07bce9d4-d0ae-43aa-be34-c0d21163901c"/>

---

## Skill Gap Recommendation

<img src="https://github.com/user-attachments/assets/57a7561e-f621-4240-9c66-c5ad77593b5e"/>

---

## Invalid Resume Detection

<img src="https://github.com/user-attachments/assets/e1615a82-22af-4af1-9704-f509d6cbea99"/>

---

## Invalid Job Detection

<img src="https://github.com/user-attachments/assets/0936bb4d-1d9b-4ebe-9838-29eca46d1c86"/>

---

## Personalized Learning Roadmap

<img src="https://github.com/user-attachments/assets/28645af4-cf32-4716-af86-ef3313e99b7d"/>

---

# 🚀 Future Enhancements

- Retrieval-Augmented Generation (RAG)
- Multi-language resume support
- LinkedIn profile analysis
- Live job portal integration
- LLM-powered career coaching
- Salary prediction
- Personalized interview preparation
- Recruiter dashboard

---

# Certificate
<img width="1448" height="1086" alt="ChatGPT Image Jul 16, 2026, 06_56_40 PM" src="https://github.com/user-attachments/assets/40543f3b-fb9d-4729-aab6-45743d2452eb" />

