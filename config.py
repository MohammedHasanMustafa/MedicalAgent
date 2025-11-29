# config.py
import os
from pathlib import Path
from typing import Dict, List, Any

class MedicalConfig:
    # Azure Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = "2024-02-01"
    
    # Models
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4"
    
    # Vector Store
    VECTOR_STORE_TYPE = "faiss"
    VECTOR_STORE_PATH = "medical_vector_store"
    
    # Data paths
    DATA_BASE_PATH = Path("Awesome-Medical-Dataset")
    
    # Comprehensive dataset configurations covering ALL medical data types
    DATASET_CONFIGS = {
        # Imaging Data
        "chest_xray14": {
            "path": "ChestX-ray14",
            "data_type": "imaging",
            "modality": "X-ray",
            "body_part": "Chest",
            "contains": ["chest diseases", "xray images", "clinical findings"]
        },
        "mimic_cxr": {
            "path": "MIMIC-CXR",
            "data_type": "imaging",
            "modality": "X-ray", 
            "body_part": "Chest",
            "contains": ["chest xrays", "radiology reports", "patient data"]
        },
        "covid_chestxray": {
            "path": "COVID-19_Radiography_Dataset",
            "data_type": "imaging",
            "modality": "X-ray",
            "body_part": "Chest",
            "contains": ["covid19 cases", "chest imaging", "pneumonia"]
        },
        "rsna_pneumonia": {
            "path": "rsna-pneumonia-detection-challenge",
            "data_type": "imaging",
            "modality": "X-ray",
            "body_part": "Chest",
            "contains": ["pneumonia detection", "chest scans"]
        },
        "brats": {
            "path": "BraTS",
            "data_type": "imaging", 
            "modality": "MRI",
            "body_part": "Brain",
            "contains": ["brain tumors", "mri scans", "segmentation"]
        },
        "isic": {
            "path": "ISIC",
            "data_type": "imaging",
            "modality": "Dermatoscopy",
            "body_part": "Skin",
            "contains": ["skin lesions", "melanoma", "dermatology images"]
        },
        
        # Clinical & EHR Data
        "mimic_iv": {
            "path": "MIMIC-IV",
            "data_type": "clinical",
            "modality": "EHR",
            "body_part": "Multi-system",
            "contains": ["electronic health records", "icu data", "vital signs", "lab results"]
        },
        "eicu": {
            "path": "eICU",
            "data_type": "clinical", 
            "modality": "EHR",
            "body_part": "Multi-system",
            "contains": ["icu patients", "vital signs", "treatment data"]
        },
        
        # Genomic Data
        "tcga": {
            "path": "TCGA",
            "data_type": "genomic",
            "modality": "Genomics",
            "body_part": "Multi-system", 
            "contains": ["cancer genomics", "dna sequencing", "tumor data"]
        },
        
        # Pathology Data
        "camelyon": {
            "path": "Camelyon",
            "data_type": "pathology",
            "modality": "Histopathology",
            "body_part": "Lymph nodes",
            "contains": ["whole slide images", "cancer detection", "pathology"]
        },
        
        # Cardiology Data
        "echonet": {
            "path": "EchoNet",
            "data_type": "cardiology", 
            "modality": "Echocardiogram",
            "body_part": "Heart",
            "contains": ["echo videos", "heart function", "cardiology"]
        },
        
        # Ophthalmology Data
        "kaggle_diabetic_retinopathy": {
            "path": "Kaggle-Diabetic-Retinopathy",
            "data_type": "ophthalmology",
            "modality": "Retinal imaging",
            "body_part": "Eyes",
            "contains": ["retinal images", "diabetic retinopathy", "eye disease"]
        }
  }
