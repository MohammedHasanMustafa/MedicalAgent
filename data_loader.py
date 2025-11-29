# data_loader.py
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import pydicom
import numpy as np
from tqdm import tqdm
import re

class ComprehensiveMedicalDataLoader:
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_all_datasets(self) -> Dict[str, List[Document]]:
        """Load all available medical datasets"""
        print("Loading ALL medical datasets from Awesome-Medical-Dataset...")
        
        all_documents = {}
        
        for dataset_name, config in self.config.DATASET_CONFIGS.items():
            dataset_path = self.config.DATA_BASE_PATH / config["path"]
            
            if dataset_path.exists():
                try:
                    documents = self._load_dataset(dataset_name, dataset_path, config)
                    all_documents[dataset_name] = documents
                    print(f"✓ Loaded {dataset_name}: {len(documents)} documents")
                except Exception as e:
                    print(f"✗ Failed to load {dataset_name}: {e}")
            else:
                print(f"⚠ Dataset path not found: {dataset_path}")
        
        return all_documents
    
    def _load_dataset(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load specific dataset based on type"""
        data_type = config["data_type"]
        
        if data_type == "imaging":
            return self._load_imaging_data(dataset_name, dataset_path, config)
        elif data_type == "clinical":
            return self._load_clinical_data(dataset_name, dataset_path, config)
        elif data_type == "genomic":
            return self._load_genomic_data(dataset_name, dataset_path, config)
        elif data_type == "pathology":
            return self._load_pathology_data(dataset_name, dataset_path, config)
        elif data_type == "cardiology":
            return self._load_cardiology_data(dataset_name, dataset_path, config)
        elif data_type == "ophthalmology":
            return self._load_ophthalmology_data(dataset_name, dataset_path, config)
        else:
            return self._load_generic_dataset(dataset_name, dataset_path, config)
    
    def _load_imaging_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load imaging datasets"""
        documents = []
        
        # Try to find CSV files with metadata
        csv_files = list(dataset_path.rglob("*.csv"))
        json_files = list(dataset_path.rglob("*.json"))
        text_files = list(dataset_path.rglob("*.txt"))
        
        # Process CSV files
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for idx, row in df.iterrows():
                    content = self._create_imaging_content(row, dataset_name, config)
                    metadata = {
                        "dataset": dataset_name,
                        "data_type": "imaging",
                        "modality": config["modality"],
                        "body_part": config["body_part"],
                        "source_file": str(csv_file),
                        "row_index": idx
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        # Process JSON files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                content = f"Imaging Data from {dataset_name}: {json.dumps(data, indent=2)}"
                metadata = {
                    "dataset": dataset_name,
                    "data_type": "imaging",
                    "modality": config["modality"],
                    "body_part": config["body_part"],
                    "source_file": str(json_file)
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return documents
    
    def _load_clinical_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load clinical/EHR datasets"""
        documents = []
        
        # Look for clinical data files
        csv_files = list(dataset_path.rglob("*.csv"))
        json_files = list(dataset_path.rglob("*.json"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for idx, row in df.iterrows():
                    content = self._create_clinical_content(row, dataset_name)
                    metadata = {
                        "dataset": dataset_name,
                        "data_type": "clinical",
                        "modality": config["modality"],
                        "body_part": config["body_part"],
                        "source_file": str(csv_file),
                        "row_index": idx
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return documents
    
    def _load_genomic_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load genomic datasets"""
        documents = []
        
        # Look for genomic data files
        csv_files = list(dataset_path.rglob("*.csv"))
        tsv_files = list(dataset_path.rglob("*.tsv"))
        
        for file_path in csv_files + tsv_files:
            try:
                df = pd.read_csv(file_path, sep='\t' if file_path.suffix == '.tsv' else ',')
                for idx, row in df.iterrows():
                    content = self._create_genomic_content(row, dataset_name)
                    metadata = {
                        "dataset": dataset_name,
                        "data_type": "genomic",
                        "modality": config["modality"],
                        "body_part": config["body_part"],
                        "source_file": str(file_path),
                        "row_index": idx
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def _load_pathology_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load pathology datasets"""
        documents = []
        
        csv_files = list(dataset_path.rglob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for idx, row in df.iterrows():
                    content = self._create_pathology_content(row, dataset_name)
                    metadata = {
                        "dataset": dataset_name,
                        "data_type": "pathology",
                        "modality": config["modality"],
                        "body_part": config["body_part"],
                        "source_file": str(csv_file),
                        "row_index": idx
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return documents
    
    def _load_cardiology_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load cardiology datasets"""
        return self._load_imaging_data(dataset_name, dataset_path, config)  # Similar structure
    
    def _load_ophthalmology_data(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load ophthalmology datasets"""
        return self._load_imaging_data(dataset_name, dataset_path, config)  # Similar structure
    
    def _create_imaging_content(self, row: pd.Series, dataset_name: str, config: Dict) -> str:
        """Create content for imaging data"""
        content_parts = [f"Dataset: {dataset_name}", f"Modality: {config['modality']}", f"Body Part: {config['body_part']}"]
        
        for col in row.index:
            if pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        return "\n".join(content_parts)
    
    def _create_clinical_content(self, row: pd.Series, dataset_name: str) -> str:
        """Create content for clinical data"""
        content_parts = [f"Clinical Data from {dataset_name}"]
        
        # Prioritize important clinical fields
        priority_fields = ['patient_id', 'age', 'gender', 'diagnosis', 'symptoms', 
                          'lab_results', 'medications', 'treatment', 'outcome']
        
        for field in priority_fields:
            if field in row and pd.notna(row[field]):
                content_parts.append(f"{field}: {row[field]}")
        
        # Add remaining fields
        for col in row.index:
            if col not in priority_fields and pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        return "\n".join(content_parts)
    
    def _create_genomic_content(self, row: pd.Series, dataset_name: str) -> str:
        """Create content for genomic data"""
        content_parts = [f"Genomic Data from {dataset_name}"]
        
        # Genomic-specific fields
        genomic_fields = ['gene', 'mutation', 'expression', 'variant', 'chromosome', 
                         'position', 'sample_id', 'cancer_type']
        
        for field in genomic_fields:
            if field in row and pd.notna(row[field]):
                content_parts.append(f"{field}: {row[field]}")
        
        for col in row.index:
            if col not in genomic_fields and pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        return "\n".join(content_parts)
    
    def _create_pathology_content(self, row: pd.Series, dataset_name: str) -> str:
        """Create content for pathology data"""
        content_parts = [f"Pathology Data from {dataset_name}"]
        
        pathology_fields = ['slide_id', 'tissue_type', 'diagnosis', 'malignancy', 
                           'grade', 'stage', 'patient_id']
        
        for field in pathology_fields:
            if field in row and pd.notna(row[field]):
                content_parts.append(f"{field}: {row[field]}")
        
        for col in row.index:
            if col not in pathology_fields and pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        return "\n".join(content_parts)
    
    def _load_generic_dataset(self, dataset_name: str, dataset_path: Path, config: Dict) -> List[Document]:
        """Load any dataset with generic approach"""
        documents = []
        
        # Try all common file types
        csv_files = list(dataset_path.rglob("*.csv"))
        json_files = list(dataset_path.rglob("*.json"))
        text_files = list(dataset_path.rglob("*.txt"))
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                for idx, row in df.iterrows():
                    content = f"Data from {dataset_name}:\n" + "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    metadata = {
                        "dataset": dataset_name,
                        "data_type": config["data_type"],
                        "modality": config["modality"],
                        "body_part": config["body_part"],
                        "source_file": str(file_path),
                        "row_index": idx
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return documents
