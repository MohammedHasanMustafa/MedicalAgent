# agents.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, List
import json

class MedicalAgents:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
    
    def query_analyzer_agent(self, state: Dict) -> Dict:
        """Analyze medical query and extract entities"""
        prompt = ChatPromptTemplate.from_template("""
        You are a medical query analyzer. Analyze the following medical query and extract all relevant medical entities, conditions, and search criteria.
        
        QUERY: {query}
        
        Extract the following information as JSON:
        {{
            "symptoms": ["list of symptoms mentioned"],
            "lab_tests": ["list of lab tests and criteria"],
            "imaging_studies": ["list of imaging studies and findings"],
            "conditions": ["list of medical conditions"],
            "demographics": ["age, gender, etc if mentioned"],
            "exclusions": ["exclusion criteria"],
            "data_types_needed": ["imaging", "clinical", "genomic", "pathology", "cardiology"],
            "search_terms": ["key terms for vector search"]
        }}
        
        Return only valid JSON.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        analysis = chain.invoke({"query": state["query"]})
        
        return {"query_analysis": analysis}
    
    def data_retrieval_agent(self, state: Dict) -> Dict:
        """Retrieve relevant data from all sources"""
        query_analysis = state.get("query_analysis", {})
        search_terms = query_analysis.get("search_terms", [])
        data_types_needed = query_analysis.get("data_types_needed", [])
        
        # Combine search terms
        search_query = " ".join(search_terms) if search_terms else state["query"]
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(search_query, k=20)
        
        # Organize results by data type
        results = {
            "patient_data": [],
            "imaging_data": [],
            "lab_results": [],
            "clinical_notes": [],
            "genomic_data": [],
            "pathology_data": [],
            "cardiology_data": [],
            "search_results": []
        }
        
        for doc in docs:
            data_type = doc.metadata.get("data_type", "unknown")
            result_entry = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "dataset": doc.metadata.get("dataset", "unknown"),
                "relevance_score": 0.8  # Placeholder
            }
            
            results["search_results"].append(result_entry)
            
            # Categorize by data type
            if data_type == "clinical":
                results["patient_data"].append(result_entry)
                results["clinical_notes"].append(result_entry)
            elif data_type == "imaging":
                results["imaging_data"].append(result_entry)
            elif data_type == "genomic":
                results["genomic_data"].append(result_entry)
            elif data_type == "pathology":
                results["pathology_data"].append(result_entry)
            elif data_type == "cardiology":
                results["cardiology_data"].append(result_entry)
        
        return results
    
    def clinical_analysis_agent(self, state: Dict) -> Dict:
        """Analyze clinical data and patient information"""
        patient_data = state.get("patient_data", [])
        clinical_notes = state.get("clinical_notes", [])
        
        if not patient_data and not clinical_notes:
            return {"structured_data": {"clinical_analysis": "No clinical data found"}}
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following clinical data and extract structured patient information:
        
        PATIENT DATA:
        {patient_data}
        
        CLINICAL NOTES:
        {clinical_notes}
        
        Extract:
        - Patient demographics
        - Medical conditions
        - Symptoms
        - Treatments
        - Lab results
        - Risk factors
        
        Return as structured JSON.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        analysis = chain.invoke({
            "patient_data": json.dumps(patient_data, indent=2),
            "clinical_notes": json.dumps(clinical_notes, indent=2)
        })
        
        return {"structured_data": {"clinical_analysis": analysis}}
    
    def imaging_analysis_agent(self, state: Dict) -> Dict:
        """Analyze imaging data"""
        imaging_data = state.get("imaging_data", [])
        
        if not imaging_data:
            return {"structured_data": {"imaging_analysis": "No imaging data found"}}
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following medical imaging data:
        
        IMAGING DATA:
        {imaging_data}
        
        Extract:
        - Imaging modalities
        - Findings
        - Abnormalities
        - Correlations with clinical data
        
        Return as structured JSON.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        analysis = chain.invoke({"imaging_data": json.dumps(imaging_data, indent=2)})
        
        return {"structured_data": {"imaging_analysis": analysis}}
    
    def lab_analysis_agent(self, state: Dict) -> Dict:
        """Analyze lab results"""
        lab_results = state.get("lab_results", [])
        
        if not lab_results:
            return {"structured_data": {"lab_analysis": "No lab data found"}}
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following laboratory results:
        
        LAB DATA:
        {lab_data}
        
        Extract:
        - Test types
        - Abnormal values
        - Trends
        - Clinical significance
        
        Return as structured JSON.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        analysis = chain.invoke({"lab_data": json.dumps(lab_results, indent=2)})
        
        return {"structured_data": {"lab_analysis": analysis}}
    
    def data_integrator_agent(self, state: Dict) -> Dict:
        """Integrate all data and generate final response"""
        all_data = {
            "query_analysis": state.get("query_analysis", {}),
            "patient_data": state.get("patient_data", []),
            "imaging_data": state.get("imaging_data", []),
            "lab_results": state.get("lab_results", []),
            "clinical_notes": state.get("clinical_notes", []),
            "genomic_data": state.get("genomic_data", []),
            "pathology_data": state.get("pathology_data", []),
            "cardiology_data": state.get("cardiology_data", []),
            "structured_data": state.get("structured_data", {}),
            "search_results": state.get("search_results", [])
        }
        
        prompt = ChatPromptTemplate.from_template("""
        You are a medical data analyst. Based on the integrated medical data from multiple sources, provide a comprehensive response to the original query.
        
        ORIGINAL QUERY: {query}
        
        INTEGRATED DATA ANALYSIS:
        {all_data}
        
        Provide a comprehensive response including:
        1. Summary of relevant findings across all data types
        2. Patient cases matching the criteria (if any)
        3. Patterns and insights discovered
        4. Clinical correlations
        5. Recommendations for further investigation
        
        Format the response in a clear, clinical manner suitable for healthcare professionals.
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": state["query"],
            "all_data": json.dumps(all_data, indent=2)
        })
        
        return {"final_response": response.content}
