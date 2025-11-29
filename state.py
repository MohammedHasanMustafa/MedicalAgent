# state.py
from typing import TypedDict, List, Dict, Annotated
import operator

class MedicalState(TypedDict):
    query: str
    query_analysis: Dict
    patient_data: Annotated[List[Dict], operator.add]
    imaging_data: Annotated[List[Dict], operator.add]
    lab_results: Annotated[List[Dict], operator.add]
    clinical_notes: Annotated[List[Dict], operator.add]
    genomic_data: Annotated[List[Dict], operator.add]
    pathology_data: Annotated[List[Dict], operator.add]
    cardiology_data: Annotated[List[Dict], operator.add]
    structured_data: Annotated[Dict, operator.add]
    final_response: str
    intermediate_results: Annotated[List[Dict], operator.add]
    search_results: Annotated[List[Dict], operator.add]
