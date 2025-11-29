# workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import MedicalState

class MedicalWorkflow:
    def __init__(self, agents):
        self.agents = agents
        self.workflow = StateGraph(MedicalState)
        
        # Add all nodes
        self.workflow.add_node("query_analyzer", self.agents.query_analyzer_agent)
        self.workflow.add_node("data_retrieval", self.agents.data_retrieval_agent)
        self.workflow.add_node("clinical_analysis", self.agents.clinical_analysis_agent)
        self.workflow.add_node("imaging_analysis", self.agents.imaging_analysis_agent)
        self.workflow.add_node("lab_analysis", self.agents.lab_analysis_agent)
        self.workflow.add_node("data_integrator", self.agents.data_integrator_agent)
        
        # Define workflow
        self.workflow.set_entry_point("query_analyzer")
        
        # Connect nodes
        self.workflow.add_edge("query_analyzer", "data_retrieval")
        self.workflow.add_edge("data_retrieval", "clinical_analysis")
        self.workflow.add_edge("data_retrieval", "imaging_analysis")
        self.workflow.add_edge("data_retrieval", "lab_analysis")
        self.workflow.add_edge("clinical_analysis", "data_integrator")
        self.workflow.add_edge("imaging_analysis", "data_integrator")
        self.workflow.add_edge("lab_analysis", "data_integrator")
        self.workflow.add_edge("data_integrator", END)
        
        # Compile with memory
        memory = SqliteSaver.from_conn_string(":memory:")
        self.app = self.workflow.compile(checkpointer=memory)
    
    def run(self, query: str) -> str:
        """Execute the medical data exploration workflow"""
        config = {"configurable": {"thread_id": "medical_thread_1"}}
        
        result = self.app.invoke(
            {"query": query},
            config=config
        )
        
        return result["final_response"]
