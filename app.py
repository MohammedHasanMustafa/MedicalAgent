# app.py
from config import MedicalConfig
from data_loader import ComprehensiveMedicalDataLoader
from vector_store import VectorStoreManager
from agents import MedicalAgents
from workflow import MedicalWorkflow

def initialize_system():
    """Initialize the complete medical data exploration system"""
    config = MedicalConfig()
    
    # Initialize Azure components
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config.EMBEDDING_MODEL,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY
    )
    
    llm = AzureChatOpenAI(
        azure_deployment=config.LLM_MODEL,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0.1
    )
    
    # Load data
    print("Loading medical datasets...")
    data_loader = ComprehensiveMedicalDataLoader(config)
    all_documents = data_loader.load_all_datasets()
    
    # Create vector store
    print("Creating vector store...")
    vector_manager = VectorStoreManager(embeddings, config)
    
    # Check if vector store exists, otherwise create it
    try:
        vector_store = vector_manager.load_vector_store()
        print("✓ Loaded existing vector store")
    except:
        print("Creating new vector store...")
        vector_store = vector_manager.create_vector_store(all_documents)
    
    # Initialize agents and workflow
    agents = MedicalAgents(llm, vector_store)
    workflow = MedicalWorkflow(agents)
    
    return workflow, config

def main():
    """Main function to run the medical data exploration system"""
    
    # Example medical queries
    example_queries = [
        "Find patients with chest pain and abnormal cardiac findings",
        "Show me cases of pneumonia with imaging confirmation",
        "Patients with elevated troponin levels and cardiac history",
        "Find brain tumor cases with MRI imaging",
        "Cases of diabetic retinopathy with retinal images",
        "Patients with genetic mutations and cancer history"
    ]
    
    print("Medical Data Exploration System")
    print("=" * 50)
    
    # Initialize system
    workflow, config = initialize_system()
    
    while True:
        print("\nEnter your medical query (or 'quit' to exit):")
        query = input("> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query.strip():
            continue
        
        print("\nProcessing query...")
        try:
            result = workflow.run(query)
            print("\n" + "="*80)
            print("RESULT:")
            print("="*80)
            print(result)
            print("="*80)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
