"""
Main CV Generator module.

This module provides the main interface for generating tailored CVs using LangGraph.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from .workflow import CVGeneratorState, create_cv_generator_graph

# Load environment variables
load_dotenv()
default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))


def generate_cv_for_job(job_row: dict, api_key: str = None) -> tuple[str, dict]:
    """
    Generate a tailored CV for a specific job posting.
    
    Args:
        job_row: A single row from the job posting dataset
        api_key: OpenAI API key (if not set in environment)
    
    Returns:
        tuple: (generated_cv, extracted_job_info) where:
            - generated_cv: The generated tailored CV
            - extracted_job_info: Dictionary containing extracted job requirements and original job data
    """
    
    # Initialize the LLM
    if api_key:
        llm = ChatOpenAI(api_key=api_key, model=default_model, temperature=default_temperature)
    else:
        llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    # Create the workflow
    workflow = create_cv_generator_graph()
    
    # Initialize the state
    initial_state = CVGeneratorState(
        job_data=job_row,
        extracted_requirements={},
        cv_sections={},
        final_cv="",
        messages=[]
    )
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Prepare extracted job information
    extracted_job_info = {
        "original_job_data": job_row,
        "extracted_requirements": result.get("extracted_requirements", {}),
        "cv_sections": result.get("cv_sections", {})
    }
    
    return result["final_cv"], extracted_job_info


# Example usage
if __name__ == "__main__":
    sample_job = {
        "title_raw": "Software Engineer",
        "description": "Looking for a software engineer with Python experience...",
        "job_category": "Technology",
        "role_k50": "Software Developer",
        "company": "Tech Corp",
        "location_raw": "Toronto, ON"
    }
    
    # Generate CV (you'll need to provide your OpenAI API key)
    # cv = generate_cv_for_job(sample_job, "your-api-key-here")
    # print(cv)