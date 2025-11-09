"""
Core workflow and state definitions for the CV Generator.
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List


class CVGeneratorState(TypedDict):
    """
    State definition for the CV generation workflow.
    
    This state holds all the information needed throughout the CV generation process:
    - job_data: The input job posting information
    - extracted_requirements: Key requirements extracted from the job posting
    - cv_sections: Different sections of the generated CV
    - final_cv: The complete tailored CV
    - messages: Conversation history for the LLM
    """
    
    # Input data
    job_data: dict  # Single row from the job posting dataset
    
    # Processing steps
    extracted_requirements: dict  # Key skills, qualifications, and requirements
    cv_sections: dict  # Different sections of the CV (experience, skills, etc.)
    
    # Output
    final_cv: str  # The complete generated CV

    # LangGraph message handling
    messages: Annotated[List, add_messages]

    # Controls
    description_only: bool  # If True, only the description is used for generation
    abort_generation: bool  # Set True to short-circuit and output NA


def create_cv_generator_graph():
    """
    Creates the LangGraph workflow for CV generation.
    
    The workflow consists of several nodes:
    1. Extract Requirements - Analyze job posting for key requirements
    2. Generate Experience Section - Create relevant work experience
    3. Generate Skills Section - Highlight matching skills
    4. Generate Education Section - Add relevant education
    5. Compile Final CV - Combine all sections into final CV
    
    Returns:
        StateGraph: The configured LangGraph workflow
    """
    
    # Import here to avoid circular imports
    from .nodes import (
        check_description_specificity,
        extract_job_requirements,
        generate_experience_section,
        generate_skills_section,
        generate_education_section,
        compile_final_cv
    )
    
    # Initialize the state graph
    workflow = StateGraph(CVGeneratorState)
    
    # Add nodes to the workflow
    # Validation now happens AFTER requirements extraction
    workflow.add_node("extract_requirements", extract_job_requirements)
    workflow.add_node("check_description", check_description_specificity)
    workflow.add_node("generate_experience", generate_experience_section)
    workflow.add_node("generate_skills", generate_skills_section)
    workflow.add_node("generate_education", generate_education_section)
    workflow.add_node("compile_cv", compile_final_cv)
    
    # Define the workflow edges (execution order)
    workflow.set_entry_point("extract_requirements")
    
    workflow.add_edge("extract_requirements", "check_description")
    workflow.add_edge("check_description", "generate_experience")
    workflow.add_edge("generate_experience", "generate_skills")
    workflow.add_edge("generate_skills", "generate_education")
    workflow.add_edge("generate_education", "compile_cv")
    workflow.add_edge("compile_cv", END)
    
    return workflow.compile()
