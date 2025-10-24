"""
Node functions for the CV generation workflow.

Each function represents a step in the LangGraph workflow for generating tailored CVs.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
default_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))


def extract_job_requirements(state):
    """
    Extract key requirements from the job posting.
    
    This node analyzes the job posting data to identify:
    - Required skills and technologies
    - Experience level needed
    - Education requirements
    - Key responsibilities
    - Company culture indicators
    """
    
    # Initialize LLM
    llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    # Create prompt for requirement extraction
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert job analyst. Extract key requirements from job postings.
        
        Return a JSON object with these fields:
        - required_skills: List of technical skills mentioned
        - soft_skills: List of soft skills mentioned
        - experience_level: Entry/Mid/Senior level
        - education_requirements: Degree requirements
        - key_responsibilities: Main job duties
        - industry_keywords: Industry-specific terms
        - company_culture: Work environment indicators
        """),
        ("human", """Analyze this job posting and extract requirements:

Job Title: {title}
Company: {company}
Description: {description}
Job Category: {job_category}
Role: {role}

Extract the key requirements in JSON format.""")
    ])
    
    # Get job data
    job_data = state["job_data"]
    
    # Format the prompt
    formatted_prompt = extraction_prompt.format_messages(
        title=job_data.get("title_raw", ""),
        company=job_data.get("company", ""),
        description=job_data.get("description", ""),
        job_category=job_data.get("job_category", ""),
        role=job_data.get("role_k50", "")
    )
    
    # Get AI response
    response = llm.invoke(formatted_prompt)
    
    # Parse the response (assuming it returns JSON)
    try:
        requirements = json.loads(response.content)
    except:
        # Fallback if JSON parsing fails - include only job information
        requirements = {
            "job_title": job_data.get("title_raw", ""),
            "company": job_data.get("company", ""),
            "description": job_data.get("description", ""),
            "job_category": job_data.get("job_category", ""),
            "role": job_data.get("role_k50", "")
        }
    
    # Update state
    state["extracted_requirements"] = requirements
    state["messages"].append(AIMessage(content=f"Extracted requirements: {requirements}"))
    
    return state


def generate_experience_section(state):
    """
    Generate relevant work experience section.
    
    Based on the extracted requirements, this creates:
    - Relevant job titles and companies
    - Accomplishments that match the job requirements
    - Quantified achievements
    """
    
    llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    requirements = state["extracted_requirements"]
    job_data = state["job_data"]
    
    experience_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional resume writer. Create a work experience section that matches job requirements.
        
        Generate 2-3 relevant work experiences with:
        - Job titles that align with the target role
        - Company names (realistic but fictional)
        - 3-4 bullet points per role with quantified achievements
        - Skills and technologies that match the job requirements
        """),
        ("human", """Create work experience for someone applying to this job:

Target Job: {title} at {company}
Required Skills: {skills}
Experience Level: {level}
Key Responsibilities: {responsibilities}

Generate realistic work experience that shows progression and relevant achievements.""")
    ])
    
    formatted_prompt = experience_prompt.format_messages(
        title=job_data.get("title_raw", ""),
        company=job_data.get("company", ""),
        skills=", ".join(requirements.get("required_skills", [])),
        level=requirements.get("experience_level", ""),
        responsibilities=", ".join(requirements.get("key_responsibilities", []))
    )
    
    response = llm.invoke(formatted_prompt)
    
    # Store the experience section
    state["cv_sections"]["experience"] = response.content
    state["messages"].append(AIMessage(content=f"Generated experience section"))
    
    return state


def generate_skills_section(state):
    """
    Generate skills section highlighting relevant capabilities.
    
    Creates a skills section that:
    - Matches the job requirements
    - Includes both technical and soft skills
    - Shows proficiency levels where appropriate
    - Uses general knowledge about the role and company when description is insufficient
    """
    
    llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    requirements = state["extracted_requirements"]
    job_data = state["job_data"]
    
    skills_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional resume writer. Create a skills section that matches job requirements.
        
        Organize skills into categories:
        - Technical Skills: Programming languages, tools, technologies
        - Soft Skills: Communication, leadership, etc.
        - Industry Knowledge: Domain-specific knowledge
        
        Include proficiency levels where appropriate (Beginner/Intermediate/Advanced/Expert).
        
        IMPORTANT: If the job description isn't specific enough about required skills, rely on your general knowledge about:
        - The specific job title and what skills are typically required
        - The company type and industry to infer relevant skills
        - Common skills for similar roles in that industry
        """),
        ("human", """Create a skills section for someone applying to this job:

Job Title: {title}
Company: {company}
Job Category: {job_category}
Job Description: {description}

Required Technical Skills: {tech_skills}
Required Soft Skills: {soft_skills}
Industry Keywords: {industry}

Generate a comprehensive skills section that matches these requirements. If the description lacks specific skill details, use your knowledge of typical requirements for this job title and company type.""")
    ])
    
    formatted_prompt = skills_prompt.format_messages(
        title=job_data.get("title_raw", ""),
        company=job_data.get("company", ""),
        job_category=job_data.get("job_category", ""),
        description=job_data.get("description", ""),
        tech_skills=", ".join(requirements.get("required_skills", [])),
        soft_skills=", ".join(requirements.get("soft_skills", [])),
        industry=", ".join(requirements.get("industry_keywords", []))
    )
    
    response = llm.invoke(formatted_prompt)
    
    state["cv_sections"]["skills"] = response.content
    state["messages"].append(AIMessage(content=f"Generated skills section"))
    
    return state


def generate_education_section(state):
    """
    Generate education section.
    
    Creates education background that:
    - Matches the job requirements
    - Includes relevant coursework
    - Shows academic achievements
    """
    
    llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    requirements = state["extracted_requirements"]
    job_data = state["job_data"]
    
    education_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional resume writer. Create an education section that matches job requirements.
        
        Include:
        - Relevant degree(s) and institutions
        - Graduation years
        - Relevant coursework
        - Academic achievements (GPA, honors, etc.)
        - Certifications if relevant
        """),
        ("human", """Create an education section for someone applying to this job:

Job Title: {title}
Education Requirements: {education_req}
Required Skills: {skills}
Industry: {industry}

Generate realistic education background that supports the candidate's qualifications.""")
    ])
    
    formatted_prompt = education_prompt.format_messages(
        title=job_data.get("title_raw", ""),
        education_req=requirements.get("education_requirements", ""),
        skills=", ".join(requirements.get("required_skills", [])),
        industry=", ".join(requirements.get("industry_keywords", []))
    )
    
    response = llm.invoke(formatted_prompt)
    
    state["cv_sections"]["education"] = response.content
    state["messages"].append(AIMessage(content=f"Generated education section"))
    
    return state


def compile_final_cv(state):
    """
    Compile all sections into the final CV.
    
    Combines all generated sections into:
    - Well-formatted CV
    - Consistent styling
    - Professional presentation
    """
    
    llm = ChatOpenAI(model=default_model, temperature=default_temperature)
    
    cv_sections = state["cv_sections"]
    job_data = state["job_data"]
    
    compile_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional resume formatter. Create a well-formatted CV.
        
        Format the CV with:
        - Clear section headers
        - Consistent formatting
        - Professional appearance
        - Easy to read layout
        - Proper spacing and alignment
        """),
        ("human", """Compile this CV for the position of {title} at {company}:

Work Experience:
{experience}

Skills:
{skills}

Education:
{education}

Format this into a professional, well-structured CV.""")
    ])
    
    formatted_prompt = compile_prompt.format_messages(
        title=job_data.get("title_raw", ""),
        company=job_data.get("company", ""),
        experience=cv_sections.get("experience", ""),
        skills=cv_sections.get("skills", ""),
        education=cv_sections.get("education", "")
    )
    
    response = llm.invoke(formatted_prompt)
    
    state["final_cv"] = response.content
    state["messages"].append(AIMessage(content=f"Compiled final CV"))
    
    return state
