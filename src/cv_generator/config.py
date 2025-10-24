"""
Configuration settings for the CV Generator.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Workflow configuration
WORKFLOW_NODES = [
    "extract_requirements",
    "generate_experience", 
    "generate_skills",
    "generate_education",
    "compile_cv"
]

# Default fallback requirements
DEFAULT_FALLBACK_REQUIREMENTS = {
    "required_skills": ["Python", "Software Development"],
    "soft_skills": ["Communication", "Teamwork"],
    "experience_level": "Mid",
    "education_requirements": "Bachelor's Degree",
    "key_responsibilities": ["Develop software", "Collaborate with team"],
    "industry_keywords": ["Technology", "Software"],
    "company_culture": ["Innovative", "Collaborative"]
}
