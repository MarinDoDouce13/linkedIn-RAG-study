"""
Example usage of the CV Generator module.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cv_generator import generate_cv_for_job

# Load environment variables
load_dotenv()

def main():
    """
    Example of how to use the CV generator.
    """
    
    # Sample job data (based on your dataset structure)
    sample_job = {
        "title_raw": "Software Engineer",
        "title_translated": "Software Engineer",
        "company": "TechCorp Solutions",
        "description": "We are looking for a skilled Software Engineer to join our development team. The ideal candidate will have experience with Python, JavaScript, and modern web frameworks. You will be responsible for developing scalable applications, collaborating with cross-functional teams, and contributing to our innovative projects.",
        "job_category": "Technology",
        "role_k50": "Software Developer",
        "role_k150": "Senior Software Developer",
        "location_raw": "Toronto, ON",
        "country": "Canada",
        "salary_min": 80000,
        "salary_max": 120000,
        "remote_type": "Hybrid"
    }
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set your OPENAI_API_KEY in a .env file")
        print("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    print("Generating CV for:", sample_job["title_raw"], "at", sample_job["company"])
    print("=" * 60)
    
    try:
        # Generate the CV
        cv = generate_cv_for_job(sample_job, api_key)
        
        print("Generated CV:")
        print("=" * 60)
        print(cv)
        
        # Save to file
        with open("generated_cv.txt", "w", encoding="utf-8") as f:
            f.write(cv)
        
        print("\nCV saved to 'generated_cv.txt'")
        
    except Exception as e:
        print(f"Error generating CV: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY in a .env file")
        print("2. Installed all requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
