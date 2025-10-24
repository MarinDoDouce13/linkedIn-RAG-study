# CV Generator Module

This module provides a LangGraph-based CV generation tool that creates tailored resumes based on job posting data.

## Structure

```
src/
├── cv_generator/
│   ├── __init__.py          # Main module interface
│   ├── workflow.py          # LangGraph workflow and state definitions
│   ├── nodes.py            # Individual workflow node functions
│   └── config.py           # Configuration settings
└── example.py              # Example usage script
```

## Usage

### Basic Usage

```python
import sys
import os
sys.path.append('src')

from cv_generator import generate_cv_for_job

# Your job data
job_data = {
    "title_raw": "Software Engineer",
    "company": "TechCorp",
    "description": "Looking for Python developer...",
    "job_category": "Technology",
    "role_k50": "Software Developer"
}

# Generate CV
cv = generate_cv_for_job(job_data)
print(cv)
```

### With API Key

```python
cv = generate_cv_for_job(job_data, api_key="your-api-key")
```

## Workflow

The CV generation follows this LangGraph workflow:

1. **Extract Requirements** - Analyze job posting for key requirements
2. **Generate Experience** - Create relevant work experience section
3. **Generate Skills** - Create skills section matching job requirements
4. **Generate Education** - Create education background
5. **Compile CV** - Combine all sections into final formatted CV

## Configuration

Set these environment variables in your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.1
```

## Running Examples

```bash
# Run the example script
python src/example.py

# Run the test script from project root
python test_cv_generator.py
```
