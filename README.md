# LinkedIn Job Postings RAG Analysis

## Project Overview
This project analyzes LinkedIn job postings using Retrieval-Augmented Generation (RAG) techniques. The dataset contains 81,920 job postings with 34 different attributes including job titles, companies, locations, salaries, and job descriptions.

## Dataset Information
- **Size**: 81,920 job postings
- **Features**: 34 columns including job metadata, company information, location data, and salary information
- **File**: `postings_linkedin_individual_0000_part_01.parquet` (250MB - not included in repository)

## Key Columns
- `job_id`: Unique identifier for each job posting
- `company`: Company name
- `title_raw` / `title_translated`: Job titles (original and translated)
- `location_raw` / `country` / `state` / `metro_area`: Location information
- `salary` / `salary_min` / `salary_max`: Salary data
- `description`: Job description text
- `remote_type`: Remote work options
- `job_category`: Job category classification
- `rics_k50` / `rics_k200` / `rics_k400`: Industry classifications


## Setup Instructions



---

**Note**: The dataset file is large (250MB) and not included in the repository. Please download it separately or contact the maintainer for access.
