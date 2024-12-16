
import openai
import faiss
import numpy as np
import re
client = openai.OpenAI(
    api_key="sk-fU_9e80K6l4Erj8Ls_KlHQ",  
    base_url="https://api.ai.it.cornell.edu"  
)

# Function to extract skills from text using OpenAI
def extract_skills_with_gpt(text):
    prompt = f"Extract all relevant skills from the following text:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model= 'openai.gpt-4o',  
            messages=[
                {"role": "system", "content": "You are an expert in skill extraction."},
                {"role": "user", "content": prompt}
            ]
        )
        print(response)
        message_content = response.choices[0].message.content
        skills = [
            skill.strip().lstrip("-").strip()
            for skill in message_content.split("\n")
            if skill.strip()  
        ]
        return skills
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []


def extract_years_from_text(text):
    try:
        match = re.search(r"(\d+(\.\d+)?)\s*(年|月|years?|months?)", text, re.IGNORECASE)
        if match:
            number = float(match.group(1)) 
            unit = match.group(3).lower()  
            if "月" in unit or "month" in unit:  
                return number / 12
            return number  
    except Exception as e:
        print(f"Error parsing years from text: {e}")
    return 0 


def extract_experience_with_gpt(text):
    prompt = f"""
    Extract the total number of years of professional work experience mentioned in the following text.
    If experience is mentioned in months, return the value with the unit (e.g., '2 months').
    If no experience is mentioned, return '0 years':
    {text}
    """
    try:
        response = client.chat.completions.create(
            model='openai.gpt-4o',
            messages=[
                {"role": "system", "content": "You are an expert in extracting work experience from text."},
                {"role": "user", "content": prompt}
            ]
        )
        message_content = response.choices[0].message.content.strip()
        print(f"GPT Response: {message_content}") 
        experience_years = extract_years_from_text(message_content)
        return experience_years
    except Exception as e:
        print(f"Error extracting experiences: {e}")
        return 0

    
# skill match score
def skill_match_score(resume_skills, job_skills):
    matched_skills = set(resume_skills) & set(job_skills)  
    skill_score = len(matched_skills) / len(job_skills) if job_skills else 0 
    return skill_score, matched_skills



def calculate_similarity_score(resume_embedding, job_embedding):
    try:
        dimension = len(resume_embedding)  
        index = faiss.IndexFlatL2(dimension)  
        index.add(np.array([resume_embedding]))  
        D, _ = index.search(np.array([job_embedding]), k=1)  
        similarity_score = 1 / (1 + D[0][0])
        return similarity_score
    except Exception as e:
        print(f"Error calculating similarity score: {e}")
        return 0


# cal similarity score
def calculate_final_score(similarity_score, skill_score, experience_score):
    weights = {"similarity": 0.5, "skills": 0.3, "experience": 0.2}
    final_score = (
        similarity_score * weights["similarity"] +
        skill_score * weights["skills"] +
        experience_score * weights["experience"]
    )
    return final_score
