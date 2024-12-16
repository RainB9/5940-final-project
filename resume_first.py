import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from jobspy import scrape_jobs
import pandas as pd
import faiss
import numpy as np
from match_alg import extract_skills_with_gpt, extract_experience_with_gpt, skill_match_score, calculate_final_score,calculate_similarity_score

# Streamlit setup
st.title("AI Job Assistant")

# Set up OpenAI proxy client
client = openai.OpenAI(
    api_key="sk-fU_9e80K6l4Erj8Ls_KlHQ",  
    base_url="https://api.ai.it.cornell.edu"  
)
embeddings = OpenAIEmbeddings(
        api_key="sk-fU_9e80K6l4Erj8Ls_KlHQ",
        base_url="https://api.ai.it.cornell.edu",
        model="openai.text-embedding-3-small"
)

# Initialize session state for uploaded resume, job search results, and selected job
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "jobs_df" not in st.session_state:
    st.session_state["jobs_df"] = None
if "selected_job" not in st.session_state:
    st.session_state["selected_job"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Upload your resume to get started."}]
if "resume_uploaded" not in st.session_state:
    st.session_state["resume_uploaded"] = False

# Function to process resume upload
def process_resume(uploaded_file):
    resume_text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
    elif uploaded_file.name.endswith(".txt"):
        resume_text += uploaded_file.read().decode("utf-8")
    return resume_text

# Function to search jobs
def search_jobs(search_term, location, job_type, days_old):
    try:
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor", "google"],
            search_term=search_term,
            location=location,
            job_type=job_type,
            hours_old=days_old * 24,  # Convert days to hours
        )
        return jobs
    except Exception as e:
        st.error(f"Error searching jobs: {str(e)}")
        return pd.DataFrame()

# Function to handle job and resume matching
def handle_matching(resume_text, job_description,embeddings):
    
    resume_skills = extract_skills_with_gpt(resume_text)
    job_skills = extract_skills_with_gpt(job_description)
    skill_score, matched_skills = skill_match_score(resume_skills, job_skills)
    
    resume_years = extract_experience_with_gpt(resume_text)
    required_years = extract_experience_with_gpt(job_description)
    
    experience_score = resume_years / required_years if required_years > 0 else 1.0
    experience_score = min(experience_score, 1.0)  

    resume_embedding = embeddings.embed_query(resume_text)
    job_embedding = embeddings.embed_query(job_description)
    similarity_score = calculate_similarity_score(resume_embedding, job_embedding)

    final_score = calculate_final_score(similarity_score, skill_score, experience_score)
    return {
        "similarity_score": similarity_score,
        "skill_score": skill_score,
        "experience_score": experience_score,
        "final_score": final_score,
        "matched_skills": matched_skills,
    }


# Resume upload interface
if not st.session_state["resume_uploaded"]:
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume", type=("txt", "pdf"))
    if uploaded_file:
        st.session_state["resume_file"] = uploaded_file  # Save the file reference
        st.session_state["resume_text"] = process_resume(uploaded_file)
        st.session_state["resume_skills"] = extract_skills_with_gpt(st.session_state["resume_text"])
        st.session_state["resume_experience"] = extract_experience_with_gpt(st.session_state["resume_text"])
        st.session_state["resume_uploaded"] = True
        st.success("Resume uploaded successfully! Proceed to job search below.")

# Job search interface
if st.session_state["resume_uploaded"] and st.session_state["selected_job"] is None:
    st.header("Job Search")
    with st.form("job_search_form"):
        search_term = st.text_input("Job Title", "Software Engineer")
        location = st.text_input("Location", "New York, NY")
        job_type = st.selectbox(
            "Job Type",
            [None, "Full-time", "Part-time", "Internship", "Contract"]
        )
        days_old = st.slider("Posted Within (days)", 1, 30, 7)

        search_submitted = st.form_submit_button("Search Jobs")
        if search_submitted:
            with st.spinner("Searching jobs across LinkedIn, Indeed, Glassdoor, and Google..."):
                jobs_df = search_jobs(
                    search_term=search_term,
                    location=location,
                    job_type=job_type,
                    days_old=days_old
                )
                if not jobs_df.empty:
                    st.session_state["jobs_df"] = jobs_df
                    st.success(f"Found {len(jobs_df)} jobs!")
                else:
                    st.error("No jobs found or error occurred during search.")

    # Display job search results
    if st.session_state["jobs_df"] is not None:
        st.subheader("Search Results")
        for idx, job in st.session_state["jobs_df"].iterrows():
            with st.expander(f"{job['title']} at {job['company']}"):
                st.write(f"**Location:** {job.get('location', 'Location not specified')}")
                st.write(f"**Job Type:** {job.get('job_type', 'Not provided')}")
                description = job.get('description', 'No description available')
                description = str(description)  
                st.write(f"**Description:** {description[:500]}...")


                if st.button("Select This Job", key=f"select_job_{idx}"):
                    st.session_state["selected_job"] = job
                    st.experimental_set_query_params(selected_job_idx=idx)
                    st.rerun()

# Chat interface for selected job
elif st.session_state["selected_job"] is not None:
    if st.button("← Back to Job Search"):
        st.session_state["selected_job"] = None

    job_description = st.session_state["selected_job"]["description"]
    
    matching_results = handle_matching(
        st.session_state["resume_text"],
        job_description,
        embeddings
    )
    st.write(f"Similarity Score: {matching_results['similarity_score']:.2f}")
    st.write(f"Skill Match Score: {matching_results['skill_score']:.2f}")
    st.write(f"Experience Match Score: {matching_results['experience_score']:.2f}")
    st.write(f"Final Match Score: {matching_results['final_score']:.2f}")
    st.write(f"Matched Skills: {', '.join(matching_results['matched_skills'])}")

    st.header(f"Selected Job: {st.session_state['selected_job']['title']}")
    st.subheader(f"Company: {st.session_state['selected_job']['company']}")

    st.write("**Resume File:**")
    if "resume_file" in st.session_state:
        st.download_button(
            label="Download Resume",
            data=st.session_state["resume_file"].getvalue(),
            file_name=st.session_state["resume_file"].name,
            mime=st.session_state["resume_file"].type,
        )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question about this job")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        context = f"Resume:\n{st.session_state['resume_text']}\n\nJob Description:\n{st.session_state['selected_job']['description']}"
        # Generate response
        prompt_template = """
            You are an assistant for job applications. Use the following resume context to answer the question. 
            If you don't know the answer, say that you don't know. Answer concisely.

            Question: {question}

            Context: {context}

            Answer:
        """
        formatted_prompt = prompt_template.format(
                question=question,
                context=context
        )
        # Updated chat completion with correct model name
        response = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        assistant_response = response.choices[0].message.content
        st.chat_message("assistant").write(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    