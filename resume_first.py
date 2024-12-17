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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Please ask questions about your resume and your match with the job application."}]
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
def handle_matching(resume_text, job_description, embeddings, weights=None):
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

    # set default weight
    default_weights = {"similarity": 0.5, "skills": 0.3, "experience": 0.2}
    weights = weights if weights else default_weights

    final_score = (
        similarity_score * weights["similarity"] +
        skill_score * weights["skills"] +
        experience_score * weights["experience"]
    )
    return {
        "similarity_score": similarity_score,
        "skill_score": skill_score,
        "experience_score": experience_score,
        "final_score": final_score,
        "matched_skills": matched_skills,
    }


def create_radar_chart(matching_results):
    # Create categories and scores
    categories = ['Similarity', 'Skills', 'Experience', 'Overall Match']
    scores = [
        matching_results['similarity_score'],
        matching_results['skill_score'],
        matching_results['experience_score'],
        matching_results['final_score']
    ]
    
    fig = go.Figure()
    
    # Add scores
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Your Match',
        fillcolor='rgba(67, 147, 195, 0.5)',
        line=dict(color='rgb(67, 147, 195)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                showline=False,
                ticks='',
                gridcolor='rgba(0,0,0,0.1)',
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
            ),
            bgcolor='rgba(0,0,0,0.02)',
        ),
        showlegend=True,
        title={
            'text': 'Job Match Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=100, b=100),
        height=600,
    )
    
    return fig


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
            [None, "fulltime", "parttime", "internship", "contract"]
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
                else:
                    st.error("No jobs found or error occurred during search.")

    # Display job search results
    if st.session_state["jobs_df"] is not None:
        # Filter out jobs with no description
        valid_jobs = st.session_state["jobs_df"].dropna(subset=['description'])
        
        if len(valid_jobs) == 0:
            st.warning("No jobs found in the search results.")
        else:
            st.success(f"Found {len(valid_jobs)} jobs")
            
            for idx, job in valid_jobs.iterrows():
                # Clean and prepare job data
                company_name = str(job.get('company', 'Unknown Company'))
                job_title = str(job.get('title', 'Untitled Position'))
                location = str(job.get('location', 'Location not specified'))
                job_type = str(job.get('job_type', 'Not specified'))
                description = str(job.get('description', '')).strip()
                
                # Handle NaN values
                if company_name == 'nan': company_name = 'Unknown Company'
                if job_title == 'nan': job_title = 'Untitled Position'
                if location == 'nan': location = 'Location not specified'
                if job_type == 'nan': job_type = 'Not specified'
                
                # Create the main job expander
                with st.expander(f"📋 {job_title} at {company_name}"):
                    st.write(f"**Location:** {location}")
                    st.write(f"**Job Type:** {job_type}")
                    
                    # Create description section with expansion functionality
                    if description and description != 'nan':
                        st.write("**Description:**")
                        
                        # Create a unique key for this job's expanded state
                        expanded_key = f"expanded_{idx}"
                        if expanded_key not in st.session_state:
                            st.session_state[expanded_key] = False
                        
                        # Find the last complete word within preview length
                        preview_length = 300
                        if len(description) > preview_length:
                            # Find the last space before preview_length
                            last_space = description[:preview_length].rfind(' ')
                            preview_text = description[:last_space] + "..."
                        else:
                            preview_text = description
                        
                        # Display the description content based on state
                        if st.session_state[expanded_key]:
                            st.markdown(description)
                            if st.button("Show less ↑", key=f"expand_{idx}", use_container_width=False):
                                st.session_state[expanded_key] = False
                                st.rerun()
                        else:
                            st.markdown(preview_text)
                            if len(description) > preview_length:
                                if st.button("Show more ↓", key=f"expand_{idx}", use_container_width=False):
                                    st.session_state[expanded_key] = True
                                    st.rerun()
                    else:
                        st.write("*No description available*")
                        continue
                    
                    st.write("")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Select This Job", key=f"select_job_{idx}", 
                                    help="Click to see detailed matching analysis for this position",
                                    use_container_width=True):
                            st.session_state["selected_job"] = job
                            st.experimental_set_query_params(selected_job_idx=idx)
                            st.rerun()

# Chat interface for selected job
elif st.session_state["selected_job"] is not None:
    if st.button("← Back to Job Search"):
        st.session_state["selected_job"] = None
        st.rerun()
    else:
        job_description = st.session_state["selected_job"].get("description", "")
        if job_description:  # Only show matching if we have a description
            # Add initial weights
            if "weights" not in st.session_state:
                st.session_state["weights"] = {"similarity": 0.5, "skills": 0.3, "experience": 0.2}

            matching_results = handle_matching(
                st.session_state["resume_text"],
                job_description,
                embeddings,
                st.session_state["weights"]
            )

            st.write(f"Similarity Score: {matching_results['similarity_score']:.2f}")
            st.write(f"Skill Match Score: {matching_results['skill_score']:.2f}")
            st.write(f"Experience Match Score: {matching_results['experience_score']:.2f}")
            st.write(f"Final Match Score: {matching_results['final_score']:.2f}")
            st.write(f"Matched Skills: {', '.join(matching_results['matched_skills'])}")

        #create donut chart for similarity scores
        def create_donut_chart(score, labels):
            scores_list = [
                    score, 1-score]
            labels_list = labels + [' ']
            
            fig, ax = plt.subplots(figsize=(6, 6))
    
            # Create the pie chart
            wedges, texts, autotexts = ax.pie(
                scores_list, 
                labels=labels_list, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['green', 'red'],
                wedgeprops={'width': 0.4}  # Adjust the width of the donut ring
            )
            
            # Add a circle in the middle to create the donut effect
            center_circle = plt.Circle((0, 0), 0.70, color='white', lw=0)
            ax.add_artist(center_circle)
            
            # Ensure the aspect ratio is equal for a circular shape
            ax.set_aspect('equal')
            
            st.pyplot(fig)

        # Streamlit dashboard setup
        st.title("Dashboard")

        # Display Similarity, Skill, Experience, and Final Match Scores
        st.header("Match Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Similarity")
            st.metric(label="job and resume compatibility", value=f"{matching_results['similarity_score']:.2f}")
            create_donut_chart(matching_results['similarity_score'], 
                        ['Similarity'])
            
            st.subheader("Experience Match")
            st.metric(label="Years of experience", value=f"{matching_results['experience_score']:.2f}")
            create_donut_chart(matching_results['experience_score'], 
                        ['Experience'])
            
            
            
            
        with col2:
            st.subheader("Skill Match")
            st.metric(label="skills alignment with job requirements", value=f"{matching_results['skill_score']:.2f}")
            create_donut_chart(matching_results['skill_score'], 
                        ['Skills'])

            st.subheader("Final Match")
            st.metric(label="overall Suitability", value=f"{matching_results['final_score']:.2f}")
        
 
            


        # Create donut chart for the match scores
        # create_donut_chart(matching_results, 
        #                 ['Similarity', 'Skill Match', 'Experience Match', 'Final Match'])

        ##################

        # Radar visualization
        st.header("Match Analysis")
        radar_fig = create_radar_chart(matching_results)
        st.plotly_chart(radar_fig, use_container_width=True)

        # Weight adjustment section
        st.header("Customize Match Score Weights")
        col1, col2, col3 = st.columns(3)
        with col1:
            similarity_weight = st.number_input(
                "Similarity Weight", 
                min_value=0.0, max_value=1.0, value=st.session_state["weights"]["similarity"], step=0.1
            )
        with col2:
            skill_weight = st.number_input(
                "Skills Weight", 
                min_value=0.0, max_value=1.0, value=st.session_state["weights"]["skills"], step=0.1
            )
        with col3:
            experience_weight = st.number_input(
                "Experience Weight", 
                min_value=0.0, max_value=1.0, value=st.session_state["weights"]["experience"], step=0.1
            )

        # Ensure the sum of the weights equals 1
        total_weight = similarity_weight + skill_weight + experience_weight
        if total_weight != 1.0:
            st.error("The sum of the weights must equal 1.0")
        else:
            if st.button("Update Weights"):
                st.session_state["weights"] = {
                    "similarity": similarity_weight,
                    "skills": skill_weight,
                    "experience": experience_weight
                }
                st.experimental_rerun()  # Re-run to update matching results


    # Job description
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
    