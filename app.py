import streamlit as st
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from main import initialize_system

# Define or import load_split function
def load_split(data_dir, split_name):
    file_path = os.path.join(data_dir, f'{split_name}.json')
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
print(f"Data directory: {DATA_DIR}")

# Load both question banks
with open(os.path.join(DATA_DIR, 'question_bank_NCERT_part1.json'), encoding='utf-8') as f:
    question_bank1 = json.load(f)
with open(os.path.join(DATA_DIR, 'question_bank_NCERT_part2.json'), encoding='utf-8') as f:
    question_bank2 = json.load(f)
with open(os.path.join(DATA_DIR, 'question_bank_NCERT_part3.json'), encoding='utf-8') as f:
    question_bank3 = json.load(f)
with open(os.path.join(DATA_DIR, 'question_bank_NCERT_part4.json'), encoding='utf-8') as f:
    question_bank4 = json.load(f)
with open(os.path.join(DATA_DIR, 'question_bank_NCERT_part5.json'), encoding='utf-8') as f:
    question_bank5 = json.load(f)   

# Combine both into a single dictionary
question_bank = {}
question_bank.update(question_bank1)
question_bank.update(question_bank2)
question_bank.update(question_bank3)
question_bank.update(question_bank4)
question_bank.update(question_bank5)

# Let user select subject/topic
subjects = list(question_bank.keys())
subject = st.selectbox("Select Subject/Topic", subjects)
questions_data = question_bank[subject]
question_ids = list(questions_data.keys())

NUM_QUESTIONS = 3  # Number of random questions to ask

# Use session state to persist selected questions and answers
if 'selected_qids' not in st.session_state or st.session_state.get('last_subject') != subject:
    st.session_state.selected_qids = random.sample(question_ids, min(NUM_QUESTIONS, len(question_ids)))
    st.session_state.user_answers = {qid: "" for qid in st.session_state.selected_qids}
    st.session_state.last_subject = subject

st.title("NLP Answer Evaluation")

st.write(f"Please answer the following randomly selected questions from **{subject}**:")

# Display questions and collect answers
for qid in st.session_state.selected_qids:
    st.session_state.user_answers[qid] = st.text_area(
        f"{questions_data[qid]['question']}",
        value=st.session_state.user_answers[qid],
        key=qid
    )

# Example: Add a checkbox or radio button for spaCy usage
use_spacy = st.sidebar.checkbox("Enable spaCy semantic analysis", value=True)

with st.spinner("Initializing evaluator..."):
    evaluator = initialize_system(use_spacy=use_spacy)

if st.button("Submit Answers"):
    # Show loading indicator during initialization
    with st.spinner("Initializing evaluator..."):
        # Initialize evaluator with or without spaCy
        evaluator = initialize_system(use_spacy=use_spacy)
    
    # Prepare reference data for evaluator
    reference_data = {}
    for qid in st.session_state.selected_qids:
        reference_data[qid] = {
            "question": questions_data[qid]["question"],
            "reference_answer": questions_data[qid]["reference_answer"],
            "keywords": questions_data[qid]["keywords"]
        }

    evaluator.reference_data = reference_data

    st.header("Results")
    for qid in st.session_state.selected_qids:
        answer = st.session_state.user_answers[qid]
        if answer.strip():
            # Evaluate with loading indicator
            with st.spinner(f"Evaluating answer to question {qid}..."):
                score, breakdown = evaluator.evaluate_answer(qid, answer)
            
            st.subheader(f"Question: {questions_data[qid]['question']}")
            
            # Two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Reference Answer:**")
                st.write(questions_data[qid]['reference_answer'])
            
            with col2:
                st.write("**Your Answer:**")
                st.write(answer)
            
            # Score visualization
            st.write("**Evaluation:**")
            st.progress(float(score)/100)
            st.write(f"**Score: {score:.1f}/100**")
            
            # Score breakdown
            with st.expander("See detailed breakdown"):
                # Create a bar chart for the scores
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Collect all available scores
                scores = {
                    "Keyword": breakdown['keyword_score'],
                    "Order": breakdown['order_score'],
                    "Completeness": breakdown['completeness_score']
                }
                
                # Add semantic scores if available
                if 'semantic_score' in breakdown:
                    scores["Semantic"] = breakdown['semantic_score']
                if 'concept_coverage_score' in breakdown:
                    scores["Concept Coverage"] = breakdown['concept_coverage_score']
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                score_names = list(scores.keys())
                score_values = list(scores.values())
                
                bars = ax.bar(score_names, score_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                ax.set_ylim(0, 105)  # Scale to allow room for annotations
                ax.set_ylabel('Score (out of 100)')
                ax.set_title('Score Breakdown')
                
                # Display the chart
                st.pyplot(fig)
                
                # Textual breakdown
                st.write("**Score Details:**")
                for component, value in scores.items():
                    st.write(f"- {component} Score: {value:.1f}/100")
                
                # Keywords and concepts
                st.write("**Matched Keywords:**")
                if breakdown['matched_keywords']:
                    st.write(", ".join(breakdown['matched_keywords']))
                else:
                    st.write("None")
                
                # Show matched concepts if available
                if 'matched_concepts' in breakdown and breakdown['matched_concepts']:
                    st.write("**Matched Concepts:**")
                    # Limit to first 15 concepts to avoid clutter
                    st.write(", ".join(breakdown['matched_concepts'][:15]))
                    if len(breakdown['matched_concepts']) > 15:
                        st.write(f"...and {len(breakdown['matched_concepts']) - 15} more")
                
                # Show extracted keywords
                st.write("**Extracted Keywords from Your Answer:**")
                if 'user_keywords' in breakdown:
                    st.write(", ".join(breakdown['user_keywords']))
                else:
                    st.write("None")

    # Optionally, allow to try again with new random questions
    if st.button("Try Different Questions"):
        st.session_state.selected_qids = random.sample(question_ids, min(NUM_QUESTIONS, len(question_ids)))
        st.session_state.user_answers = {qid: "" for qid in st.session_state.selected_qids}
        st.experimental_rerun()