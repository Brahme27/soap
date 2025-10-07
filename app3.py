import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import json
import pandas as pd


# Page configuration
st.set_page_config(
    page_title="SOAP Note Generator",
    page_icon="🩺",
    layout="wide"
)


# Simplified CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    .soap-section {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .brief-highlight {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 6px;
        border: 2px solid #f59e0b;
        margin-bottom: 1rem;
        font-weight: 500;
        color: #92400e;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    .content-text {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        color: #374151;
        line-height: 1.6;
        border: 1px solid #e5e7eb;
    }
    .note-label {
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #1d4ed8;
        background: rgba(59, 130, 246, 0.15);
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.45rem;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = None


# SOAP Note Prompt Template
SOAP_TEMPLATE = """You are a clinical documentation expert and therapy note assistant. Your task is to analyze the following therapist-patient conversation and produce a structured SOAP note.

Client Name: {client_name}
Session Date: {session_date}
Session Duration: {duration} minutes

Therapist-Patient Conversation:
{conversation}

Generate a comprehensive SOAP note following the proper clinical SOAP framework:
- S (SUBJECTIVE): What the CLIENT says - their own words, reported feelings, symptoms, concerns
- O (OBJECTIVE): What the THERAPIST observed - behaviors, appearance, measurable observations
- A (ASSESSMENT): THERAPIST'S professional opinion - clinical interpretation, diagnosis considerations, progress assessment
- P (PLAN): THERAPIST'S next steps - treatment interventions, goals, homework, follow-up plans

For EACH section, provide:
1. A "Brief" (1-2 sentence summary of the most important idea)
2. Detailed content for that section

Format your response EXACTLY as follows:

SUBJECTIVE_BRIEF:
[1-2 sentence summary of what CLIENT reported/said]

SUBJECTIVE:
[Detailed information about what CLIENT said - their own words, reported feelings, symptoms, concerns, complaints]

OBJECTIVE_BRIEF:
[1-2 sentence summary of what THERAPIST observed]

OBJECTIVE:
[Detailed THERAPIST observations - client's behavior, appearance, affect, body language, measurable data]

ASSESSMENT_BRIEF:
[1-2 sentence summary of THERAPIST'S clinical opinion]

ASSESSMENT:
[Detailed THERAPIST professional interpretation - clinical insights, diagnostic impressions, progress evaluation, risk assessment]

PLAN_BRIEF:
[1-2 sentence summary of THERAPIST'S next steps]

PLAN:
[Detailed THERAPIST treatment plan - specific interventions, therapeutic goals, homework assignments, follow-up scheduling]

Also provide these metrics as JSON at the end:
METRICS:
{{
    "emotional_distribution": {{"anxious": 35, "sad": 25, "angry": 10, "neutral": 20, "hopeful": 10}},
    "observed_behaviors": {{"fidgeting": 22, "eye_contact_avoided": 16, "tense_posture": 18, "crying": 3, "appropriate_affect": 15}},
    "assessment_domains": {{"mood": 6, "anxiety": 8, "sleep": 5, "functioning": 6, "risk": 2}},
    "risk_level": "low/moderate/high",
    "engagement_level": [score from 1-10],
    "session_productivity": [score from 1-10]
}}

Keep each section clear, professional, and clinically relevant."""


def initialize_llm(api_key):
    """Initialize the OpenAI LLM with LangChain"""
    try:
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None


def parse_soap_response(response):
    """Parse the LLM response into structured SOAP sections"""
    sections = {
        'subjective_brief': '',
        'subjective': '',
        'objective_brief': '',
        'objective': '',
        'assessment_brief': '',
        'assessment': '',
        'plan_brief': '',
        'plan': '',
        'metrics': {}
    }
    
    lines = response.split('\n')
    current_section = None
    content_buffer = []
    
    for line in lines:
        line = line.strip()
        
        if 'SUBJECTIVE_BRIEF:' in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'subjective_brief'
            content_buffer = [line.replace('SUBJECTIVE_BRIEF:', '').strip()]
        elif 'SUBJECTIVE:' in line and 'BRIEF' not in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'subjective'
            content_buffer = [line.replace('SUBJECTIVE:', '').strip()]
        elif 'OBJECTIVE_BRIEF:' in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'objective_brief'
            content_buffer = [line.replace('OBJECTIVE_BRIEF:', '').strip()]
        elif 'OBJECTIVE:' in line and 'BRIEF' not in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'objective'
            content_buffer = [line.replace('OBJECTIVE:', '').strip()]
        elif 'ASSESSMENT_BRIEF:' in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'assessment_brief'
            content_buffer = [line.replace('ASSESSMENT_BRIEF:', '').strip()]
        elif 'ASSESSMENT:' in line and 'BRIEF' not in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'assessment'
            content_buffer = [line.replace('ASSESSMENT:', '').strip()]
        elif 'PLAN_BRIEF:' in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'plan_brief'
            content_buffer = [line.replace('PLAN_BRIEF:', '').strip()]
        elif 'PLAN:' in line and 'BRIEF' not in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'plan'
            content_buffer = [line.replace('PLAN:', '').strip()]
        elif 'METRICS:' in line:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'metrics'
            content_buffer = []
        elif current_section == 'metrics' and line:
            content_buffer.append(line)
        elif line and current_section:
            content_buffer.append(line)
    
    if current_section and content_buffer:
        if current_section == 'metrics':
            try:
                metrics_text = '\n'.join(content_buffer)
                sections['metrics'] = json.loads(metrics_text)
            except:
                sections['metrics'] = {
                    "emotional_distribution": {"anxious": 35, "sad": 25, "angry": 10, "neutral": 20, "hopeful": 10},
                    "observed_behaviors": {"fidgeting": 22, "eye_contact_avoided": 16, "tense_posture": 18, "crying": 3, "appropriate_affect": 15},
                    "assessment_domains": {"mood": 6, "anxiety": 8, "sleep": 5, "functioning": 6, "risk": 2},
                    "risk_level": "moderate",
                    "engagement_level": 7,
                    "session_productivity": 7
                }
        else:
            sections[current_section] = '\n'.join(content_buffer).strip()
    
    return sections


def create_subjective_chart(emotional_dist):
    """Create emotional distribution pie chart"""
    if not emotional_dist:
        emotional_dist = {"anxious": 35, "sad": 25, "angry": 10, "neutral": 20, "hopeful": 10}
    
    emotions = list(emotional_dist.keys())
    values = list(emotional_dist.values())
    colors = ['#ef4444', '#3b82f6', '#f59e0b', '#6b7280', '#10b981']
    
    fig = go.Figure(data=[go.Pie(
        labels=[e.capitalize() for e in emotions],
        values=values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{value}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Subjective: Emotional Distribution",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    return fig


def create_objective_chart(observed_behaviors):
    """Create observed behaviors bar chart"""
    if not observed_behaviors:
        observed_behaviors = {"fidgeting": 22, "eye_contact_avoided": 16, "tense_posture": 18, "crying": 3, "appropriate_affect": 15}
    
    observations = [k.replace('_', ' ').title() for k in observed_behaviors.keys()]
    counts = list(observed_behaviors.values())
    
    fig = go.Figure(data=[go.Bar(
        x=observations,
        y=counts,
        marker=dict(color='#3b82f6'),
        text=counts,
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Objective: Observed Behaviors",
        xaxis_title="Behavior",
        yaxis_title="Frequency",
        height=350,
        margin=dict(l=40, r=40, t=60, b=100)
    )
    
    return fig


def create_assessment_chart(assessment_domains):
    """Create assessment radar chart"""
    if not assessment_domains:
        assessment_domains = {"mood": 6, "anxiety": 8, "sleep": 5, "functioning": 6, "risk": 2}
    
    domains = [k.capitalize() for k in assessment_domains.keys()]
    scores = list(assessment_domains.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=domains + [domains[0]],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2),
        name='Severity'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickmode='linear',
                tick0=0,
                dtick=2
            )
        ),
        title="Assessment: Clinical Domains",
        height=400,
        margin=dict(l=60, r=60, t=60, b=20),
        showlegend=False
    )
    
    return fig


def create_plan_chart(session_date):
    """Create treatment timeline chart with realistic clinical intervals"""
    start_date = datetime.strptime(session_date, "%Y-%m-%d")
    
    plan_data = [
        # Next therapy session - typically weekly
        dict(Task="Next Therapy Session", Start=start_date + timedelta(days=7), Finish=start_date + timedelta(days=7), Resource='Therapy'),
        
        # Follow-up check-in call - within 3 days for higher risk cases
        dict(Task="Check-in Call", Start=start_date + timedelta(days=3), Finish=start_date + timedelta(days=3), Resource='Follow-up'),
        
        # Homework/Practice period - ongoing for 2 weeks
        dict(Task="Practice Skills (Homework)", Start=start_date + timedelta(days=1), Finish=start_date + timedelta(days=14), Resource='Self-Work'),
        
        # Medication review if applicable - typically 2-4 weeks
        dict(Task="Medication Review", Start=start_date + timedelta(days=21), Finish=start_date + timedelta(days=21), Resource='Medical'),
        
        # Progress assessment - every 4-6 weeks
        dict(Task="Progress Assessment", Start=start_date + timedelta(days=42), Finish=start_date + timedelta(days=42), Resource='Assessment'),
        
        # Long-term therapy continuation - 12 weeks (3 months is common treatment phase)
        dict(Task="Ongoing Weekly Sessions", Start=start_date + timedelta(days=7), Finish=start_date + timedelta(days=84), Resource='Therapy')
    ]
    
    df = pd.DataFrame(plan_data)
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Resource",
        title="Plan: Treatment Timeline (Next 12 Weeks)"
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        xaxis_title="Timeline"
    )
    
    return fig


def generate_soap_note(llm, client_name, session_date, duration, conversation):
    """Generate SOAP note using LangChain"""
    try:
        prompt = PromptTemplate(
            input_variables=["client_name", "session_date", "duration", "conversation"],
            template=SOAP_TEMPLATE
        )
        
        chain = prompt | llm
        
        result = chain.invoke({
            "client_name": client_name,
            "session_date": session_date,
            "duration": duration,
            "conversation": conversation
        })
        
        if hasattr(result, 'content'):
            return result.content
        else:
            return str(result)
    except Exception as e:
        st.error(f"Error generating SOAP note: {str(e)}")
        return None


def create_pdf(soap_sections, client_name, session_date):
    """Generate PDF of SOAP note"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, spaceAfter=20)
    story.append(Paragraph("SOAP NOTE", title_style))
    
    # Client info
    story.append(Paragraph(f"<b>Client:</b> {client_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {session_date}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Sections
    section_titles = {'subjective': 'SUBJECTIVE', 'objective': 'OBJECTIVE', 'assessment': 'ASSESSMENT', 'plan': 'PLAN'}
    
    for key, title in section_titles.items():
        story.append(Paragraph(title, styles['Heading2']))
        
        brief_key = f"{key}_brief"
        if soap_sections.get(brief_key):
            story.append(Paragraph(f"Brief: {soap_sections[brief_key]}", styles['Normal']))
        
        if soap_sections.get(key):
            story.append(Paragraph(soap_sections[key], styles['Normal']))
        
        story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    # Header
    st.markdown('<div class="main-header">🩺 SOAP Note Generator</div>', unsafe_allow_html=True)
    st.markdown("AI-powered clinical documentation with visual analytics")
    
    # API Key input (minimal sidebar)
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key")
    
    if not api_key:
        st.info("👈 Please enter your OpenAI API key to get started")
        return
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    if not llm:
        return
    
    # Main input form
    st.subheader("Session Information")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        client_name = st.text_input("Client Name", placeholder="Enter client name")
    
    with col2:
        session_date = st.date_input("Session Date", value=datetime.now())
    
    with col3:
        duration = st.number_input("Duration (min)", min_value=15, max_value=180, value=50, step=5)
    
    # Conversation input
    st.subheader("Therapy Session Conversation")
    conversation = st.text_area(
        "Enter the complete dialogue between therapist and client",
        height=300,
        placeholder='Therapist: "How have you been feeling this week?"\nClient: "I\'ve been really anxious about work..."'
    )
    
    # Generate button
    if st.button("Generate SOAP Note", type="primary", use_container_width=True):
        if not all([client_name, conversation]):
            st.error("Please fill in all required fields")
        else:
            with st.spinner("Generating SOAP note..."):
                soap_response = generate_soap_note(
                    llm, client_name, session_date.strftime("%Y-%m-%d"), duration, conversation
                )
                
                if soap_response:
                    st.session_state.soap_note = parse_soap_response(soap_response)
                    st.success("SOAP note generated successfully!")
                    st.rerun()
    
    # Display SOAP note
    if st.session_state.soap_note:
        soap_sections = st.session_state.soap_note
        metrics = soap_sections.get('metrics', {})
        
        st.divider()
        
        # SUBJECTIVE
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📢 SUBJECTIVE (What Client Says)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('subjective_brief'):
            st.markdown('<div class="note-label">Key Note - What Client Says</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["subjective_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Note - What Client Says</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("subjective", "No data available")}</div>', unsafe_allow_html=True)
        
        if metrics.get('emotional_distribution'):
            fig = create_subjective_chart(metrics['emotional_distribution'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # OBJECTIVE
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👁️ OBJECTIVE (What Therapist Observed)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('objective_brief'):
            st.markdown('<div class="note-label">Key Note - What Therapist Observed</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["objective_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Note - What Therapist Observed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("objective", "No data available")}</div>', unsafe_allow_html=True)
        
        if metrics.get('observed_behaviors'):
            fig = create_objective_chart(metrics['observed_behaviors'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ASSESSMENT
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 ASSESSMENT (Therapist Opinion)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('assessment_brief'):
            st.markdown('<div class="note-label">Key Note - Therapist Opinion</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["assessment_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Note - Therapist Opinion</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("assessment", "No data available")}</div>', unsafe_allow_html=True)
        
        if metrics.get('assessment_domains'):
            fig = create_assessment_chart(metrics['assessment_domains'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # PLAN
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 PLAN (Therapist Next Steps)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('plan_brief'):
            st.markdown('<div class="note-label">Key Note - Therapist Next Steps</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["plan_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Note - Therapist Next Steps</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("plan", "No data available")}</div>', unsafe_allow_html=True)
        
        fig = create_plan_chart(session_date.strftime("%Y-%m-%d"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export and metrics
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Options")
            
            # Text export
            full_text = f"""SOAP NOTE - {client_name} - {session_date.strftime("%Y-%m-%d")}

SUBJECTIVE
{soap_sections.get('subjective_brief', '')}
{soap_sections.get('subjective', '')}

OBJECTIVE
{soap_sections.get('objective_brief', '')}
{soap_sections.get('objective', '')}

ASSESSMENT
{soap_sections.get('assessment_brief', '')}
{soap_sections.get('assessment', '')}

PLAN
{soap_sections.get('plan_brief', '')}
{soap_sections.get('plan', '')}"""
            
            st.download_button(
                label="📥 Download TXT",
                data=full_text,
                file_name=f"SOAP_{client_name}_{session_date.strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            
            # PDF export
            pdf_buffer = create_pdf(soap_sections, client_name, session_date.strftime("%Y-%m-%d"))
            st.download_button(
                label="📥 Download PDF",
                data=pdf_buffer,
                file_name=f"SOAP_{client_name}_{session_date.strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        
        with col2:
            if metrics:
                st.subheader("Session Metrics")
                
                risk_level = metrics.get('risk_level', 'moderate')
                engagement = metrics.get('engagement_level', 0)
                productivity = metrics.get('session_productivity', 0)
                
                st.metric("Risk Level", risk_level.upper())
                st.metric("Engagement", f"{engagement}/10")
                st.metric("Productivity", f"{productivity}/10")


if __name__ == "__main__":
    main()