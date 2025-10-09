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


# Enhanced CSS
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
    .context-box {
        background: #eff6ff;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = None


# Enhanced SOAP Note Prompt Template
SOAP_TEMPLATE = """You are a licensed clinical documentation expert and therapy assistant. Your role is to analyze the therapist-client conversation and generate realistic, clinically accurate SOAP notes with supporting metrics for visualization. All content must be tied directly to the conversation.

Client Name: {client_name}
Session Date: {session_date}
Session Duration: {duration} minutes

Therapist-Client Conversation:
{conversation}

Generate a comprehensive SOAP note following these rules:

1. **SUBJECTIVE**: What the CLIENT says
   - Include direct quotes from the client
   - Report feelings, symptoms, and concerns
   - Highlight the most important points in a brief summary

2. **OBJECTIVE**: What the THERAPIST observed
   - Report observed behaviors (fidgeting, avoiding eye contact, tense posture, crying, affect)
   - Include measurable observations and therapist notes
   - Highlight key observations in a brief summary

3. **ASSESSMENT**: THERAPIST'S clinical opinion
   - Provide clinical interpretation, diagnostic considerations, and progress assessment
   - Include risk evaluation
   - Highlight key assessment points in a brief summary

4. **PLAN**: THERAPIST'S actionable next steps
   - Suggest specific interventions, homework, follow-ups, therapy sessions
   - Include realistic timelines for each step
   - Highlight key plan points in a brief summary

5. **Metrics and Visual Data**:
   - Extract emotional distribution (percentages must sum to 100)
   - Provide emotional context describing situations triggering each emotion
   - Count observed behaviors accurately from the conversation
   - Rate assessment domains (mood, anxiety, sleep, functioning, risk) on 0-10 scale
   - Rate engagement level (1-10) and session productivity (1-10)
   - Propose plan events with dates for next 12 weeks

Format your response EXACTLY as follows:

SUBJECTIVE_BRIEF:
[1-2 sentence summary of client-reported feelings and concerns]

SUBJECTIVE:
[Detailed subjective notes with direct client quotes and reported symptoms]

OBJECTIVE_BRIEF:
[1-2 sentence summary of therapist observations]

OBJECTIVE:
[Detailed observed behaviors, measurable notes, and context tied to conversation]

ASSESSMENT_BRIEF:
[1-2 sentence summary of clinical assessment]

ASSESSMENT:
[Detailed clinical interpretation, progress, risk assessment, and reasoning]

PLAN_BRIEF:
[1-2 sentence summary of next steps]

PLAN:
[Detailed, actionable plan including homework, interventions, therapy sessions, and follow-up]

METRICS:
{{
  "emotional_distribution": {{"anxious": 35, "sad": 25, "angry": 10, "neutral": 20, "hopeful": 10}},
  "emotional_context": {{
    "anxious": "when discussing work deadlines and performance reviews",
    "sad": "when talking about family relationships",
    "angry": "when recalling past trauma",
    "neutral": "during check-in questions",
    "hopeful": "when discussing coping strategies"
  }},
  "observed_behaviors": {{
    "fidgeting": 22,
    "eye_contact_avoided": 16,
    "tense_posture": 18,
    "crying": 3,
    "appropriate_affect": 15
  }},
  "assessment_domains": {{
    "mood": 6,
    "anxiety": 8,
    "sleep": 5,
    "functioning": 6,
    "risk": 2
  }},
  "risk_level": "moderate",
  "engagement_level": 7,
  "session_productivity": 7,
  "plan_timeline": [
    {{"Task": "Next Therapy Session", "Start": "{next_week}", "Finish": "{next_week}", "Resource": "Therapy"}},
    {{"Task": "Practice Coping Skills", "Start": "{tomorrow}", "Finish": "{two_weeks}", "Resource": "Self-Work"}},
    {{"Task": "Check-in Call", "Start": "{three_days}", "Finish": "{three_days}", "Resource": "Follow-Up"}},
    {{"Task": "Progress Assessment", "Start": "{six_weeks}", "Finish": "{six_weeks}", "Resource": "Assessment"}}
  ]
}}

CRITICAL REQUIREMENTS:
- Base ALL content on the actual conversation provided
- Use direct quotes from the client wherever possible
- Ensure emotional distribution percentages sum to exactly 100
- Make behavior counts realistic based on conversation content
- Avoid generic filler content - be specific and accurate
- Dates in plan_timeline should follow format YYYY-MM-DD"""


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
        line_stripped = line.strip()
        
        if 'SUBJECTIVE_BRIEF:' in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'subjective_brief'
            content_buffer = [line_stripped.replace('SUBJECTIVE_BRIEF:', '').strip()]
        elif line_stripped.startswith('SUBJECTIVE:') and 'BRIEF' not in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'subjective'
            content_buffer = [line_stripped.replace('SUBJECTIVE:', '').strip()]
        elif 'OBJECTIVE_BRIEF:' in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'objective_brief'
            content_buffer = [line_stripped.replace('OBJECTIVE_BRIEF:', '').strip()]
        elif line_stripped.startswith('OBJECTIVE:') and 'BRIEF' not in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'objective'
            content_buffer = [line_stripped.replace('OBJECTIVE:', '').strip()]
        elif 'ASSESSMENT_BRIEF:' in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'assessment_brief'
            content_buffer = [line_stripped.replace('ASSESSMENT_BRIEF:', '').strip()]
        elif line_stripped.startswith('ASSESSMENT:') and 'BRIEF' not in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'assessment'
            content_buffer = [line_stripped.replace('ASSESSMENT:', '').strip()]
        elif 'PLAN_BRIEF:' in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'plan_brief'
            content_buffer = [line_stripped.replace('PLAN_BRIEF:', '').strip()]
        elif line_stripped.startswith('PLAN:') and 'BRIEF' not in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'plan'
            content_buffer = [line_stripped.replace('PLAN:', '').strip()]
        elif 'METRICS:' in line_stripped:
            if current_section and content_buffer:
                sections[current_section] = '\n'.join(content_buffer).strip()
            current_section = 'metrics'
            content_buffer = []
        elif current_section == 'metrics' and line_stripped:
            content_buffer.append(line_stripped)
        elif line_stripped and current_section and current_section != 'metrics':
            content_buffer.append(line_stripped)
    
    # Save final section
    if current_section and content_buffer:
        if current_section == 'metrics':
            try:
                metrics_text = '\n'.join(content_buffer)
                # Handle cases where the JSON might be wrapped in ```json blocks
                metrics_text = metrics_text.replace('```json', '').replace('```', '').strip()
                sections['metrics'] = json.loads(metrics_text)
            except json.JSONDecodeError as e:
                st.warning(f"Could not parse metrics JSON: {e}. Using default values.")
                sections['metrics'] = get_default_metrics()
        else:
            sections[current_section] = '\n'.join(content_buffer).strip()
    
    # Ensure metrics exist
    if not sections['metrics']:
        sections['metrics'] = get_default_metrics()
    
    return sections


def get_default_metrics():
    """Return default metrics structure"""
    return {
        "emotional_distribution": {"anxious": 35, "sad": 25, "angry": 10, "neutral": 20, "hopeful": 10},
        "emotional_context": {
            "anxious": "when discussing stressful situations",
            "sad": "when talking about losses",
            "angry": "when recalling frustrating events",
            "neutral": "during routine check-in",
            "hopeful": "when discussing future plans"
        },
        "observed_behaviors": {
            "fidgeting": 20,
            "eye_contact_avoided": 15,
            "tense_posture": 18,
            "crying": 3,
            "appropriate_affect": 15
        },
        "assessment_domains": {
            "mood": 6,
            "anxiety": 7,
            "sleep": 5,
            "functioning": 6,
            "risk": 2
        },
        "risk_level": "moderate",
        "engagement_level": 7,
        "session_productivity": 7,
        "plan_timeline": []
    }


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
        title="Emotional Distribution During Session",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig


def create_objective_chart(observed_behaviors):
    """Create observed behaviors bar chart"""
    if not observed_behaviors:
        observed_behaviors = {"fidgeting": 22, "eye_contact_avoided": 16, "tense_posture": 18, "crying": 3, "appropriate_affect": 15}
    
    observations = [k.replace('_', ' ').title() for k in observed_behaviors.keys()]
    counts = list(observed_behaviors.values())
    
    colors = ['#ef4444' if count > 15 else '#3b82f6' for count in counts]
    
    fig = go.Figure(data=[go.Bar(
        x=observations,
        y=counts,
        marker=dict(color=colors),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Observed Behaviors (Frequency)",
        xaxis_title="Behavior Type",
        yaxis_title="Frequency Count",
        height=400,
        margin=dict(l=40, r=40, t=60, b=120),
        xaxis_tickangle=-45
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
        name='Severity Score',
        hovertemplate='<b>%{theta}</b><br>Score: %{r}/10<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickmode='linear',
                tick0=0,
                dtick=2,
                showticklabels=True
            )
        ),
        title="Clinical Assessment Domains (0-10 Scale)",
        height=450,
        margin=dict(l=80, r=80, t=60, b=20),
        showlegend=False
    )
    
    return fig


def create_plan_chart(plan_timeline):
    """Create treatment timeline chart from provided plan_timeline"""
    if not plan_timeline or len(plan_timeline) == 0:
        # Default timeline if none provided
        start_date = datetime.now()
        plan_timeline = [
            {"Task": "Next Therapy Session", "Start": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), "Resource": "Therapy"},
            {"Task": "Practice Skills", "Start": (start_date + timedelta(days=1)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=14)).strftime("%Y-%m-%d"), "Resource": "Self-Work"},
            {"Task": "Check-in Call", "Start": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), "Resource": "Follow-Up"},
            {"Task": "Progress Assessment", "Start": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), "Resource": "Assessment"}
        ]
    
    df = pd.DataFrame(plan_timeline)
    
    # Convert string dates to datetime
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])
    
    color_map = {
        'Therapy': '#3b82f6',
        'Self-Work': '#10b981',
        'Follow-Up': '#f59e0b',
        'Assessment': '#8b5cf6',
        'Medical': '#ef4444'
    }
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Resource",
        title="Treatment Plan Timeline (Next 12 Weeks)",
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        xaxis_title="Timeline",
        yaxis_title=""
    )
    
    fig.update_yaxes(categoryorder='total ascending')
    
    return fig


def generate_soap_note(llm, client_name, session_date, duration, conversation):
    """Generate SOAP note using LangChain with dynamic dates"""
    try:
        # Calculate dynamic dates for the prompt
        session_dt = datetime.strptime(session_date, "%Y-%m-%d")
        dates = {
            "tomorrow": (session_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            "three_days": (session_dt + timedelta(days=3)).strftime("%Y-%m-%d"),
            "next_week": (session_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
            "two_weeks": (session_dt + timedelta(days=14)).strftime("%Y-%m-%d"),
            "six_weeks": (session_dt + timedelta(days=42)).strftime("%Y-%m-%d")
        }
        
        prompt = PromptTemplate(
            input_variables=["client_name", "session_date", "duration", "conversation", 
                           "tomorrow", "three_days", "next_week", "two_weeks", "six_weeks"],
            template=SOAP_TEMPLATE
        )
        
        chain = prompt | llm
        
        result = chain.invoke({
            "client_name": client_name,
            "session_date": session_date,
            "duration": duration,
            "conversation": conversation,
            **dates
        })
        
        if hasattr(result, 'content'):
            return result.content
        else:
            return str(result)
    except Exception as e:
        st.error(f"Error generating SOAP note: {str(e)}")
        return None


def create_pdf(soap_sections, client_name, session_date, metrics):
    """Generate enhanced PDF of SOAP note with metrics"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                fontSize=24, spaceAfter=20, textColor='#2563eb')
    story.append(Paragraph("CLINICAL SOAP NOTE", title_style))
    
    # Client info
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11, spaceAfter=6)
    story.append(Paragraph(f"<b>Client:</b> {client_name}", info_style))
    story.append(Paragraph(f"<b>Session Date:</b> {session_date}", info_style))
    
    if metrics:
        story.append(Paragraph(f"<b>Risk Level:</b> {metrics.get('risk_level', 'N/A').upper()}", info_style))
        story.append(Paragraph(f"<b>Engagement:</b> {metrics.get('engagement_level', 'N/A')}/10", info_style))
        story.append(Paragraph(f"<b>Productivity:</b> {metrics.get('session_productivity', 'N/A')}/10", info_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Sections
    section_titles = {
        'subjective': 'SUBJECTIVE (Client Reports)',
        'objective': 'OBJECTIVE (Therapist Observations)',
        'assessment': 'ASSESSMENT (Clinical Opinion)',
        'plan': 'PLAN (Next Steps)'
    }
    
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], 
                                  fontSize=14, textColor='#1e40af', spaceAfter=10)
    brief_style = ParagraphStyle('Brief', parent=styles['Normal'], 
                                fontSize=10, textColor='#92400e', 
                                backColor='#fef3c7', borderPadding=8)
    
    for key, title in section_titles.items():
        story.append(Paragraph(title, section_style))
        
        brief_key = f"{key}_brief"
        if soap_sections.get(brief_key):
            story.append(Paragraph(f"<b>Brief:</b> {soap_sections[brief_key]}", brief_style))
            story.append(Spacer(1, 0.1*inch))
        
        if soap_sections.get(key):
            story.append(Paragraph(soap_sections[key], styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    # Header
    st.markdown('<div class="main-header">🩺 Clinical SOAP Note Generator</div>', unsafe_allow_html=True)
    st.markdown("AI-powered clinical documentation with comprehensive visual analytics")
    
    # API Key input
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to enable note generation")
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key")
        
        st.divider()
        st.markdown("### 📋 About SOAP Notes")
        st.markdown("""
        **S**ubjective: Client's reported experience  
        **O**bjective: Therapist's observations  
        **A**ssessment: Clinical interpretation  
        **P**lan: Treatment next steps
        """)
    
    if not api_key:
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started")
        return
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    if not llm:
        return
    
    # Main input form
    st.subheader("📝 Session Information")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        client_name = st.text_input("Client Name", placeholder="Enter client name")
    
    with col2:
        session_date = st.date_input("Session Date", value=datetime.now())
    
    with col3:
        duration = st.number_input("Duration (min)", min_value=15, max_value=180, value=50, step=5)
    
    # Conversation input
    st.subheader("💬 Therapy Session Conversation")
    st.markdown("*Enter the complete dialogue between therapist and client. Include speaker labels for best results.*")
    
    conversation = st.text_area(
        "Conversation Transcript",
        height=350,
        placeholder='Therapist: "How have you been feeling since our last session?"\nClient: "I\'ve been really anxious about work. My manager has been putting a lot of pressure on me..."\nTherapist: "Can you tell me more about what\'s been happening?"\nClient: "Well, there\'s this big project deadline coming up..."',
        help="Paste the full conversation here. The more detailed, the better the SOAP note."
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate SOAP Note", type="primary", use_container_width=True):
            if not all([client_name, conversation]):
                st.error("❌ Please fill in all required fields (Client Name and Conversation)")
            else:
                with st.spinner("🔄 Analyzing conversation and generating SOAP note..."):
                    soap_response = generate_soap_note(
                        llm, client_name, session_date.strftime("%Y-%m-%d"), duration, conversation
                    )
                    
                    if soap_response:
                        st.session_state.soap_note = parse_soap_response(soap_response)
                        st.success("✅ SOAP note generated successfully!")
                        st.rerun()
    
    # Display SOAP note
    if st.session_state.soap_note:
        soap_sections = st.session_state.soap_note
        metrics = soap_sections.get('metrics', {})
        
        st.divider()
        st.markdown("## 📊 Generated SOAP Note")
        
        # SUBJECTIVE
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📢 SUBJECTIVE (What Client Says)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('subjective_brief'):
            st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["subjective_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("subjective", "No data available")}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if metrics.get('emotional_distribution'):
                fig = create_subjective_chart(metrics['emotional_distribution'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if metrics.get('emotional_context'):
                st.markdown('<div class="note-label">Emotional Context</div>', unsafe_allow_html=True)
                emotional_context = metrics['emotional_context']
                context_html = '<div class="content-text">'
                for emotion, context in emotional_context.items():
                    context_html += f'<div class="context-box"><b>😟 {emotion.capitalize()}:</b> {context}</div>'
                context_html += '</div>'
                st.markdown(context_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # OBJECTIVE
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👁️ OBJECTIVE (What Therapist Observed)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('objective_brief'):
            st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["objective_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("objective", "No data available")}</div>', unsafe_allow_html=True)
        
        if metrics.get('observed_behaviors'):
            fig = create_objective_chart(metrics['observed_behaviors'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ASSESSMENT
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 ASSESSMENT (Clinical Opinion)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('assessment_brief'):
            st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["assessment_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("assessment", "No data available")}</div>', unsafe_allow_html=True)
        
        if metrics.get('assessment_domains'):
            fig = create_assessment_chart(metrics['assessment_domains'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # PLAN
        st.markdown('<div class="soap-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 PLAN (Treatment Next Steps)</div>', unsafe_allow_html=True)
        
        if soap_sections.get('plan_brief'):
            st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="brief-highlight">{soap_sections["plan_brief"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content-text">{soap_sections.get("plan", "No data available")}</div>', unsafe_allow_html=True)
        
        # Use plan_timeline from metrics if available
        plan_timeline = metrics.get('plan_timeline', [])
        fig = create_plan_chart(plan_timeline)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Session Metrics Summary
        st.divider()
        st.markdown("## 📈 Session Metrics Summary")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            risk_level = metrics.get('risk_level', 'moderate').upper()
            risk_color = {
                'LOW': '🟢',
                'MODERATE': '🟡',
                'HIGH': '🔴'
            }.get(risk_level, '⚪')
            st.metric("Risk Level", f"{risk_color} {risk_level}")
        
        with metric_col2:
            engagement = metrics.get('engagement_level', 0)
            st.metric("Engagement Level", f"{engagement}/10")
        
        with metric_col3:
            productivity = metrics.get('session_productivity', 0)
            st.metric("Session Productivity", f"{productivity}/10")
        
        with metric_col4:
            # Calculate total emotional intensity
            emotional_dist = metrics.get('emotional_distribution', {})
            negative_emotions = emotional_dist.get('anxious', 0) + emotional_dist.get('sad', 0) + emotional_dist.get('angry', 0)
            st.metric("Negative Affect", f"{negative_emotions}%")
        
        # Export Options
        st.divider()
        st.markdown("## 💾 Export Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Text export
            full_text = f"""CLINICAL SOAP NOTE
{'='*60}

CLIENT: {client_name}
SESSION DATE: {session_date.strftime("%Y-%m-%d")}
DURATION: {duration} minutes
RISK LEVEL: {metrics.get('risk_level', 'N/A').upper()}
ENGAGEMENT: {metrics.get('engagement_level', 'N/A')}/10
PRODUCTIVITY: {metrics.get('session_productivity', 'N/A')}/10

{'='*60}

SUBJECTIVE (What Client Says)
{'-'*60}
Brief: {soap_sections.get('subjective_brief', '')}

{soap_sections.get('subjective', '')}

{'='*60}

OBJECTIVE (What Therapist Observed)
{'-'*60}
Brief: {soap_sections.get('objective_brief', '')}

{soap_sections.get('objective', '')}

{'='*60}

ASSESSMENT (Clinical Opinion)
{'-'*60}
Brief: {soap_sections.get('assessment_brief', '')}

{soap_sections.get('assessment', '')}

{'='*60}

PLAN (Treatment Next Steps)
{'-'*60}
Brief: {soap_sections.get('plan_brief', '')}

{soap_sections.get('plan', '')}

{'='*60}
End of SOAP Note
"""
            
            st.download_button(
                label="📥 Download as TXT",
                data=full_text,
                file_name=f"SOAP_{client_name.replace(' ', '_')}_{session_date.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with export_col2:
            # PDF export
            pdf_buffer = create_pdf(soap_sections, client_name, session_date.strftime("%Y-%m-%d"), metrics)
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_buffer,
                file_name=f"SOAP_{client_name.replace(' ', '_')}_{session_date.strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with export_col3:
            # JSON export with all data
            json_data = {
                "client_name": client_name,
                "session_date": session_date.strftime("%Y-%m-%d"),
                "duration": duration,
                "soap_note": {
                    "subjective": {
                        "brief": soap_sections.get('subjective_brief', ''),
                        "detailed": soap_sections.get('subjective', '')
                    },
                    "objective": {
                        "brief": soap_sections.get('objective_brief', ''),
                        "detailed": soap_sections.get('objective', '')
                    },
                    "assessment": {
                        "brief": soap_sections.get('assessment_brief', ''),
                        "detailed": soap_sections.get('assessment', '')
                    },
                    "plan": {
                        "brief": soap_sections.get('plan_brief', ''),
                        "detailed": soap_sections.get('plan', '')
                    }
                },
                "metrics": metrics
            }
            
            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"SOAP_{client_name.replace(' ', '_')}_{session_date.strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Clear button
        st.divider()
        if st.button("🔄 Generate New SOAP Note", use_container_width=True):
            st.session_state.soap_note = None
            st.rerun()


if __name__ == "__main__":
    main()