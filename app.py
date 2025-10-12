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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import json
import pandas as pd
import re
from collections import Counter


def analyze_conversation_emotions(conversation):
    """Analyze conversation to extract emotional distribution based on actual content with enhanced accuracy"""
    if not conversation:
        return {"neutral": 100}
    
    # Separate client lines for more accurate emotion analysis
    lines = conversation.split('\n')
    client_lines = [line for line in lines if line.lower().startswith('client:')]
    
    # If no clear client lines, analyze entire conversation
    if not client_lines:
        analysis_text = conversation.lower()
    else:
        # Focus on client statements for emotion analysis
        analysis_text = ' '.join(client_lines).lower()
    
    # Enhanced emotion keywords with weighted importance
    emotion_keywords = {
        'anxious': {
            'primary': ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'stressed', 'stress', 'panic', 'overwhelmed', 'fear', 'scared'],
            'secondary': ['tense', 'restless', 'uneasy', 'concern', 'concerned', 'frightened', 'terrified', 'apprehensive']
        },
        'sad': {
            'primary': ['sad', 'sadness', 'depressed', 'depression', 'down', 'upset', 'hopeless', 'empty', 'lonely'],
            'secondary': ['cry', 'crying', 'tears', 'grief', 'loss', 'hurt', 'pain', 'miserable', 'devastated', 'heartbroken']
        },
        'angry': {
            'primary': ['angry', 'anger', 'mad', 'frustrated', 'frustration', 'irritated', 'annoyed', 'furious'],
            'secondary': ['rage', 'pissed', 'resentful', 'bitter', 'hostile', 'infuriated', 'outraged']
        },
        'hopeful': {
            'primary': ['hopeful', 'hope', 'optimistic', 'positive', 'better', 'improving', 'progress', 'excited'],
            'secondary': ['good', 'confident', 'motivated', 'encouraged', 'looking forward', 'optimism', 'cheerful']
        },
        'neutral': {
            'primary': ['okay', 'fine', 'normal', 'same', 'usual', 'routine', 'alright'],
            'secondary': ['regular', 'typical', 'ordinary', 'standard', 'moderate']
        }
    }
    
    # Calculate weighted emotion scores
    emotion_scores = {}
    total_weighted_score = 0
    
    for emotion, keyword_groups in emotion_keywords.items():
        weighted_count = 0
        
        # Primary keywords (weight = 2)
        for keyword in keyword_groups['primary']:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, analysis_text))
            weighted_count += matches * 2
        
        # Secondary keywords (weight = 1) 
        for keyword in keyword_groups['secondary']:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, analysis_text))
            weighted_count += matches * 1
        
        emotion_scores[emotion] = weighted_count
        total_weighted_score += weighted_count
    
    # Analyze conversation context for additional emotion inference
    context_emotions = analyze_contextual_emotions(analysis_text)
    
    # Add context-based emotion scores
    for emotion, context_score in context_emotions.items():
        if emotion in emotion_scores:
            emotion_scores[emotion] += context_score
            total_weighted_score += context_score
    
    # If no emotional indicators found, infer from conversation tone
    if total_weighted_score == 0:
        tone_emotion = infer_emotional_tone(analysis_text)
        return tone_emotion
    
    # Convert to percentages with minimum thresholds
    emotion_percentages = {}
    for emotion, score in emotion_scores.items():
        if score > 0:
            percentage = round((score / total_weighted_score) * 100)
            if percentage >= 5:  # Minimum 5% threshold for inclusion
                emotion_percentages[emotion] = percentage
    
    # Ensure we have at least one emotion
    if not emotion_percentages:
        return {"neutral": 100}
    
    # Normalize to exactly 100%
    total_percentage = sum(emotion_percentages.values())
    if total_percentage != 100:
        largest_emotion = max(emotion_percentages, key=emotion_percentages.get)
        emotion_percentages[largest_emotion] += (100 - total_percentage)
    
    return emotion_percentages


def analyze_contextual_emotions(text):
    """Analyze conversation context to infer additional emotions"""
    context_indicators = {
        'anxious': ['what if', 'worried about', 'scared that', 'afraid of', 'nervous about', 'can\'t stop thinking'],
        'sad': ['feel empty', 'miss', 'lost', 'gone', 'died', 'ended', 'over', 'nothing matters'],
        'angry': ['hate', 'can\'t stand', 'so annoying', 'drives me crazy', 'fed up', 'had enough'],
        'hopeful': ['looking forward', 'excited about', 'can\'t wait', 'things are getting', 'starting to feel']
    }
    
    context_scores = {'anxious': 0, 'sad': 0, 'angry': 0, 'hopeful': 0, 'neutral': 0}
    
    for emotion, indicators in context_indicators.items():
        for indicator in indicators:
            if indicator in text:
                context_scores[emotion] += 1
    
    return context_scores


def infer_emotional_tone(text):
    """Infer emotional tone when no explicit emotion keywords are found"""
    # Analyze sentence structure and word choice
    words = text.split()
    
    if len(words) < 10:  # Very short responses might indicate withdrawal
        return {"neutral": 60, "sad": 40}
    
    # Look for question marks (might indicate anxiety/uncertainty)
    question_count = text.count('?')
    if question_count >= 3:
        return {"anxious": 70, "neutral": 30}
    
    # Look for negative language patterns
    negative_words = ['no', 'not', 'never', 'nothing', 'nobody', 'none', 'can\'t', 'won\'t', 'shouldn\'t']
    negative_count = sum(1 for word in negative_words if word in text)
    
    if negative_count >= 5:
        return {"sad": 60, "neutral": 40}
    
    # Default to neutral with slight variation
    return {"neutral": 100}
    
    return emotion_percentages


def analyze_observed_behaviors(conversation):
    """Generate clinical behavioral assessment scores (1-10) for four key areas based on conversation content"""
    if not conversation:
        return {
            "emotional_expression": 5,
            "mood_presentation": 5, 
            "engagement_level": 5,
            "thought_patterns": 5
        }
    
    lines = conversation.split('\n')
    text = conversation.lower()
    
    # Analyze conversation structure
    client_lines = [line for line in lines if line.lower().startswith('client:')]
    therapist_lines = [line for line in lines if line.lower().startswith('therapist:')]
    
    # Calculate conversation metrics
    total_client_words = sum(len(line.split()) for line in client_lines)
    avg_response_length = total_client_words / len(client_lines) if client_lines else 0
    
    clinical_assessment = {}
    
    # 1. EMOTIONAL EXPRESSION (AFFECT) - How patient shows emotions
    emotional_expression_score = 5  # Start at baseline
    
    # Positive indicators (increase score)
    positive_affect = ['expressive', 'animated', 'enthusiastic', 'laughing', 'smiling', 'excited']
    appropriate_affect = ['calm', 'relaxed', 'composed', 'appropriate', 'stable']
    
    # Negative indicators (decrease score) 
    flat_affect = ['flat', 'monotone', 'emotionless', 'blank', 'expressionless', 'dull']
    inappropriate_affect = ['laughing inappropriately', 'crying suddenly', 'angry outburst']
    
    for indicator in positive_affect + appropriate_affect:
        if indicator in text:
            emotional_expression_score += 1
    
    for indicator in flat_affect + inappropriate_affect:
        if indicator in text:
            emotional_expression_score -= 2
    
    # Assess based on emotional content richness
    emotion_words = ['feel', 'feeling', 'felt', 'emotion', 'emotional']
    emotion_mentions = sum(text.count(word) for word in emotion_words)
    if emotion_mentions >= 5:
        emotional_expression_score += 1
    elif emotion_mentions <= 1:
        emotional_expression_score -= 1
    
    clinical_assessment["emotional_expression"] = max(1, min(10, emotional_expression_score))
    
    # 2. MOOD PRESENTATION - Overall emotional state conveyed
    mood_score = 5  # Start at baseline
    
    # Mood indicators
    positive_mood = ['happy', 'hopeful', 'optimistic', 'good', 'better', 'improving', 'positive', 'cheerful']
    neutral_mood = ['okay', 'fine', 'alright', 'normal', 'stable']
    negative_mood = ['sad', 'depressed', 'anxious', 'worried', 'angry', 'frustrated', 'hopeless', 'upset']
    
    positive_count = sum(text.count(word) for word in positive_mood)
    negative_count = sum(text.count(word) for word in negative_mood)
    
    # Adjust score based on mood indicators
    mood_score += positive_count - negative_count
    
    # Assess overall tone
    if any(word in text for word in ['excited', 'happy', 'great', 'wonderful']):
        mood_score += 2
    elif any(word in text for word in ['terrible', 'awful', 'hopeless', 'devastating']):
        mood_score -= 3
    
    clinical_assessment["mood_presentation"] = max(1, min(10, mood_score))
    
    # 3. ENGAGEMENT/PARTICIPATION - Level of involvement in conversation
    engagement_score = 5  # Start at baseline
    
    # Engagement indicators
    high_engagement = ['yes', 'absolutely', 'definitely', 'i think', 'i believe', 'i understand', 'that makes sense']
    low_engagement = ['i don\'t know', 'maybe', 'i guess', 'whatever', 'sure', 'fine']
    
    high_count = sum(text.count(phrase) for phrase in high_engagement)
    low_count = sum(text.count(phrase) for phrase in low_engagement)
    
    engagement_score += high_count - low_count
    
    # Assess response quality
    if avg_response_length > 20:  # Detailed responses
        engagement_score += 2
    elif avg_response_length < 5:  # Very brief responses
        engagement_score -= 2
    
    # Participation ratio
    if len(client_lines) > 0 and len(therapist_lines) > 0:
        participation_ratio = len(client_lines) / len(therapist_lines)
        if participation_ratio > 0.8:  # Good participation
            engagement_score += 1
        elif participation_ratio < 0.3:  # Poor participation
            engagement_score -= 2
    
    clinical_assessment["engagement_level"] = max(1, min(10, engagement_score))
    
    # 4. THOUGHT PATTERNS - How patient speaks and thinks
    thought_patterns_score = 5  # Start at baseline
    
    # Clear thinking indicators
    clear_thinking = ['understand', 'realize', 'makes sense', 'logical', 'clear', 'rational']
    organized_speech = ['first', 'second', 'then', 'because', 'therefore', 'so']
    
    # Unclear thinking indicators
    confused_thinking = ['confused', 'don\'t understand', 'mixed up', 'scattered', 'all over the place']
    negative_patterns = ['always', 'never', 'everyone', 'nobody', 'everything', 'nothing']
    repetitive_patterns = ['but', 'however', 'and then', 'you know']
    
    clear_count = sum(text.count(word) for word in clear_thinking + organized_speech)
    unclear_count = sum(text.count(word) for word in confused_thinking)
    negative_count = sum(text.count(word) for word in negative_patterns)
    
    thought_patterns_score += clear_count - unclear_count - (negative_count // 2)
    
    # Assess coherence and organization
    questions_count = text.count('?')
    if questions_count > 10:  # Excessive questioning might indicate confusion
        thought_patterns_score -= 1
    
    # Look for repetitive or circular thinking
    if any(phrase in text for phrase in ['i keep thinking', 'over and over', 'can\'t stop']):
        thought_patterns_score -= 1
    
    # Positive thought restructuring
    if any(phrase in text for phrase in ['maybe i could', 'what if i try', 'another way']):
        thought_patterns_score += 1
    
    clinical_assessment["thought_patterns"] = max(1, min(10, thought_patterns_score))
    
    return clinical_assessment


def analyze_assessment_domains(conversation, emotional_distribution):
    """Calculate assessment domain scores based on conversation content"""
    if not conversation:
        return {"mood": 5, "anxiety": 5, "sleep": 5, "functioning": 5, "risk": 1}
    
    text = conversation.lower()
    
    # Domain-specific indicators
    domain_indicators = {
        'mood': {
            'negative': ['depressed', 'sad', 'down', 'hopeless', 'empty', 'crying', 'tears'],
            'positive': ['better', 'good mood', 'happy', 'positive', 'improving']
        },
        'anxiety': {
            'negative': ['anxious', 'worry', 'panic', 'stressed', 'nervous', 'overwhelmed', 'fear'],
            'positive': ['calm', 'relaxed', 'peaceful', 'less anxious']
        },
        'sleep': {
            'negative': ['can\'t sleep', 'insomnia', 'waking up', 'nightmares', 'tired', 'exhausted'],
            'positive': ['sleeping well', 'good sleep', 'rested']
        },
        'functioning': {
            'negative': ['can\'t work', 'struggling', 'difficult', 'problems at work', 'can\'t focus'],
            'positive': ['working well', 'productive', 'managing', 'doing better']
        },
        'risk': {
            'negative': ['hurt myself', 'suicide', 'end it all', 'no point', 'hopeless', 'worthless'],
            'positive': ['safe', 'no thoughts of harm', 'support system']
        }
    }
    
    domain_scores = {}
    
    for domain, indicators in domain_indicators.items():
        negative_count = 0
        positive_count = 0
        
        for phrase in indicators['negative']:
            negative_count += len(re.findall(re.escape(phrase), text))
        
        for phrase in indicators['positive']:
            positive_count += len(re.findall(re.escape(phrase), text))
        
        # Calculate score (0-10 scale)
        if negative_count == 0 and positive_count == 0:
            # No indicators - use emotional distribution to infer
            if domain == 'anxiety' and 'anxious' in emotional_distribution:
                score = min(10, int(emotional_distribution['anxious'] / 10))
            elif domain == 'mood' and 'sad' in emotional_distribution:
                score = min(10, int(emotional_distribution['sad'] / 10))
            else:
                score = 5  # Neutral/moderate
        else:
            # Calculate based on ratio of negative to positive indicators
            total_indicators = negative_count + positive_count
            negative_ratio = negative_count / total_indicators if total_indicators > 0 else 0
            score = int(negative_ratio * 10)
            score = max(1, min(10, score))  # Ensure score is between 1-10
        
        domain_scores[domain] = score
    
    return domain_scores


def calculate_risk_level(assessment_domains, conversation):
    """Determine risk level based on assessment and conversation content"""
    if not conversation:
        return "low"
    
    text = conversation.lower()
    
    # High risk indicators
    high_risk_phrases = ['hurt myself', 'kill myself', 'suicide', 'end it all', 'no point in living']
    moderate_risk_phrases = ['hopeless', 'worthless', 'can\'t go on', 'too much pain']
    
    high_risk_count = sum(len(re.findall(re.escape(phrase), text)) for phrase in high_risk_phrases)
    moderate_risk_count = sum(len(re.findall(re.escape(phrase), text)) for phrase in moderate_risk_phrases)
    
    # Check assessment scores
    risk_score = assessment_domains.get('risk', 1)
    mood_score = assessment_domains.get('mood', 5)
    
    if high_risk_count > 0 or risk_score >= 8:
        return "high"
    elif moderate_risk_count > 0 or risk_score >= 5 or mood_score >= 7:
        return "moderate"
    else:
        return "low"


def calculate_engagement_productivity(conversation):
    """Calculate engagement and productivity scores based on conversation quality"""
    if not conversation:
        return 5, 5
    
    text = conversation.lower()
    lines = conversation.split('\n')
    
    # Count client responses
    client_responses = [line for line in lines if line.lower().startswith('client:')]
    therapist_responses = [line for line in lines if line.lower().startswith('therapist:')]
    
    # Engagement indicators
    engagement_positive = ['yes', 'i think', 'i feel', 'i\'ve been', 'it\'s like', 'i understand']
    engagement_negative = ['i don\'t know', 'maybe', 'i guess', 'whatever', 'fine']
    
    positive_engagement = sum(len(re.findall(re.escape(phrase), text)) for phrase in engagement_positive)
    negative_engagement = sum(len(re.findall(re.escape(phrase), text)) for phrase in engagement_negative)
    
    # Calculate engagement (1-10)
    if len(client_responses) == 0:
        engagement_level = 1
    else:
        avg_response_length = sum(len(response) for response in client_responses) / len(client_responses)
        length_score = min(10, avg_response_length / 50)  # Normalize to 0-10
        interaction_score = min(10, positive_engagement - negative_engagement + 5)
        engagement_level = max(1, int((length_score + interaction_score) / 2))
    
    # Productivity based on conversation depth and therapeutic progress
    productivity_indicators = ['understand', 'realize', 'makes sense', 'i see', 'that helps', 'better now']
    productivity_score = sum(len(re.findall(re.escape(phrase), text)) for phrase in productivity_indicators)
    
    if len(therapist_responses) > 0 and len(client_responses) > 0:
        interaction_ratio = len(client_responses) / len(therapist_responses)
        productivity_level = min(10, max(1, int(productivity_score + interaction_ratio * 3)))
    else:
        productivity_level = 5
    
    return engagement_level, productivity_level


def extract_emotional_context(conversation, emotional_distribution):
    """Extract context for when different emotions appear in conversation - analyzes actual conversation content"""
    if not conversation or not emotional_distribution:
        return {}
    
    # Additional safety check for None values
    if emotional_distribution is None:
        return {}
    
    lines = conversation.split('\n')
    context_mapping = {}
    
    # Enhanced topic/context identification patterns
    topic_patterns = {
        'work_stress': ['work', 'job', 'boss', 'manager', 'colleague', 'office', 'project', 'deadline', 'meeting', 'career', 'promotion', 'workload', 'overtime', 'workplace'],
        'relationship_issues': ['husband', 'wife', 'partner', 'boyfriend', 'girlfriend', 'marriage', 'divorce', 'dating', 'breakup', 'relationship', 'spouse', 'romantic', 'intimacy'],
        'family_dynamics': ['mother', 'father', 'mom', 'dad', 'parent', 'child', 'son', 'daughter', 'brother', 'sister', 'family', 'kids', 'siblings', 'relatives'],
        'health_concerns': ['health', 'sick', 'doctor', 'hospital', 'medicine', 'pain', 'illness', 'surgery', 'treatment', 'diagnosis', 'medical', 'symptoms'],
        'financial_worries': ['money', 'financial', 'debt', 'bills', 'budget', 'salary', 'income', 'expensive', 'afford', 'loan', 'mortgage', 'rent', 'cost'],
        'social_situations': ['friends', 'social', 'party', 'gathering', 'isolation', 'lonely', 'crowd', 'people', 'public', 'social media', 'friendship'],
        'academic_pressure': ['school', 'college', 'university', 'exam', 'study', 'grades', 'homework', 'professor', 'student', 'assignment', 'test'],
        'personal_development': ['therapy', 'counseling', 'goals', 'progress', 'change', 'improvement', 'growth', 'development', 'self-help', 'meditation'],
        'past_experiences': ['childhood', 'past', 'memory', 'history', 'before', 'used to', 'growing up', 'years ago', 'when I was'],
        'future_uncertainty': ['future', 'tomorrow', 'next week', 'next month', 'what if', 'afraid of', 'scared about', 'uncertain', 'unknown', 'planning'],
        'daily_routine': ['morning', 'evening', 'daily', 'routine', 'schedule', 'home', 'house', 'chores', 'cooking', 'cleaning'],
        'sleep_issues': ['sleep', 'sleeping', 'insomnia', 'nightmares', 'dreams', 'tired', 'exhausted', 'bed', 'wake up', 'rest']
    }
    
    # Comprehensive emotion keywords
    emotion_keywords = {
        'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'overwhelmed', 'fear', 'scared', 'tense', 'restless', 'uneasy', 'concern', 'worry', 'anxiety'],
        'sad': ['sad', 'depressed', 'down', 'upset', 'cry', 'crying', 'tears', 'hopeless', 'empty', 'lonely', 'grief', 'loss', 'hurt', 'pain', 'miserable'],
        'angry': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed', 'rage', 'furious', 'pissed', 'resentful', 'bitter', 'hostile', 'infuriated'],
        'hopeful': ['hopeful', 'optimistic', 'positive', 'better', 'improving', 'progress', 'good', 'excited', 'confident', 'motivated', 'encouraged', 'looking forward']
    }
    
    # Analyze conversation for emotional contexts
    for emotion in emotional_distribution.keys():
        if emotion == 'neutral' or emotional_distribution.get(emotion, 0) <= 0:
            continue
            
        emotion_contexts = []
        keywords = emotion_keywords.get(emotion, [])
        
        # Find lines with emotional content and extract specific context
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip empty lines or non-client lines for more accuracy
            if not line_lower or line_lower.startswith('therapist:'):
                continue
            
            # Check if this line contains the emotion
            emotion_found = any(keyword in line_lower for keyword in keywords)
            
            if emotion_found:
                # Look for specific topics in this emotional line
                found_topics = []
                for topic, topic_words in topic_patterns.items():
                    if any(word in line_lower for word in topic_words):
                        found_topics.append(topic)
                
                if found_topics:
                    emotion_contexts.extend(found_topics)
                else:
                    # Try to extract context from the sentence structure
                    context_extracted = False
                    
                    # Look for "about", "with", "when", "because" patterns
                    context_indicators = ['about', 'with', 'when', 'because', 'regarding', 'concerning']
                    for indicator in context_indicators:
                        if indicator in line_lower:
                            try:
                                parts = line_lower.split(indicator, 1)
                                if len(parts) > 1:
                                    context_part = parts[1].strip()[:50]  # First 50 chars after indicator
                                    # Clean up the context
                                    context_part = context_part.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                                    if len(context_part.split()) >= 2:  # At least 2 words
                                        emotion_contexts.append(context_part.split()[:4])  # Take first 4 words
                                        context_extracted = True
                                        break
                            except:
                                continue
                    
                    # If no specific context found, look at the overall sentence topic
                    if not context_extracted:
                        # Check for subject matter in the sentence
                        if any(word in line_lower for word in ['my', 'i', 'me']):
                            # Personal context - try to identify what it's about
                            words = line_lower.split()
                            for i, word in enumerate(words):
                                if word in ['my', 'i', 'me'] and i < len(words) - 1:
                                    next_words = ' '.join(words[i+1:i+4])  # Get next 3 words
                                    emotion_contexts.append(next_words)
                                    break
        
        # Determine the most appropriate context for this emotion
        if emotion_contexts:
            # Flatten list if needed and count contexts
            flat_contexts = []
            for context in emotion_contexts:
                if isinstance(context, list):
                    flat_contexts.append(' '.join(context))
                else:
                    flat_contexts.append(str(context))
            
            if flat_contexts:
                context_counter = Counter(flat_contexts)
                most_common_context = context_counter.most_common(1)[0][0]
                
                # Convert topic codes to readable descriptions
                topic_descriptions = {
                    'work_stress': 'work-related pressures and challenges',
                    'relationship_issues': 'relationship concerns and interpersonal dynamics',
                    'family_dynamics': 'family interactions and responsibilities', 
                    'health_concerns': 'health issues and medical worries',
                    'financial_worries': 'financial stress and money concerns',
                    'social_situations': 'social interactions and peer relationships',
                    'academic_pressure': 'academic performance and educational goals',
                    'personal_development': 'personal growth and self-improvement efforts',
                    'past_experiences': 'past events and childhood memories',
                    'future_uncertainty': 'future planning and life uncertainties',
                    'daily_routine': 'daily life activities and routines',
                    'sleep_issues': 'sleep patterns and rest concerns'
                }
                
                if most_common_context in topic_descriptions:
                    context_mapping[emotion] = f"when discussing {topic_descriptions[most_common_context]}"
                else:
                    # Use extracted context directly, cleaning it up
                    clean_context = most_common_context.strip()
                    if len(clean_context) > 5:  # Only use if meaningful
                        context_mapping[emotion] = f"when discussing {clean_context}"
                    else:
                        context_mapping[emotion] = f"when talking about personal concerns"
            else:
                context_mapping[emotion] = f"during personal reflection"
        else:
            # More specific fallback contexts
            specific_defaults = {
                'anxious': 'when discussing uncertain or challenging situations',
                'sad': 'when reflecting on losses or disappointing experiences',
                'angry': 'when recalling frustrating or unfair situations',
                'hopeful': 'when talking about positive changes and future possibilities'
            }
            context_mapping[emotion] = specific_defaults.get(emotion, 'during meaningful conversation topics')
    
    return context_mapping


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
    .session-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .session-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
    }
    .session-number {
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        font-weight: bold;
    }
    .roadmap-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = None
if 'treatment_roadmap' not in st.session_state:
    st.session_state.treatment_roadmap = None


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
  "emotional_distribution": {{[Calculate exact percentages based on emotional language in conversation - must sum to 100]}},
  "emotional_context": {{
    [For each emotion with >0%]: "specific context from conversation where this emotion appeared"
  }},
  "observed_behaviors": {{
    [Only include behaviors explicitly mentioned or strongly implied in conversation with actual counts]
  }},
  "assessment_domains": {{
    "mood": [0-10 scale based on conversation content],
    "anxiety": [0-10 scale based on anxiety indicators in conversation],
    "sleep": [0-10 scale based on sleep mentions in conversation],
    "functioning": [0-10 scale based on daily functioning discussion],
    "risk": [0-10 scale based on risk factors mentioned]
  }},
  "risk_level": "[low/moderate/high based on actual conversation content]",
  "engagement_level": [1-10 based on client participation and response quality],
  "session_productivity": [1-10 based on therapeutic progress evident in conversation],
  "plan_timeline": [
    {{"Task": "Next Therapy Session", "Start": "{next_week}", "Finish": "{next_week}", "Resource": "Therapy"}},
    {{"Task": "[Specific intervention based on conversation]", "Start": "{tomorrow}", "Finish": "{two_weeks}", "Resource": "Self-Work"}},
    {{"Task": "[Follow-up action based on session content]", "Start": "{three_days}", "Finish": "{three_days}", "Resource": "Follow-Up"}},
    {{"Task": "[Assessment based on discussed needs]", "Start": "{six_weeks}", "Finish": "{six_weeks}", "Resource": "Assessment"}}
  ]
}}

CRITICAL REQUIREMENTS:
- Base ALL content on the actual conversation provided
- Use direct quotes from the client wherever possible
- Ensure emotional distribution percentages sum to exactly 100
- Make behavior counts realistic based on conversation content
- Avoid generic filler content - be specific and accurate
- Dates in plan_timeline should follow format YYYY-MM-DD"""


# Treatment Roadmap Prompt Template
ROADMAP_TEMPLATE = """You are an expert clinical psychologist specializing in treatment planning. Based on the client's information and therapy goals, create a comprehensive, structured treatment roadmap.

Client Information:
- Primary Issues: {primary_issues}
- Therapy Goals: {therapy_goals}
- Preferred Approach: {preferred_approach}
- Session Duration: {session_duration} minutes
- Severity Level: {severity_level}
- Additional Context: {additional_context}

Create a detailed treatment plan with the following:

1. Determine the appropriate number of sessions (typically 6-24 sessions based on severity)
2. For EACH session, provide:
   - Session number and objective/focus
   - Specific therapeutic techniques to be used
   - Expected outcomes or progress markers
   - Homework or between-session activities

Format your response EXACTLY as JSON:

{{
  "total_sessions": 12,
  "treatment_duration_weeks": 16,
  "primary_diagnosis": "Generalized Anxiety Disorder",
  "treatment_approach": "Cognitive Behavioral Therapy with Mindfulness Integration",
  "overall_goals": [
    "Reduce anxiety symptoms by 50%",
    "Develop effective coping strategies",
    "Improve daily functioning and sleep quality"
  ],
  "sessions": [
    {{
      "session_number": 1,
      "week": 1,
      "title": "Initial Assessment and Goal Setting",
      "objective": "Establish therapeutic alliance, complete comprehensive assessment, and collaboratively set treatment goals",
      "techniques": [
        "Clinical interview",
        "Symptom assessment scales (GAD-7, PHQ-9)",
        "Goal setting framework",
        "Psychoeducation on anxiety"
      ],
      "expected_outcomes": [
        "Client feels heard and understood",
        "Clear baseline established",
        "Shared understanding of treatment goals",
        "Client understands anxiety cycle"
      ],
      "homework": "Keep daily anxiety log noting triggers, intensity (0-10), and responses"
    }},
    {{
      "session_number": 2,
      "week": 2,
      "title": "Understanding Anxiety and Cognitive Model",
      "objective": "Educate client on cognitive model and identify thought patterns",
      "techniques": [
        "Psychoeducation on CBT model",
        "Thought record introduction",
        "Cognitive distortion identification",
        "Anxiety cycle mapping"
      ],
      "expected_outcomes": [
        "Client understands connection between thoughts, feelings, and behaviors",
        "Can identify automatic thoughts",
        "Recognizes common cognitive distortions"
      ],
      "homework": "Complete thought records for anxiety episodes; Read CBT handout"
    }}
  ],
  "milestones": [
    {{
      "session": 4,
      "milestone": "Client can independently identify and challenge negative thoughts"
    }},
    {{
      "session": 8,
      "milestone": "50% reduction in anxiety symptoms; improved sleep quality"
    }},
    {{
      "session": 12,
      "milestone": "Treatment goals achieved; relapse prevention plan in place"
    }}
  ],
  "success_criteria": [
    "GAD-7 score reduced from moderate/severe to mild range",
    "Client demonstrates mastery of at least 3 coping techniques",
    "Improved functioning in work and relationships",
    "Confident in managing future anxiety episodes"
  ]
}}

CRITICAL REQUIREMENTS:
- Be specific and evidence-based in technique selection
- Ensure logical progression from session to session
- Adapt session count to severity (mild: 6-8, moderate: 10-14, severe: 16-24)
- Include concrete, measurable outcomes
- Balance between structure and flexibility
- Ensure all JSON is valid and properly formatted"""


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


def enhance_metrics_with_analysis(sections, conversation):
    """Enhance parsed SOAP sections with calculated metrics from conversation analysis"""
    if not conversation:
        return sections
    
    try:
        # Calculate metrics from conversation
        emotional_distribution = analyze_conversation_emotions(conversation)
        observed_behaviors = analyze_observed_behaviors(conversation)
        assessment_domains = analyze_assessment_domains(conversation, emotional_distribution)
        risk_level = calculate_risk_level(assessment_domains, conversation)
        engagement_level, session_productivity = calculate_engagement_productivity(conversation)
        emotional_context = extract_emotional_context(conversation, emotional_distribution)
        
        # Update metrics if they're missing or seem generic
        if not sections.get('metrics'):
            sections['metrics'] = {}
        
        metrics = sections['metrics']
        
        # Override with calculated values for reliability
        metrics['emotional_distribution'] = emotional_distribution
        metrics['emotional_context'] = emotional_context
        metrics['observed_behaviors'] = observed_behaviors
        metrics['assessment_domains'] = assessment_domains
        metrics['risk_level'] = risk_level
        metrics['engagement_level'] = engagement_level
        metrics['session_productivity'] = session_productivity
        
        # Keep plan_timeline from LLM if it exists, otherwise add default
        if 'plan_timeline' not in metrics:
            metrics['plan_timeline'] = []
            
    except Exception as e:
        st.warning(f"Error enhancing metrics: {e}")
    
    return sections


def get_default_metrics():
    """Return minimal default metrics structure - should only be used as fallback"""
    return {
        "emotional_distribution": {"neutral": 100},
        "emotional_context": {
            "neutral": "during general session discussion"
        },
        "observed_behaviors": {
            "appropriate_affect": 1
        },
        "assessment_domains": {
            "mood": 5,
            "anxiety": 5,
            "sleep": 5,
            "functioning": 5,
            "risk": 1
        },
        "risk_level": "low",
        "engagement_level": 5,
        "session_productivity": 5,
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
    """Create clinical behavior assessment chart with 4 key areas"""
    # Default clinical assessment if no data provided
    if not observed_behaviors:
        observed_behaviors = {
            "emotional_expression": 6, 
            "mood_presentation": 7, 
            "engagement_level": 5, 
            "thought_patterns": 6
        }
    
    # Map clinical categories with descriptive labels
    clinical_categories = {
        "emotional_expression": "Emotional Expression (Affect)",
        "mood_presentation": "Mood Presentation", 
        "engagement_level": "Engagement/Participation",
        "thought_patterns": "Thought Patterns"
    }
    
    # Extract or map data to clinical categories
    clinical_data = {}
    for key, label in clinical_categories.items():
        if key in observed_behaviors:
            clinical_data[label] = observed_behaviors[key]
        else:
            # Map old behavior keys to new clinical categories if needed
            if key == "emotional_expression":
                clinical_data[label] = observed_behaviors.get("appropriate_affect", 
                                    observed_behaviors.get("emotional_expression", 
                                    observed_behaviors.get("crying", 5)))
            elif key == "mood_presentation":
                clinical_data[label] = observed_behaviors.get("mood_presentation", 
                                     observed_behaviors.get("withdrawn", 6))
            elif key == "engagement_level":
                clinical_data[label] = observed_behaviors.get("engagement_level",
                                     observed_behaviors.get("animated", 
                                     observed_behaviors.get("appropriate_affect", 5)))
            elif key == "thought_patterns":
                clinical_data[label] = observed_behaviors.get("thought_patterns",
                                     observed_behaviors.get("agitated", 6))
    
    categories = list(clinical_data.keys())
    scores = list(clinical_data.values())
    
    # Color coding based on clinical significance (1-10 scale)
    colors = []
    for score in scores:
        if score <= 3:
            colors.append('#ef4444')  # Red - Concerning
        elif score <= 5:
            colors.append('#f59e0b')  # Yellow - Moderate concern
        elif score <= 7:
            colors.append('#3b82f6')  # Blue - Normal range
        else:
            colors.append('#10b981')  # Green - Good/Positive
    
    # Create horizontal bar chart for better readability
    fig = go.Figure(data=[go.Bar(
        y=categories,
        x=scores,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{score}/10" for score in scores],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Arial Bold'),
        hovertemplate='<b>%{y}</b><br>Clinical Rating: %{x}/10<br><extra></extra>'
    )])
    
    fig.update_layout(
        title="Clinical Behavioral Assessment (1-10 Scale)",
        xaxis_title="Clinical Rating (1=Concerning, 10=Excellent)",
        yaxis_title="",
        height=400,
        margin=dict(l=200, r=40, t=60, b=40),
        xaxis=dict(range=[0, 10], tickmode='linear', dtick=1),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#f8fafc'
    )
    
    # Add reference lines for clinical ranges
    fig.add_vline(x=3, line_dash="dash", line_color="red", opacity=0.3, 
                  annotation_text="Concerning", annotation_position="top")
    fig.add_vline(x=7, line_dash="dash", line_color="green", opacity=0.3,
                  annotation_text="Good Range", annotation_position="top")
    
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


def create_plan_visual(plan_timeline):
    """Create simple next steps visualization with clean table format"""
    if not plan_timeline or len(plan_timeline) == 0:
        # Default timeline if none provided
        start_date = datetime.now()
        plan_timeline = [
            {"Task": "Next Therapy Session", "Start": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), "Resource": "Therapy"},
            {"Task": "Practice Coping Skills", "Start": (start_date + timedelta(days=1)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=14)).strftime("%Y-%m-%d"), "Resource": "Self-Work"},
            {"Task": "Check-in Call", "Start": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), "Resource": "Follow-Up"},
            {"Task": "Progress Assessment", "Start": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), "Resource": "Assessment"}
        ]
    
    # Sort by start date to show chronological order
    sorted_timeline = sorted(plan_timeline, key=lambda x: datetime.strptime(x['Start'], '%Y-%m-%d'))
    
    return sorted_timeline


def create_plan_chart(plan_timeline):
    """Create a simple, clean next steps timeline visualization"""
    if not plan_timeline or len(plan_timeline) == 0:
        # Default timeline if none provided
        start_date = datetime.now()
        plan_timeline = [
            {"Task": "Next Therapy Session", "Start": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), "Resource": "Therapy"},
            {"Task": "Practice Coping Skills", "Start": (start_date + timedelta(days=1)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=14)).strftime("%Y-%m-%d"), "Resource": "Self-Work"},
            {"Task": "Check-in Call", "Start": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), "Resource": "Follow-Up"},
            {"Task": "Progress Assessment", "Start": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), "Resource": "Assessment"}
        ]
    
    # Sort by start date to show chronological order
    sorted_timeline = sorted(plan_timeline, key=lambda x: datetime.strptime(x['Start'], '%Y-%m-%d'))
    
    # Create a simple horizontal timeline visualization
    tasks = []
    dates = []
    colors = []
    
    # Color mapping for different resource types
    color_map = {
        'Therapy': '#3b82f6',      # Blue
        'Self-Work': '#10b981',    # Green
        'Follow-Up': '#f59e0b',    # Yellow
        'Assessment': '#8b5cf6',   # Purple
        'Default': '#6b7280'       # Gray
    }
    
    for i, item in enumerate(sorted_timeline):
        task_name = item.get('Task', 'Unknown Task')
        start_date = item.get('Start', '')
        resource = item.get('Resource', 'Default')
        
        # Format date for display
        try:
            date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%b %d, %Y')
            days_from_now = (date_obj - datetime.now()).days
            
            if days_from_now == 0:
                date_display = f"{formatted_date} (Today)"
            elif days_from_now == 1:
                date_display = f"{formatted_date} (Tomorrow)"
            elif days_from_now > 0:
                date_display = f"{formatted_date} (in {days_from_now} days)"
            else:
                date_display = f"{formatted_date} ({abs(days_from_now)} days ago)"
                
        except ValueError:
            date_display = start_date
        
        tasks.append(f"{i+1}. {task_name}")
        dates.append(date_display)
        colors.append(color_map.get(resource, color_map['Default']))
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=tasks,
        x=[1] * len(tasks),  # All bars same length for visual consistency
        orientation='h',
        marker=dict(color=colors),
        text=dates,
        textposition='middle right',
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>%{text}<br><extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title={
            'text': "Treatment Plan - Next Steps Timeline",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e40af'}
        },
        xaxis=dict(
            visible=False,
            range=[0, 1.2]
        ),
        yaxis=dict(
            tickfont=dict(size=12, color='#374151'),
            autorange='reversed'  # Shows first item at top
        ),
        height=max(300, len(tasks) * 60),
        margin=dict(l=20, r=120, t=60, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        font=dict(family='Arial, sans-serif')
    )
    
    return fig


def display_plan_timeline(plan_timeline):
    """Display a clean, Streamlit-friendly timeline of next steps"""
    if not plan_timeline or len(plan_timeline) == 0:
        # Default timeline if none provided
        start_date = datetime.now()
        plan_timeline = [
            {"Task": "Next Therapy Session", "Start": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=7)).strftime("%Y-%m-%d"), "Resource": "Therapy"},
            {"Task": "Practice Coping Skills", "Start": (start_date + timedelta(days=1)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=14)).strftime("%Y-%m-%d"), "Resource": "Self-Work"},
            {"Task": "Check-in Call", "Start": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=3)).strftime("%Y-%m-%d"), "Resource": "Follow-Up"},
            {"Task": "Progress Assessment", "Start": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), 
             "Finish": (start_date + timedelta(days=42)).strftime("%Y-%m-%d"), "Resource": "Assessment"}
        ]
    
    # Sort by start date to show chronological order
    sorted_timeline = sorted(plan_timeline, key=lambda x: datetime.strptime(x['Start'], '%Y-%m-%d'))
    
    # Display modern timeline header
    st.markdown("### 📅 Suggested Next Steps")
    
    # Resource type configuration
    resource_config = {
        'Therapy': {'icon': '🩺', 'color': '#3b82f6', 'bg_color': '#eff6ff'},
        'Self-Work': {'icon': '📝', 'color': '#10b981', 'bg_color': '#f0fdf4'},
        'Follow-Up': {'icon': '📞', 'color': '#f59e0b', 'bg_color': '#fffbeb'},
        'Assessment': {'icon': '📋', 'color': '#8b5cf6', 'bg_color': '#faf5ff'},
        'Default': {'icon': '📌', 'color': '#6b7280', 'bg_color': '#f9fafb'}
    }
    
    # Create timeline using Streamlit columns and containers
    for i, item in enumerate(sorted_timeline):
        task_name = item.get('Task', 'Unknown Task')
        start_date_str = item.get('Start', '')
        resource = item.get('Resource', 'Default')
        
        # Get resource configuration
        config = resource_config.get(resource, resource_config['Default'])
        
        # Format date for display
        try:
            date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            today = datetime.now()
            days_until = (date_obj - today).days
            
            if days_until == 0:
                urgency_text = "🔥 **Today**"
                date_display = date_obj.strftime('Today • %b %d')
            elif days_until == 1:
                urgency_text = "⚡ **Tomorrow**"
                date_display = date_obj.strftime('Tomorrow • %b %d')
            elif days_until <= 7:
                urgency_text = f"📅 **In {days_until} days**"
                date_display = date_obj.strftime('%a, %b %d')
            else:
                urgency_text = f"📆 **In {days_until} days**"
                date_display = date_obj.strftime('%b %d, %Y')
                
        except ValueError:
            urgency_text = "📅"
            date_display = start_date_str
        
        # Create a container for each timeline item
        with st.container():
            col1, col2 = st.columns([1, 10])
            
            with col1:
                # Icon in a centered position
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    width: 50px;
                    height: 50px;
                    background: {config['color']};
                    border-radius: 50%;
                    font-size: 1.5rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                ">
                    {config['icon']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Card content using Streamlit components
                with st.container():
                    # Header row
                    header_col1, header_col2 = st.columns([3, 1])
                    
                    with header_col1:
                        st.markdown(f"**{task_name}**")
                        st.caption(f"📅 {date_display}")
                    
                    with header_col2:
                        # Priority and resource badges
                        st.markdown(f"""
                        <div style="text-align: right;">
                            <div style="
                                background: {config['color']};
                                color: white;
                                padding: 0.2rem 0.6rem;
                                border-radius: 15px;
                                font-size: 0.75rem;
                                font-weight: 600;
                                margin-bottom: 0.25rem;
                                display: inline-block;
                            ">
                                {resource}
                            </div>
                            <div style="
                                color: {config['color']};
                                font-size: 0.8rem;
                                font-weight: 600;
                            ">
                                {urgency_text.replace('**', '')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress indicator
                    progress_percent = ((i + 1) / len(sorted_timeline)) * 100
                    st.progress(progress_percent / 100, text=f"Step {i+1} of {len(sorted_timeline)}")
        
        # Add some spacing between items
        if i < len(sorted_timeline) - 1:
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)


def create_roadmap_gantt(roadmap_data):
    """Create Gantt chart for treatment roadmap"""
    sessions = roadmap_data.get('sessions', [])
    if not sessions:
        return None
    
    # Create timeline data
    timeline_data = []
    start_date = datetime.now()
    
    for session in sessions:
        week = session.get('week', session.get('session_number', 1))
        session_date = start_date + timedelta(weeks=week-1)
        
        timeline_data.append({
            'Task': f"Session {session['session_number']}: {session['title']}",
            'Start': session_date.strftime("%Y-%m-%d"),
            'Finish': (session_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            'Type': 'Session'
        })
    
    df = pd.DataFrame(timeline_data)
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Type",
        title=f"Treatment Roadmap Timeline ({roadmap_data.get('total_sessions', 'N/A')} Sessions)",
        color_discrete_map={'Session': '#8b5cf6'}
    )
    
    fig.update_layout(
        height=max(400, len(sessions) * 40),
        margin=dict(l=20, r=40, t=60, b=60),
        xaxis_title="Timeline",
        yaxis_title="",
        showlegend=False
    )
    
    fig.update_yaxes(categoryorder='total descending')
    
    return fig


def generate_soap_note(llm, client_name, session_date, duration, conversation):
    """Generate SOAP note using LangChain with dynamic dates and conversation analysis"""
    try:
        # Pre-analyze conversation to provide context to LLM
        emotional_distribution = analyze_conversation_emotions(conversation)
        observed_behaviors = analyze_observed_behaviors(conversation)
        assessment_domains = analyze_assessment_domains(conversation, emotional_distribution)
        risk_level = calculate_risk_level(assessment_domains, conversation)
        engagement_level, session_productivity = calculate_engagement_productivity(conversation)
        emotional_context = extract_emotional_context(conversation, emotional_distribution)
        
        # Calculate dynamic dates for the prompt
        session_dt = datetime.strptime(session_date, "%Y-%m-%d")
        dates = {
            "tomorrow": (session_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            "three_days": (session_dt + timedelta(days=3)).strftime("%Y-%m-%d"),
            "next_week": (session_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
            "two_weeks": (session_dt + timedelta(days=14)).strftime("%Y-%m-%d"),
            "six_weeks": (session_dt + timedelta(days=42)).strftime("%Y-%m-%d")
        }
        
        # Enhanced template with analysis context (properly escaped)
        analysis_context = f"""

CONVERSATION ANALYSIS CONTEXT:
- Emotional Distribution: {str(emotional_distribution).replace('{', '{{').replace('}', '}}')}
- Observed Behaviors: {str(observed_behaviors).replace('{', '{{').replace('}', '}}')}
- Assessment Domains: {str(assessment_domains).replace('{', '{{').replace('}', '}}')}
- Risk Level: {risk_level}
- Engagement Level: {engagement_level}/10
- Session Productivity: {session_productivity}/10
- Emotional Context: {str(emotional_context).replace('{', '{{').replace('}', '}}')}

Use this analysis to inform your SOAP note generation and ensure all metrics in the METRICS section are consistent with this conversation analysis."""
        
        enhanced_template = SOAP_TEMPLATE + analysis_context
        
        prompt = PromptTemplate(
            input_variables=["client_name", "session_date", "duration", "conversation", 
                           "tomorrow", "three_days", "next_week", "two_weeks", "six_weeks"],
            template=enhanced_template
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


def generate_treatment_roadmap(llm, primary_issues, therapy_goals, preferred_approach, 
                               session_duration, severity_level, additional_context):
    """Generate treatment roadmap using LangChain"""
    try:
        prompt = PromptTemplate(
            input_variables=["primary_issues", "therapy_goals", "preferred_approach", 
                           "session_duration", "severity_level", "additional_context"],
            template=ROADMAP_TEMPLATE
        )
        
        chain = prompt | llm
        
        result = chain.invoke({
            "primary_issues": primary_issues,
            "therapy_goals": therapy_goals,
            "preferred_approach": preferred_approach,
            "session_duration": session_duration,
            "severity_level": severity_level,
            "additional_context": additional_context
        })
        
        if hasattr(result, 'content'):
            response_text = result.content
        else:
            response_text = str(result)
        
        # Clean and parse JSON
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        roadmap_data = json.loads(response_text)
        
        return roadmap_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing roadmap JSON: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error generating roadmap: {str(e)}")
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


def create_roadmap_pdf(roadmap_data):
    """Generate PDF of treatment roadmap"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                fontSize=24, spaceAfter=20, textColor='#8b5cf6')
    story.append(Paragraph("TREATMENT ROADMAP", title_style))
    
    # Summary info
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11, spaceAfter=6)
    story.append(Paragraph(f"<b>Total Sessions:</b> {roadmap_data.get('total_sessions', 'N/A')}", info_style))
    story.append(Paragraph(f"<b>Duration:</b> {roadmap_data.get('treatment_duration_weeks', 'N/A')} weeks", info_style))
    story.append(Paragraph(f"<b>Primary Focus:</b> {roadmap_data.get('primary_diagnosis', 'N/A')}", info_style))
    story.append(Paragraph(f"<b>Approach:</b> {roadmap_data.get('treatment_approach', 'N/A')}", info_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Overall goals
    story.append(Paragraph("TREATMENT GOALS", styles['Heading2']))
    for goal in roadmap_data.get('overall_goals', []):
        story.append(Paragraph(f"• {goal}", styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Sessions
    story.append(Paragraph("SESSION-BY-SESSION PLAN", styles['Heading2']))
    
    for session in roadmap_data.get('sessions', []):
        story.append(Spacer(1, 0.2*inch))
        
        session_title = f"Session {session['session_number']}: {session['title']}"
        story.append(Paragraph(session_title, styles['Heading3']))
        
        story.append(Paragraph(f"<b>Week:</b> {session.get('week', 'N/A')}", info_style))
        story.append(Paragraph(f"<b>Objective:</b> {session['objective']}", styles['Normal']))
        
        story.append(Paragraph("<b>Techniques:</b>", info_style))
        for technique in session.get('techniques', []):
            story.append(Paragraph(f"  • {technique}", styles['Normal']))
        
        story.append(Paragraph("<b>Expected Outcomes:</b>", info_style))
        for outcome in session.get('expected_outcomes', []):
            story.append(Paragraph(f"  • {outcome}", styles['Normal']))
        
        if session.get('homework'):
            story.append(Paragraph(f"<b>Homework:</b> {session['homework']}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def render_soap_tab():
    """Render the SOAP Note generation tab"""
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
    
    return client_name, session_date, duration, conversation


def render_roadmap_tab(llm):
    """Render the Treatment Roadmap tab"""
    st.subheader("🗺️ Treatment Roadmap Generator")
    st.markdown("*Answer the following questions to generate a comprehensive treatment plan*")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        primary_issues = st.text_area(
            "Primary Issues/Concerns",
            height=120,
            placeholder="E.g., Generalized anxiety, panic attacks, work-related stress, relationship difficulties...",
            help="Describe the main issues the client is presenting with"
        )
        
        therapy_goals = st.text_area(
            "Therapy Goals",
            height=120,
            placeholder="E.g., Reduce anxiety symptoms, improve coping skills, better work-life balance...",
            help="What does the client hope to achieve through therapy?"
        )
        
        preferred_approach = st.selectbox(
            "Preferred Therapy Approach",
            [
                "Cognitive Behavioral Therapy (CBT)",
                "Dialectical Behavior Therapy (DBT)",
                "Acceptance and Commitment Therapy (ACT)",
                "Psychodynamic Therapy",
                "Mindfulness-Based Therapy",
                "Solution-Focused Brief Therapy",
                "Integrative/Eclectic Approach",
                "Other"
            ],
            help="Select the primary therapeutic modality"
        )
    
    with col2:
        severity_level = st.select_slider(
            "Symptom Severity Level",
            options=["Mild", "Moderate", "Severe", "Very Severe"],
            value="Moderate",
            help="Rate the overall severity of the client's presenting issues"
        )
        
        session_duration = st.selectbox(
            "Session Duration (minutes)",
            [30, 45, 50, 60, 75, 90],
            index=2,
            help="Typical length of each therapy session"
        )
        
        additional_context = st.text_area(
            "Additional Context (Optional)",
            height=120,
            placeholder="E.g., Previous therapy experience, medication, support system, cultural considerations...",
            help="Any other relevant information that might inform the treatment plan"
        )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Treatment Roadmap", type="primary", use_container_width=True):
            if not all([primary_issues, therapy_goals]):
                st.error("❌ Please fill in at least Primary Issues and Therapy Goals")
            else:
                with st.spinner("🔄 Creating personalized treatment roadmap..."):
                    roadmap = generate_treatment_roadmap(
                        llm,
                        primary_issues,
                        therapy_goals,
                        preferred_approach,
                        session_duration,
                        severity_level,
                        additional_context if additional_context else "None provided"
                    )
                    
                    if roadmap:
                        st.session_state.treatment_roadmap = roadmap
                        st.success("✅ Treatment roadmap generated successfully!")
                        st.rerun()
    
    # Display roadmap if it exists
    if st.session_state.treatment_roadmap:
        roadmap = st.session_state.treatment_roadmap
        
        st.divider()
        st.markdown("## 📋 Treatment Roadmap Overview")
        
        # Summary Metrics (always visible)
        st.markdown('<div class="roadmap-summary">', unsafe_allow_html=True)
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        with sum_col1:
            st.metric("Total Sessions", roadmap.get('total_sessions', 'N/A'))
        
        with sum_col2:
            st.metric("Duration", f"{roadmap.get('treatment_duration_weeks', 'N/A')} weeks")
        
        with sum_col3:
            st.metric("Approach", roadmap.get('treatment_approach', 'N/A').split()[0])
        
        with sum_col4:
            st.metric("Milestones", len(roadmap.get('milestones', [])))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Treatment Focus - Expandable
        with st.expander("🎯 **Treatment Focus**", expanded=True):
            focus_col1, focus_col2 = st.columns(2)
            
            with focus_col1:
                st.markdown(f"**Primary Focus:** {roadmap.get('primary_diagnosis', 'N/A')}")
            
            with focus_col2:
                st.markdown(f"**Therapeutic Approach:** {roadmap.get('treatment_approach', 'N/A')}")
        
        # Overall Goals - Expandable
        with st.expander("🎯 **Overall Treatment Goals**", expanded=True):
            for i, goal in enumerate(roadmap.get('overall_goals', []), 1):
                st.markdown(f"{i}. {goal}")
        
        # Timeline visualization - Expandable
        with st.expander("📅 **Treatment Timeline**", expanded=False):
            gantt_fig = create_roadmap_gantt(roadmap)
            if gantt_fig:
                st.plotly_chart(gantt_fig, use_container_width=True)
        
        # Session-by-Session Breakdown - Expandable with Modern Schedule UI
        with st.expander("📚 **Session-by-Session Breakdown**", expanded=False):
            
            # Create a modern weekly schedule-style layout
            st.markdown("### Treatment Schedule Overview")
            
            sessions = roadmap.get('sessions', [])
            
            # Group sessions by week for better organization
            sessions_by_week = {}
            for session in sessions:
                week = session.get('week', session.get('session_number', 1))
                if week not in sessions_by_week:
                    sessions_by_week[week] = []
                sessions_by_week[week].append(session)
            
            # Display sessions in a modern card-based weekly layout
            for week_num in sorted(sessions_by_week.keys()):
                week_sessions = sessions_by_week[week_num]
                
                # Week header
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1rem;
                    border-radius: 12px;
                    margin: 1rem 0;
                    text-align: center;
                    font-weight: 600;
                    font-size: 1.1rem;
                ">
                    📅 Week {week_num}
                </div>
                """, unsafe_allow_html=True)
                
                # Session cards for this week
                for session in week_sessions:
                    session_num = session.get('session_number', 1)
                    session_title = session.get('title', 'Session')
                    session_objective = session.get('objective', 'No objective specified')
                    
                    # Create modern session card using Streamlit components instead of complex HTML
                    with st.container():
                        # Session header with number badge
                        col1, col2 = st.columns([1, 10])
                        with col1:
                            st.markdown(f"""
                            <div style="
                                width: 40px;
                                height: 40px;
                                background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
                                border-radius: 50%;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-weight: bold;
                                font-size: 0.9rem;
                                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                                margin-top: 0.5rem;
                            ">
                                #{session_num}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="
                                background: white;
                                border: 1px solid #e5e7eb;
                                border-radius: 12px;
                                padding: 1.2rem;
                                margin: 0.5rem 0;
                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="
                                    margin: 0 0 0.5rem 0;
                                    font-size: 1.2rem;
                                    font-weight: 600;
                                    color: #1f2937;
                                ">{session_title}</h3>
                                <p style="
                                    margin: 0 0 1rem 0;
                                    color: #6b7280;
                                    font-size: 0.95rem;
                                ">{session_objective}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                        # Techniques section
                        techniques = session.get('techniques', [])
                        if techniques:
                            with col2:
                                st.markdown("""
                                <div style="
                                    background: #f8fafc;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    border-left: 4px solid #3b82f6;
                                    margin-top: 0.5rem;
                                ">
                                    <h4 style="
                                        margin: 0 0 0.5rem 0;
                                        font-size: 0.9rem;
                                        font-weight: 600;
                                        color: #3b82f6;
                                        text-transform: uppercase;
                                        letter-spacing: 0.5px;
                                    ">
                                        🔧 Therapeutic Techniques
                                    </h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for technique in techniques:
                                    st.write(f"• {technique}")
                    
                        # Expected outcomes section
                        outcomes = session.get('expected_outcomes', [])
                        if outcomes:
                            with col2:
                                st.markdown("""
                                <div style="
                                    background: #f0fdf4;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    border-left: 4px solid #10b981;
                                    margin-top: 0.5rem;
                                ">
                                    <h4 style="
                                        margin: 0 0 0.5rem 0;
                                        font-size: 0.9rem;
                                        font-weight: 600;
                                        color: #10b981;
                                        text-transform: uppercase;
                                        letter-spacing: 0.5px;
                                    ">
                                        ✅ Expected Outcomes
                                    </h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for outcome in outcomes:
                                    st.write(f"• {outcome}")
                    
                        # Homework section
                        homework = session.get('homework')
                        if homework:
                            with col2:
                                st.markdown(f"""
                                <div style="
                                    background: #fffbeb;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    border-left: 4px solid #f59e0b;
                                    margin-top: 0.5rem;
                                ">
                                    <h4 style="
                                        margin: 0 0 0.5rem 0;
                                        font-size: 0.9rem;
                                        font-weight: 600;
                                        color: #f59e0b;
                                        text-transform: uppercase;
                                        letter-spacing: 0.5px;
                                    ">
                                        📝 Homework & Activities
                                    </h4>
                                    <div style="
                                        color: #374151;
                                        font-size: 0.9rem;
                                        line-height: 1.4;
                                    ">
                                        {homework}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Add spacing after each session
                        st.markdown("---")
        
        # Milestones
        if roadmap.get('milestones'):
            st.markdown("### 🏆 Key Milestones")
            
            milestone_data = []
            for milestone in roadmap.get('milestones', []):
                milestone_data.append({
                    'Session': f"Session {milestone['session']}",
                    'Milestone': milestone['milestone']
                })
            
            if milestone_data:
                df_milestones = pd.DataFrame(milestone_data)
                st.table(df_milestones)
        
        # Success criteria
        if roadmap.get('success_criteria'):
            st.markdown("### ✨ Success Criteria")
            for i, criterion in enumerate(roadmap.get('success_criteria', []), 1):
                st.markdown(f"{i}. {criterion}")
        
        # Export options
        st.divider()
        st.markdown("## 💾 Export Roadmap")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # JSON export
            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(roadmap, indent=2),
                file_name=f"Treatment_Roadmap_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col2:
            # PDF export
            pdf_buffer = create_roadmap_pdf(roadmap)
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_buffer,
                file_name=f"Treatment_Roadmap_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        # Clear button
        st.divider()
        if st.button("🔄 Create New Roadmap", use_container_width=True):
            st.session_state.treatment_roadmap = None
            st.rerun()


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
        
        st.divider()
        st.markdown("### 🗺️ About Roadmap")
        st.markdown("""
        Generate comprehensive treatment plans with:
        - Session-by-session breakdown
        - Specific therapeutic techniques
        - Expected outcomes
        - Homework assignments
        """)
    
    if not api_key:
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started")
        return
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    if not llm:
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 SOAP Note", "🗺️ Roadmap"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        client_name, session_date, duration, conversation = render_soap_tab()
        
        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate SOAP Note", type="primary", use_container_width=True, key="generate_soap"):
                if not all([client_name, conversation]):
                    st.error("❌ Please fill in all required fields (Client Name and Conversation)")
                else:
                    with st.spinner("🔄 Analyzing conversation and generating SOAP note..."):
                        soap_response = generate_soap_note(
                            llm, client_name, session_date.strftime("%Y-%m-%d"), duration, conversation
                        )
                        
                        if soap_response:
                            parsed_soap = parse_soap_response(soap_response)
                            # Enhance with calculated metrics
                            enhanced_soap = enhance_metrics_with_analysis(parsed_soap, conversation)
                            st.session_state.soap_note = enhanced_soap
                            st.success("✅ SOAP note generated successfully with calculated metrics!")
                            st.rerun()
        
        # Display SOAP note
        if st.session_state.soap_note:
            soap_sections = st.session_state.soap_note
            metrics = soap_sections.get('metrics', {})
            
            st.divider()
            
            # 1. Session Metrics Summary (moved to top)
            st.markdown("## � Session Metrics Summary")
            
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
            
            st.divider()
            
            # 2. SOAP Note with Expandable Sections
            st.markdown("## 📊 Generated SOAP Note")
            
            # SUBJECTIVE - Expandable
            with st.expander("📢 **SUBJECTIVE** - What Client Says", expanded=True):
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
            
            # OBJECTIVE - Expandable
            with st.expander("👁️ **OBJECTIVE** - What Therapist Observed", expanded=True):
                if soap_sections.get('objective_brief'):
                    st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="brief-highlight">{soap_sections["objective_brief"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="content-text">{soap_sections.get("objective", "No data available")}</div>', unsafe_allow_html=True)
                
                if metrics.get('observed_behaviors'):
                    fig = create_objective_chart(metrics['observed_behaviors'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # ASSESSMENT - Expandable
            with st.expander("🧠 **ASSESSMENT** - Clinical Opinion", expanded=True):
                if soap_sections.get('assessment_brief'):
                    st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="brief-highlight">{soap_sections["assessment_brief"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="content-text">{soap_sections.get("assessment", "No data available")}</div>', unsafe_allow_html=True)
                
                if metrics.get('assessment_domains'):
                    fig = create_assessment_chart(metrics['assessment_domains'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # PLAN - Expandable
            with st.expander("📋 **PLAN** - Treatment Next Steps", expanded=True):
                if soap_sections.get('plan_brief'):
                    st.markdown('<div class="note-label">Key Summary</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="brief-highlight">{soap_sections["plan_brief"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="note-label">Detailed Notes</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="content-text">{soap_sections.get("plan", "No data available")}</div>', unsafe_allow_html=True)
                
                # Use plan_timeline from metrics if available
                plan_timeline = metrics.get('plan_timeline', [])
                if plan_timeline:
                    display_plan_timeline(plan_timeline)
                else:
                    st.info("💡 No specific timeline generated. Timeline will be created based on treatment plan.")
            
            # 4. Remarks Section for Therapist
            st.divider()
            st.markdown("## � Therapist Remarks")
            
            # Initialize remarks in session state if not exists
            if 'therapist_remarks' not in st.session_state:
                st.session_state.therapist_remarks = ""
            
            therapist_remarks = st.text_area(
                "Additional Notes & Observations",
                value=st.session_state.therapist_remarks,
                height=150,
                placeholder="Add any additional clinical observations, treatment considerations, or notes that weren't captured in the automated SOAP generation...",
                help="This section allows you to add manual observations, clinical insights, or any additional context relevant to the session.",
                key="remarks_input"
            )
            
            # Save remarks to session state
            if therapist_remarks != st.session_state.therapist_remarks:
                st.session_state.therapist_remarks = therapist_remarks
            
            if therapist_remarks:
                st.markdown('<div class="content-text">', unsafe_allow_html=True)
                st.markdown(f"**Therapist's Additional Notes:**\n\n{therapist_remarks}")
                st.markdown('</div>', unsafe_allow_html=True)
            
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        render_roadmap_tab(llm)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()