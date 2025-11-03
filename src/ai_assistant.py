import streamlit as st
from openai import OpenAI

# Initialize OpenAI client securely using Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_ai_response(prompt: str, context: str = "") -> str:
    """Generate a natural AI response using GPT-4-mini."""
    try:
        full_prompt = (
            "You are a dream interpretation assistant. "
            "You analyze dream patterns, emotions, and symbols in a psychological but empathetic way.\n\n"
            f"Context from recent dreams:\n{context}\n\n"
            f"User question: {prompt}\n\n"
            "Provide a thoughtful and insightful interpretation."
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert dream analyst and psychologist."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.8,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error during AI analysis: {e}"
