# src/ai_assistant.py

from openai import OpenAI
import streamlit as st

# Initialize the OpenAI client (API key will come from Streamlit Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_ai_response(user_input: str) -> str:
    """
    Generate an AI response to user input using OpenAI's GPT model.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to 'gpt-4-turbo' or 'gpt-3.5-turbo'
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an insightful and empathetic AI dream interpreter. "
                        "When users ask about their dreams or feelings, respond thoughtfully, "
                        "mixing psychological insight with gentle curiosity. "
                        "Avoid giving predictions or medical advice."
                    ),
                },
                {"role": "user", "content": user_input},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Sorry, something went wrong: {str(e)}"
