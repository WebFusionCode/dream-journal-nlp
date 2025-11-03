from openai import OpenAI
import os

def get_ai_response(prompt: str, context: str = ""):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing OpenAI API key. Please set it in your environment or Streamlit secrets."

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful dream interpretation assistant. Respond thoughtfully and insightfully."},
            {"role": "user", "content": f"Dream context:\n{context}\n\nUser question:\n{prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()
