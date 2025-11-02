import openai
import os
import pandas as pd

# Make sure your OpenAI API key is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are an empathetic and insightful Dream Analysis AI.
You analyze dream logs and identify psychological, emotional, and symbolic patterns.
You always ground your analysis in the provided dream data, not in imagination.
Provide insights in a warm, reflective tone.
"""

def generate_ai_response(prompt, df: pd.DataFrame):
    """
    Generate a grounded AI response based on dream journal data.
    """
    if df is None or df.empty:
        return "I need at least one dream entry to analyze. Please upload your journal first."

    # Summarize recent dreams as grounding context
    context = []
    for _, row in df.iterrows():
        date = str(row["date"])[:10]
        snippet = row["text"][:250].replace("\n", " ")
        context.append(f"- ({date}) {snippet}...")
    dreams_context = "\n".join(context)

    user_prompt = f"""
Dream journal context:
{dreams_context}

User request:
{prompt}
"""

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Or gpt-4-turbo / gpt-5 if available
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message["content"]

    except Exception as e:
        return f"Error during AI analysis: {e}"
