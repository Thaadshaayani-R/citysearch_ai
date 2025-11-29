#ml_explain.py

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def explain_ml_results(query, df):
    """
    Generate a short AI explanation for ranking results.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Convert top results to text
    rows = df[["city", "state", "population", "median_age", "avg_household_size"]].head(10)
    rows_text = rows.to_string(index=False)

    prompt = f"""
    The user asked: {query}

    Here are the top-ranked cities based on the ML model. 
    Provide a short, professional summary explaining why these cities scored high.

    City data:
    {rows_text}

    Keep the answer short (4–6 sentences). Use simple language.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
