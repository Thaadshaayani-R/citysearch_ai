#compare_cities.py

import re
import pandas as pd
import plotly.graph_objects as go

 
# 1. Extract Cities from Natural Language
 
def extract_multiple_cities(query: str, df_features: pd.DataFrame):
    """
    Detects 2+ cities mentioned in the user's question.
    Works for:
        - "Miami vs Dallas"
        - "compare Chicago and Austin"
        - "Is Miami better than Tampa and Orlando?"
        - "Dallas Miami comparison"
    """
    q = query.lower()

    possible_cities = df_features["city"].unique()
    found = []

    for city in possible_cities:
        if city.lower() in q:
            found.append(city)

    # Remove duplicates
    found = list(dict.fromkeys(found))

    if len(found) >= 2:
        return found

    return []


 
# 2. Build Comparison Table
 
def build_comparison_table(city_list, df_features):
    """
    Returns a DataFrame comparing the selected cities.
    Includes Basic + ML features (Option A)
    """
    subset = df_features[df_features["city"].isin(city_list)]

    cols = [
        "city",
        "state",
        "population",
        "median_age",
        "avg_household_size",
        "lifestyle_score",
        "lifestyle_rank",
        "opportunity_index",
        "youth_index",
        "family_index",
        "cluster_label",
    ]

    return subset[cols].reset_index(drop=True)


 
# 3. Radar Chart for Comparison
 
def build_radar_chart(df):
    """
    Radar comparison for Opportunity, Youth, Family Index.
    Works for 2 or more cities.
    """
    categories = ["opportunity_index", "youth_index", "family_index"]

    fig = go.Figure()

    for _, row in df.iterrows():
        values = [
            row["opportunity_index"],
            row["youth_index"],
            row["family_index"],
        ]
        values.append(values[0])  # close loop

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=["Opportunity", "Youth", "Family", "Opportunity"],
                fill="toself",
                name=row["city"],
            )
        )

    fig.update_layout(
        title="Lifestyle Profile Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


 
# 4. Generate AI Summary Prompt
 
def comparison_summary_prompt(df):
    """
    Generates a summary table for LLM.
    """
    table_md = df.to_markdown(index=False)

    prompt = f"""
Compare the following cities based on their population, age, household size,
lifestyle score, and indexes.

Table:
{table_md}

Write:
1. Key differences
2. Which city leads in what category
3. Short recommendation (2–3 lines)
"""

    return prompt
