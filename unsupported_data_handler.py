# unsupported_data_handler.py
import streamlit as st


def handle_unsupported_data_query(query: str, classification: dict, df=None):
    """
    Handle queries that ask for data we don't have.
    Shows a helpful message instead of wrong results.
    
    Args:
        query: Original user query
        classification: Classification result with 'missing_data' field
        df: Optional DataFrame of cities to show as alternative
    """
    
    missing_data = classification.get("missing_data", "this information")
    keyword = classification.get("keyword_matched", "")
    state = classification.get("state")
    
    # Title
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.15));
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 1.5rem;">⚠️</span>
            <span style="font-size: 1.25rem; font-weight: 600; color: #fbbf24;">
                Data Not Available
            </span>
        </div>
        <div style="color: #fef3c7; line-height: 1.6;">
            Our database doesn't currently include <strong>{missing_data}</strong>.
            <br><br>
            We only have demographic data: <strong>population</strong>, <strong>median age</strong>, 
            and <strong>household size</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show what we CAN do
    st.markdown("""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-weight: 600; color: #667eea; margin-bottom: 0.75rem;">
            💡 What You Can Ask Instead
        </div>
        <div style="color: #a0aec0; line-height: 1.8;">
            • "Top 10 cities in Texas" — by population<br>
            • "Cities with median age under 30" — younger cities<br>
            • "Compare Austin and Denver" — side-by-side stats<br>
            • "Life in Miami" — lifestyle profile (AI-generated)<br>
            • "Cities similar to Chicago" — semantic similarity
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggest external resources for the data they wanted
    resource_map = {
        "cost of living": "Try: <a href='https://www.numbeo.com/cost-of-living/' target='_blank' style='color: #667eea;'>Numbeo</a> or <a href='https://www.bestplaces.net/' target='_blank' style='color: #667eea;'>BestPlaces</a>",
        "crime/safety statistics": "Try: <a href='https://ucr.fbi.gov/' target='_blank' style='color: #667eea;'>FBI Crime Data</a> or <a href='https://www.neighborhoodscout.com/' target='_blank' style='color: #667eea;'>NeighborhoodScout</a>",
        "population growth rate": "Try: <a href='https://www.census.gov/' target='_blank' style='color: #667eea;'>US Census Bureau</a>",
        "unemployment rate": "Try: <a href='https://www.bls.gov/lau/' target='_blank' style='color: #667eea;'>Bureau of Labor Statistics</a>",
        "income/salary data": "Try: <a href='https://www.bls.gov/oes/' target='_blank' style='color: #667eea;'>BLS Wage Data</a> or <a href='https://datausa.io/' target='_blank' style='color: #667eea;'>DataUSA</a>",
        "weather/climate data": "Try: <a href='https://www.weather.gov/' target='_blank' style='color: #667eea;'>Weather.gov</a> or <a href='https://www.bestplaces.net/' target='_blank' style='color: #667eea;'>BestPlaces</a>",
        "education statistics": "Try: <a href='https://nces.ed.gov/' target='_blank' style='color: #667eea;'>NCES</a> or <a href='https://www.greatschools.org/' target='_blank' style='color: #667eea;'>GreatSchools</a>",
        "employment statistics": "Try: <a href='https://www.bls.gov/' target='_blank' style='color: #667eea;'>Bureau of Labor Statistics</a>",
    }
    
    if missing_data in resource_map:
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: #a0aec0;
        ">
            🔗 <strong>For {missing_data}:</strong><br>
            {resource_map[missing_data]}
        </div>
        """, unsafe_allow_html=True)
    
    # If state was mentioned, offer to show cities in that state
    if state:
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="color: #667eea; font-weight: 500;">
                📍 Would you like to see cities in {state} instead?
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show button to load cities
        if st.button(f"Show largest cities in {state}", type="primary"):
            st.session_state["redirect_query"] = f"Top 20 largest cities in {state}"
            st.rerun()
    
    # Show alternative query suggestions based on what they asked
    suggested_queries = _get_suggested_queries(query, missing_data, state)
    
    if suggested_queries:
        st.markdown("""
        <div style="margin-top: 1rem; margin-bottom: 0.5rem; color: #a0aec0; font-size: 0.9rem;">
            Try one of these instead:
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(suggested_queries))
        for i, suggested in enumerate(suggested_queries):
            with cols[i]:
                if st.button(suggested, key=f"suggest_{i}"):
                    st.session_state["search_query"] = suggested
                    st.rerun()


def _get_suggested_queries(original_query: str, missing_data: str, state: str = None) -> list:
    """Generate suggested alternative queries based on what user asked."""
    
    suggestions = []
    
    if state:
        suggestions.append(f"Top 10 cities in {state}")
        suggestions.append(f"Cities in {state} with youngest population")
    else:
        suggestions.append("Top 10 largest US cities")
        suggestions.append("Cities with median age under 30")
    
    # Add a lifestyle suggestion if they were asking about quality of life
    if missing_data in ["cost of living", "crime/safety statistics", "employment statistics"]:
        if state:
            suggestions.append(f"Life in the largest city in {state}")
        else:
            suggestions.append("Life in Austin")
    
    return suggestions[:3]  # Max 3 suggestions


 
# INTEGRATION INSTRUCTIONS
 

"""
HOW TO INTEGRATE:

1. Add this import to query_handlers.py:
   from unsupported_data_handler import handle_unsupported_data_query

2. In your main query routing (process_query or similar), add this check:

   def process_query(query):
       classification = classify_query_hybrid(query)
       
       # Handle unsupported data queries
       if classification.get("query_type") == "unsupported_data":
           handle_unsupported_data_query(query, classification)
           return
       
       # ... rest of your routing logic

3. That's it! Now queries like "cheap cities in California" will show 
   a helpful message instead of wrong data.
"""