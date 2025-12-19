# test_core.py
"""
Unit tests for CitySearch AI core functionality.

Run with: pytest tests/test_core.py -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


 
# TEST: Intent Classification
 
class TestIntentClassifier:
    """Tests for intent_classifier.py"""
    
    def test_sql_intent_population(self):
        """Test that population queries return SQL intent."""
        from core.intent_classifier import classify_query_intent
        
        queries = [
            "population of cities in Texas",
            "how many cities have population > 1000000",
            "top 10 largest cities",
            "cities with median age > 40",
        ]
        
        for q in queries:
            mode, state = classify_query_intent(q)
            assert mode == "sql", f"Expected 'sql' for query: {q}, got {mode}"
    
    def test_semantic_intent_lifestyle(self):
        """Test that lifestyle queries return semantic intent."""
        from core.intent_classifier import classify_query_intent
        
        queries = [
            "best cities for families",
            "cities with good nightlife",
            "family-friendly cities",
        ]
        
        for q in queries:
            mode, state = classify_query_intent(q)
            assert mode in ["semantic", "ml_family"], f"Expected semantic/ml for: {q}, got {mode}"
    
    def test_state_extraction(self):
        """Test that states are correctly extracted."""
        from core.intent_classifier import classify_query_intent
        
        mode, state = classify_query_intent("cities in Texas")
        assert state == "Texas", f"Expected 'Texas', got {state}"
        
        mode, state = classify_query_intent("population of California cities")
        assert state == "California", f"Expected 'California', got {state}"
    
    def test_ml_family_intent(self):
        """Test ML family intent detection."""
        from core.intent_classifier import classify_query_intent
        
        mode, _ = classify_query_intent("best cities for families")
        assert mode == "ml_family"
    
    def test_ml_young_intent(self):
        """Test ML young professionals intent detection."""
        from core.intent_classifier import classify_query_intent
        
        mode, _ = classify_query_intent("best cities for young professionals")
        assert mode == "ml_young"
    
    def test_ml_retirement_intent(self):
        """Test ML retirement intent detection."""
        from core.intent_classifier import classify_query_intent
        
        mode, _ = classify_query_intent("best cities for retirement")
        assert mode == "ml_retirement"


 
# TEST: SQL Builder (Security)
 
class TestSQLBuilder:
    """Tests for sql_builder.py security."""
    
    def test_sql_injection_prevention(self):
        """Test that SQL injection attempts are neutralized."""
        from core.sql_builder import build_safe_query
        
        # Attempt SQL injection
        malicious_queries = [
            "cities starting with 'A'; DROP TABLE cities;--",
            "cities containing ' OR '1'='1",
            "top 10 cities in Texas' OR 1=1--",
        ]
        
        for q in malicious_queries:
            result = build_safe_query(q)
            # The query should use parameters, not string interpolation
            assert "DROP" not in result.sql
            assert "OR 1=1" not in result.sql
            assert ":pattern" in result.sql or ":state" in result.sql or "TOP(:limit)" in result.sql
    
    def test_parameterized_state_query(self):
        """Test that state queries use parameters."""
        from core.sql_builder import build_state_query
        
        result = build_state_query("Texas", 10)
        
        assert ":state" in result.sql
        assert "Texas" not in result.sql  # Should be in params, not SQL
        assert result.params["state"] == "Texas"
    
    def test_limit_validation(self):
        """Test that limit is capped."""
        from core.sql_builder import extract_limit, MAX_LIMIT
        
        assert extract_limit("top 5 cities") == 5
        assert extract_limit("top 1000 cities") == MAX_LIMIT
        assert extract_limit("show me cities") == 10  # Default


 
# TEST: Cluster Definitions
 
class TestClusterDefinitions:
    """Tests for cluster_definitions.py"""
    
    def test_cluster_count(self):
        """Test that CLUSTER_COUNT matches actual definitions."""
        from core.cluster_definitions import CLUSTERS, CLUSTER_COUNT
        
        assert len(CLUSTERS) == CLUSTER_COUNT
        assert all(i in CLUSTERS for i in range(CLUSTER_COUNT))
    
    def test_cluster_info_structure(self):
        """Test that all clusters have required fields."""
        from core.cluster_definitions import CLUSTERS
        
        required_fields = ["name", "tagline", "summary", "best_for"]
        
        for cid, info in CLUSTERS.items():
            for field in required_fields:
                assert field in info, f"Cluster {cid} missing field: {field}"
    
    def test_explain_cluster_dict(self):
        """Test explain_cluster with return_dict=True."""
        from core.cluster_definitions import explain_cluster
        
        result = explain_cluster(cluster_id=0, return_dict=True)
        
        assert isinstance(result, dict)
        assert "cluster_id" in result
        assert "cluster_name" in result
        assert "detailed_summary" in result
    
    def test_explain_cluster_markdown(self):
        """Test explain_cluster with return_dict=False."""
        from core.cluster_definitions import explain_cluster
        
        result = explain_cluster(cluster_id=0, return_dict=False)
        
        assert isinstance(result, str)
        assert "Cluster 0" in result
    
    def test_invalid_cluster_id(self):
        """Test handling of invalid cluster ID."""
        from core.cluster_definitions import get_cluster_info
        
        result = get_cluster_info(999)
        
        assert "Unknown" in result["name"]


 
# TEST: Data Loader
 
class TestDataLoader:
    """Tests for data_loader.py constants and utilities."""
    
    def test_us_states_count(self):
        """Test that all 50 states are listed."""
        from core.data_loader import US_STATES
        
        assert len(US_STATES) == 50
    
    def test_is_valid_state(self):
        """Test state validation."""
        from core.data_loader import is_valid_state
        
        assert is_valid_state("Texas") == True
        assert is_valid_state("texas") == True
        assert is_valid_state("TEXAS") == True
        assert is_valid_state("Texass") == False
        assert is_valid_state("Canada") == False
    
    def test_normalize_state_name(self):
        """Test state name normalization."""
        from core.data_loader import normalize_state_name
        
        assert normalize_state_name("texas") == "Texas"
        assert normalize_state_name("NEW YORK") == "New York"
        assert normalize_state_name("north carolina") == "North Carolina"


 
# TEST: Fuzzy Matching
 
class TestFuzzyMatching:
    """Tests for fuzzy matching functionality."""
    
    def test_city_fuzzy_match(self):
        """Test fuzzy city name matching."""
        from difflib import get_close_matches
        
        cities = ["Dallas", "Houston", "Austin", "San Antonio"]
        cities_lower = [c.lower() for c in cities]
        
        # Exact match
        matches = get_close_matches("dallas", cities_lower, n=1, cutoff=0.8)
        assert matches == ["dallas"]
        
        # Typo match
        matches = get_close_matches("dalas", cities_lower, n=1, cutoff=0.7)
        assert matches == ["dallas"]
    
    def test_state_fuzzy_match(self):
        """Test fuzzy state name matching."""
        from difflib import get_close_matches
        
        states = ["Texas", "Tennessee", "Florida"]
        states_lower = [s.lower() for s in states]
        
        # Typo match
        matches = get_close_matches("texs", states_lower, n=1, cutoff=0.7)
        assert matches == ["texas"]


 
# FIXTURE: Mock Database (for integration tests)
 
@pytest.fixture
def mock_cities_df():
    """Create a mock cities DataFrame for testing."""
    import pandas as pd
    
    return pd.DataFrame({
        "city": ["Dallas", "Houston", "Austin", "Miami", "Orlando"],
        "state": ["Texas", "Texas", "Texas", "Florida", "Florida"],
        "population": [1300000, 2300000, 1000000, 450000, 300000],
        "median_age": [33.5, 34.0, 32.0, 40.0, 35.0],
        "avg_household_size": [2.5, 2.7, 2.3, 2.4, 2.6],
    })


class TestWithMockData:
    """Tests that use mock data."""
    
    def test_city_extraction(self, mock_cities_df):
        """Test city extraction from query."""
        cities = mock_cities_df["city"].str.lower().tolist()
        
        query = "what is life like in Dallas"
        query_lower = query.lower()
        
        found = [c for c in cities if c in query_lower]
        assert found == ["dallas"]
    
    def test_state_filtering(self, mock_cities_df):
        """Test filtering by state."""
        texas_cities = mock_cities_df[mock_cities_df["state"] == "Texas"]
        
        assert len(texas_cities) == 3
        assert "Dallas" in texas_cities["city"].values


 
# RUN TESTS
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
