import traceback
import pandas as pd

# Core imports
from core.query_router import build_sql_with_fallback
from core.semantic_search import semantic_city_search
from core.smart_router import smart_route
from core.intent_classifier import classify_query_intent
from core.ml_explain import explain_ml_results
from core.lifestyle_rag import try_build_lifestyle_card
from core.ml_utils import load_trained_model, load_feature_data

# MLOps imports
from mlops.monitoring import run_monitoring
from mlops.retrain import retrain

print("🔍 Running Automated QA Test...\n")

def test(name, func):
    try:
        func()
        print(f"✅ PASS: {name}")
    except Exception as e:
        print(f"❌ FAIL: {name}")
        print("Error:", str(e))
        traceback.print_exc()
        print("---")

 
# 1. Test Model Loading
 
def test_model_loading():
    from mlops.registry import load_registry  # not needed if you want simple import
    reg = load_registry()
    assert "silhouette_score" in reg
    assert "version" in reg
    assert "model_path" in reg


test("Model Loading", test_model_loading)

 
# 2. Test SQL Generation
 
def test_sql_generation():
    sql = build_sql_with_fallback("cities with population > 1000000", use_gpt=False)
    assert "population" in sql.lower()

test("SQL Generation", test_sql_generation)

 
# 3. Test Semantic Search
 
def test_semantic_search():
    results = semantic_city_search("largest cities", top_k=5)
    assert len(results) > 0

test("Semantic Search", test_semantic_search)

 
# 4. Test Intent Classifier
 
def test_intent_classifier():
    mode, state = classify_query_intent("cluster all cities")
    assert mode in ["sql", "semantic", "hybrid"]

test("Intent Classifier", test_intent_classifier)

 
# 5. Test Smart Routing (ML)
 
def test_smart_router():
    mode, target = smart_route("best cities for families")
    assert mode in ["ml_family", "ml_young", "ml_retirement", "none"]

test("ML Smart Router", test_smart_router)

 
# 6. Test RAG Lifestyle Card
 
def test_rag():
    card = try_build_lifestyle_card("Tell me about Miami")
    assert card is None or "city" in card

test("RAG Lifestyle Card", test_rag)

 
# 7. Test ML Features Loaded
 
def test_features_load():
    df = load_feature_data()
    assert "city" in df.columns
    assert "lifestyle_score" in df.columns

test("Feature Data Load", test_features_load)

 
# 8. Test Monitoring (Dry Run)
 
def test_monitoring():
    report = run_monitoring()
    assert "drift_report" in report

test("MLOps Monitoring", test_monitoring)

 
# 9. Test Retraining (Dry Run)
 
def test_retraining():
    out = retrain()
    assert "status" in out

test("MLOps Retrain", test_retraining)

print("\n🎉 Automated QA Completed")
