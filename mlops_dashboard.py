# mlops_dashboard.py

import json
import streamlit as st


def load_registry():
    """Load model registry from JSON file."""
    try:
        with open("mlops/registry.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def render_mlops_dashboard():
    """Render the complete MLOps dashboard."""
    
    # Hero header
    st.markdown(
        """
        <div class='hero-section'>
            <div class='hero-title'>MLOps Dashboard</div>
            <div class='hero-subtitle'>
                Monitor model health, drift, and retraining for the city clustering engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load registry
    registry = load_registry()
    
    if not registry:
        st.error("⚠️ No registry.json found in mlops/. Please create it first.")
        st.markdown("""
        ### How to create registry.json
        
        Create a file at `mlops/registry.json` with the following structure:
        
        ```json
        {
            "model_name": "city_clusters",
            "version": "1.0.0",
            "trained_on": "2024-01-01",
            "model_path": "models/city_clusters.pkl",
            "silhouette_score": 0.45,
            "num_cities": 500,
            "notes": "Initial model"
        }
        ```
        """)
        return
    
     
    # TOP METRICS ROW
     
    st.markdown("<div class='section-header'>📊 Model Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Name", 
            registry.get("model_name", "N/A"),
            help="Name of the current model"
        )
    
    with col2:
        st.metric(
            "Version", 
            registry.get("version", "N/A"),
            help="Current model version"
        )
    
    with col3:
        silhouette = registry.get("silhouette_score", 0)
        st.metric(
            "Silhouette Score", 
            f"{silhouette:.3f}",
            help="Clustering quality score (-1 to 1, higher is better)"
        )
    
    with col4:
        st.metric(
            "Num Cities", 
            registry.get("num_cities", 0),
            help="Number of cities in the model"
        )
    
     
    # TRAINING INFORMATION
     
    st.markdown("<div class='section-header'>🔧 Training Information</div>", unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.8rem; color: #667eea; margin-bottom: 0.5rem;">📅 Trained On</div>
            <div style="font-size: 1.1rem; font-weight: 600;">{registry.get("trained_on", "N/A")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.8rem; color: #667eea; margin-bottom: 0.5rem;">📁 Model Path</div>
            <div style="font-size: 1.1rem; font-weight: 600;">{registry.get("model_path", "N/A")}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Notes
    notes = registry.get("notes", "")
    if notes:
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.05);
            border-left: 3px solid #667eea;
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.8rem; color: #667eea; margin-bottom: 0.5rem;">📝 Notes</div>
            <div>{notes}</div>
        </div>
        """, unsafe_allow_html=True)
    
     
    # ACTION BUTTONS
     
    st.markdown("---")
    st.markdown("<div class='section-header'>⚡ Actions</div>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        monitor_clicked = st.button("🔍 Run Monitoring", use_container_width=True)
    
    with btn_col2:
        retrain_clicked = st.button("🔄 Run Retrain", use_container_width=True)
    
    with btn_col3:
        refresh_clicked = st.button("🔃 Refresh Registry", use_container_width=True)
    
     
    # MONITORING RESULTS
     
    if monitor_clicked:
        _run_monitoring()
    
     
    # RETRAIN RESULTS
     
    if retrain_clicked:
        _run_retrain()
    
     
    # REFRESH REGISTRY
     
    if refresh_clicked:
        st.rerun()
    
     
    # FULL REGISTRY VIEW
     
    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Full Registry</div>", unsafe_allow_html=True)
    
    with st.expander("View Raw Registry JSON", expanded=False):
        st.json(registry)
    
     
    # MODEL HEALTH INDICATORS
     
    _show_health_indicators(registry)


def _run_monitoring():
    """Run drift monitoring."""
    
    st.markdown("<div class='section-header'>📈 Drift Report</div>", unsafe_allow_html=True)
    
    try:
        from mlops.monitoring import run_monitoring
        
        with st.spinner("Running monitoring..."):
            result = run_monitoring()
        
        # Drift report
        st.markdown("#### Drift Analysis")
        st.json(result.get("drift_report", {}))
        
        # Retrain needed?
        retrain_needed = result.get("retrain_needed", False)
        if retrain_needed:
            st.warning("⚠️ Retraining Recommended: Data drift detected!")
        else:
            st.success("✅ Model is healthy. No retraining needed.")
        
        # New silhouette
        new_silhouette = result.get("new_silhouette", 0)
        st.metric("New Silhouette Score (temp)", f"{new_silhouette:.3f}")
        
    except ImportError:
        st.error("❌ Could not import mlops.monitoring module.")
        st.markdown("""
        Make sure you have the monitoring module at `mlops/monitoring.py` with a `run_monitoring()` function.
        """)
    except Exception as e:
        st.error(f"❌ Monitoring failed: {str(e)}")


def _run_retrain():
    """Run model retraining."""
    
    st.markdown("<div class='section-header'>🔄 Retrain Results</div>", unsafe_allow_html=True)
    
    try:
        from mlops.retrain import retrain
        
        with st.spinner("Retraining model... This may take a moment."):
            result = retrain()
        
        status = result.get("status", "unknown")
        
        if status == "accepted":
            st.success(f"""
            ✅ **New model accepted!**
            - Version: {result.get('new_version', 'N/A')}
            - New Silhouette: {result.get('new_silhouette', 0):.3f}
            """)
            
            # Show improvement
            old_score = result.get("old_silhouette", 0)
            new_score = result.get("new_silhouette", 0)
            improvement = new_score - old_score
            
            if improvement > 0:
                st.metric("Improvement", f"+{improvement:.3f}", delta=f"{improvement:.3f}")
            
        elif status == "rejected":
            st.warning(f"""
            ⚠️ **New model rejected** (worse performance)
            - Old Silhouette: {result.get('old_silhouette', 0):.3f}
            - New Silhouette: {result.get('new_silhouette', 0):.3f}
            """)
        else:
            st.info(f"Retrain status: {status}")
        
        # Reload and show updated registry
        st.markdown("#### Updated Registry")
        updated_registry = load_registry()
        if updated_registry:
            st.json(updated_registry)
        
    except ImportError:
        st.error("❌ Could not import mlops.retrain module.")
        st.markdown("""
        Make sure you have the retrain module at `mlops/retrain.py` with a `retrain()` function.
        """)
    except Exception as e:
        st.error(f"❌ Retraining failed: {str(e)}")


def _show_health_indicators(registry: dict):
    """Show model health indicators."""
    
    st.markdown("---")
    st.markdown("<div class='section-header'>🏥 Model Health</div>", unsafe_allow_html=True)
    
    silhouette = registry.get("silhouette_score", 0)
    num_cities = registry.get("num_cities", 0)
    
    # Health checks
    health_checks = []
    
    # Silhouette score check
    if silhouette >= 0.5:
        health_checks.append(("Clustering Quality", "Excellent", "🟢"))
    elif silhouette >= 0.3:
        health_checks.append(("Clustering Quality", "Good", "🟡"))
    elif silhouette >= 0.1:
        health_checks.append(("Clustering Quality", "Fair", "🟠"))
    else:
        health_checks.append(("Clustering Quality", "Poor", "🔴"))
    
    # Data coverage check
    if num_cities >= 500:
        health_checks.append(("Data Coverage", "Comprehensive", "🟢"))
    elif num_cities >= 200:
        health_checks.append(("Data Coverage", "Good", "🟡"))
    elif num_cities >= 50:
        health_checks.append(("Data Coverage", "Limited", "🟠"))
    else:
        health_checks.append(("Data Coverage", "Insufficient", "🔴"))
    
    # Model freshness check
    trained_on = registry.get("trained_on", "")
    if trained_on:
        try:
            from datetime import datetime
            trained_date = datetime.strptime(trained_on, "%Y-%m-%d")
            days_old = (datetime.now() - trained_date).days
            
            if days_old <= 30:
                health_checks.append(("Model Freshness", f"{days_old} days old", "🟢"))
            elif days_old <= 90:
                health_checks.append(("Model Freshness", f"{days_old} days old", "🟡"))
            else:
                health_checks.append(("Model Freshness", f"{days_old} days old", "🟠"))
        except ValueError:
            health_checks.append(("Model Freshness", "Unknown", "⚪"))
    
    # Display health checks
    for check_name, status, icon in health_checks:
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 8px;
            margin-bottom: 0.5rem;
        ">
            <span style="font-weight: 500;">{check_name}</span>
            <span>{icon} {status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall health score
    green_count = sum(1 for _, _, icon in health_checks if icon == "🟢")
    total_checks = len(health_checks)
    health_pct = (green_count / total_checks * 100) if total_checks > 0 else 0
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9;">Overall Health Score</div>
        <div style="font-size: 2.5rem; font-weight: 700;">{health_pct:.0f}%</div>
        <div style="font-size: 0.85rem; opacity: 0.8;">{green_count}/{total_checks} checks passed</div>
    </div>
    """, unsafe_allow_html=True)


def render_mlops_sidebar():
    """Render MLOps-specific sidebar content."""
    
    st.sidebar.markdown("### MLOps Quick Actions")
    
    if st.sidebar.button("📊 View Metrics", use_container_width=True):
        st.session_state["mlops_view"] = "metrics"
    
    if st.sidebar.button("📈 View Drift History", use_container_width=True):
        st.session_state["mlops_view"] = "drift"
    
    if st.sidebar.button("🔧 Model Settings", use_container_width=True):
        st.session_state["mlops_view"] = "settings"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #a0aec0;">
        <strong>MLOps Tips:</strong><br>
        • Run monitoring weekly<br>
        • Retrain when drift detected<br>
        • Check silhouette score > 0.3
    </div>
    """, unsafe_allow_html=True)
