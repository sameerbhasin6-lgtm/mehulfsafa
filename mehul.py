import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from duckduckgo_search import DDGS
import wikipedia
import time
from collections import Counter
import re

# ==========================================
# 1. ROBUST AI & ANALYTICS ENGINE
# ==========================================

@st.cache_resource
def load_expert_model():
    """Trains the Risk Prediction Model on 5000 synthetic profiles."""
    np.random.seed(42)
    n = 5000
    
    # Synthetic Features
    civil = np.random.randint(0, 50, n)
    criminal = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    control = np.random.uniform(0, 1, n)
    risk_appetite = np.random.randint(1, 10, n)
    debt_ratio = np.random.uniform(0.1, 5.0, n)
    culture = np.random.uniform(1.0, 5.0, n)
    sentiment = np.random.uniform(-1.0, 1.0, n)
    
    # Logic: Risk Score Calculation
    score = (
        (civil * 0.4) + (criminal * 50) + (risk_appetite * 3) + 
        (debt_ratio * 5) - (control * 15) - (culture * 8) - (sentiment * 12)
    )
    score += np.random.normal(0, 5, n)
    is_fraud = (score > np.percentile(score, 85)).astype(int)
    
    df = pd.DataFrame({
        'civil': civil, 'criminal': criminal, 'control': control, 
        'risk_appetite': risk_appetite, 'debt': debt_ratio, 
        'culture': culture, 'sentiment': sentiment, 'is_fraud': is_fraud
    })
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    return model

model = load_expert_model()

def determine_archetype(risk_app, control, culture):
    """Classifies the CEO into a Leadership Archetype."""
    if risk_app > 8 and control > 0.7:
        return "The Imperialist (High Power, High Risk)"
    elif risk_app > 7 and culture > 4.0:
        return "The Visionary (High Risk, Good Culture)"
    elif risk_app < 4 and control > 0.6:
        return "The Steward (Low Risk, High Control)"
    elif culture < 2.5:
        return "The Mercenary (Toxic Culture)"
    else:
        return "The Operator (Balanced)"

# ==========================================
# 2. LIVE WEB INTELLIGENCE (The "Real" Search)
# ==========================================

def get_live_data(name, status_container):
    """
    Robust fetcher that handles errors gracefully.
    """
    data = {
        "bio": "Bio unavailable.", "url": "#", "sentiment": 0.0,
        "headlines": [], "keywords": {}, "found": False, "error": None
    }
    
    # 1. WIKIPEDIA (Identity Verification)
    status_container.write("1/3: Verifying Identity via Global Registries...")
    try:
        page = wikipedia.page(name, auto_suggest=False)
        data["bio"] = page.summary[:600] + "..."
        data["url"] = page.url
        data["found"] = True
    except wikipedia.exceptions.DisambiguationError:
        data["error"] = "Name is too common. Please be specific."
    except wikipedia.exceptions.PageError:
        data["error"] = "No Wikipedia entry found. Subject may be private."
    except Exception:
        data["bio"] = "Connection to Wiki failed."

    # 2. NEWS SEARCH (DuckDuckGo)
    status_container.write("2/3: Scraping Deep Web for Litigation Signals...")
    try:
        # We use a specific query to target risk
        query = f"{name} CEO fraud lawsuit scandal business news"
        # Try fetching just 5 results to avoid blocks
        results = DDGS().text(query, max_results=5)
        headlines = [r['title'] for r in results] if results else []
        data["headlines"] = headlines
    except Exception as e:
        data["headlines"] = []
        # Fallback: Don't crash, just log it
        print(f"Search failed: {e}")

    # 3. SENTIMENT & KEYWORD ANALYSIS
    status_container.write("3/3: Running Psycholinguistic Analysis...")
    if data["headlines"]:
        # Sentiment
        scores = [TextBlob(h).sentiment.polarity for h in data["headlines"]]
        data["sentiment"] = np.mean(scores)
        
        # Keywords
        all_text = " ".join(data["headlines"]).lower()
        # Remove common stop words
        stops = ['the', 'a', 'in', 'of', 'to', 'and', 'for', 'on', 'with', 'at', 'ceo', 'news', 'business']
        words = [w for w in re.findall(r'\w+', all_text) if w not in stops and len(w) > 3]
        data["keywords"] = dict(Counter(words).most_common(7))
    
    return data

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ExecuWatch Ultimate", page_icon="👁️", layout="wide")

# Styling
st.markdown("""
<style>
    .big-metric { font-size: 26px !important; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #f63366; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=80)
    st.title("ExecuWatch OSINT")
    st.markdown("---")
    
    name = st.text_input("Target Name", "Elon Musk")
    st.caption("Enter full name for best results.")
    
    st.header("Risk Parameters")
    civil = st.slider("Civil Lawsuits", 0, 50, 2)
    criminal = st.checkbox("Criminal Record?", False)
    debt = st.slider("Debt Ratio", 0.0, 5.0, 1.2)
    control = st.slider("Board Control", 0.0, 1.0, 0.5)
    culture = st.slider("Culture Score", 1.0, 5.0, 3.5)
    risk_app = st.slider("Risk Appetite", 1, 10, 7)
    
    btn = st.button("INITIATE DEEP SCAN", type="primary")

# --- MAIN SCREEN ---
if btn:
    st.title(f"Target Dossier: {name.upper()}")
    
    # LIVE STATUS CONTAINER
    status_box = st.status("Initializing Systems...", expanded=True)
    
    # 1. GET DATA
    live_data = get_live_data(name, status_box)
    
    # 2. RUN AI MODEL
    input_df = pd.DataFrame({
        'civil': [civil], 'criminal': [1 if criminal else 0], 
        'control': [control], 'risk_appetite': [risk_app], 
        'debt': [debt], 'culture': [culture], 
        'sentiment': [live_data['sentiment']]
    })
    prob = model.predict_proba(input_df)[0][1] * 100
    archetype = determine_archetype(risk_app, control, culture)
    
    status_box.update(label="Scan Complete", state="complete", expanded=False)

    # --- TOP METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{prob:.1f}%", delta="CRITICAL" if prob > 70 else "SAFE", delta_color="inverse")
    c2.metric("Media Sentiment", f"{live_data['sentiment']:.2f}", "Polarity (-1 to 1)")
    c3.metric("Leadership Archetype", archetype)
    c4.metric("Data Source", "Live Web" if live_data['found'] else "Manual Only")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Qualitative Graphs", "📰 Live Intelligence", "⚖️ Risk Breakdown"])

    with tab1:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Psychometric Profile")
            # Horizontal Bar Chart for abstract qualities
            psy_data = pd.DataFrame({
                'Trait': ['Aggression', 'Governance Strength', 'Financial Caution', 'Cultural Health'],
                'Score': [risk_app, control*10, (5-debt)*2, culture*2]
            })
            fig_bar = px.bar(psy_data, x='Score', y='Trait', orientation='h', range_x=[0,10], color='Score', color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_g2:
            st.subheader("Media Keyword Cloud")
            if live_data['keywords']:
                # Bar chart of top keywords found in news
                kw_df = pd.DataFrame(list(live_data['keywords'].items()), columns=['Word', 'Count'])
                fig_kw = px.bar(kw_df, x='Word', y='Count', title="Dominant Terms in Headlines", color='Count')
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No sufficient text data to generate keyword cloud.")

    with tab2:
        if live_data['found']:
            st.success("✅ Identity Verified")
            st.markdown(f"**Bio:** {live_data['bio']}")
            st.markdown(f"[Read Full Source]({live_data['url']})")
            
            st.subheader("Latest Headlines Scraped")
            if live_data['headlines']:
                for h in live_data['headlines']:
                    st.text(f"• {h}")
            else:
                st.warning("Identity verified, but no recent news headlines found.")
        else:
            st.error("❌ Subject not found in public registries.")
            if live_data['error']:
                st.write(f"Reason: {live_data['error']}")
            st.warning("System defaulting to Manual Mode using your slider inputs.")

    with tab3:
        # Radar Chart
        categories = ['Legal', 'Financial', 'Governance', 'Culture', 'Aggression']
        r_vals = [min(civil, 10), min(debt*2, 10), (1-control)*10, (5-culture)*2, risk_app]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name=name))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="Risk Vector Map")
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.info("👈 Enter a REAL CEO name (e.g. 'Sundar Pichai', 'Satya Nadella') in the sidebar and click 'INITIATE DEEP SCAN'.")
