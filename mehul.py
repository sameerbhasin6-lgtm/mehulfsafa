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
# 1. ENTERPRISE AI RISK ENGINE
# ==========================================
@st.cache_resource
def load_enterprise_model():
    """
    Trains a high-fidelity model with 12 distinct risk factors.
    """
    np.random.seed(999)
    n = 8000
    
    # --- CORE RISK FACTORS ---
    civil = np.random.randint(0, 40, n)
    criminal = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    control = np.random.uniform(0, 1, n)
    
    # --- FINANCIAL METRICS ---
    debt_ratio = np.random.uniform(0.1, 4.0, n)
    insider_sell = np.random.uniform(0, 80, n) # % of holdings sold last 12m
    ebitda_margin = np.random.uniform(-10, 40, n)
    
    # --- BEHAVIORAL & SOCIAL ---
    tenure = np.random.randint(1, 25, n)
    social_activity = np.random.randint(1, 10, n) # 1=Ghost, 10=Influencer
    risk_appetite = np.random.randint(1, 10, n)
    culture_score = np.random.uniform(1.0, 5.0, n)
    
    # --- EXTERNAL SIGNALS ---
    sentiment = np.random.uniform(-1.0, 1.0, n)
    industry_risk = np.random.randint(1, 5, n)

    # --- FRAUD LOGIC ---
    # Fraud correlates with: High Insider Selling + Low Margins + High Debt + Bad Culture
    risk_score = (
        (civil * 0.2) + 
        (criminal * 60) + 
        (insider_sell * 0.5) + 
        (debt_ratio * 3) - 
        (ebitda_margin * 0.4) - 
        (control * 12) - 
        (culture_score * 8) + 
        (social_activity * 1.5) - 
        (tenure * 0.5)
    )
    
    # Normalize & Noise
    risk_score += np.random.normal(0, 5, n)
    threshold = np.percentile(risk_score, 88) # Top 12% are flagged
    is_fraud = (risk_score > threshold).astype(int)
    
    df = pd.DataFrame({
        'civil': civil, 'criminal': criminal, 'control': control, 
        'debt': debt_ratio, 'insider_sell': insider_sell, 'ebitda': ebitda_margin,
        'tenure': tenure, 'social': social_activity, 'risk_app': risk_appetite,
        'culture': culture_score, 'sentiment': sentiment, 'ind_risk': industry_risk,
        'is_fraud': is_fraud
    })
    
    model = RandomForestClassifier(n_estimators=250, max_depth=18, random_state=42)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    return model

model = load_enterprise_model()

# ==========================================
# 2. LIVE INTELLIGENCE FEED
# ==========================================
def get_live_data(name, status_box):
    data = {"bio": "Data Unavailable", "url": "#", "headlines": [], "sentiment": 0.0, "found": False}
    
    # 1. Wiki Search
    status_box.write(f"🔍 Scanning Global Registries for '{name}'...")
    try:
        page = wikipedia.page(name, auto_suggest=False)
        data['bio'] = page.summary[:500] + "..."
        data['url'] = page.url
        data['found'] = True
    except:
        data['error'] = "Wiki Not Found"

    # 2. News Scraper (Robust)
    status_box.write("📰 Aggregating Deep Web News Signals...")
    try:
        results = DDGS().text(f"{name} CEO lawsuit fraud investigation", max_results=8)
        data['headlines'] = [r['title'] for r in results]
    except Exception as e:
        print(e)

    # 3. Sentiment Analysis
    if data['headlines']:
        pols = [TextBlob(h).sentiment.polarity for h in data['headlines']]
        data['sentiment'] = np.mean(pols)
    
    return data

def generate_report(name, prob, archetype, factors):
    """Auto-generates a professional audit text."""
    risk_lvl = "CRITICAL" if prob > 70 else "ELEVATED" if prob > 40 else "LOW"
    
    txt = f"""
    **EXECUTIVE RISK AUDIT: {name.upper()}**
    **DATE:** {time.strftime('%Y-%m-%d')} | **RISK LEVEL:** {risk_lvl} ({prob:.1f}%)
    
    **1. LEADERSHIP DIAGNOSIS**
    The subject is classified as **"{archetype}"**. 
    {'High insider selling (>' + str(factors['insider_sell']) + '%) raises concerns of lack of confidence in future growth.' if factors['insider_sell'] > 20 else 'Insider trading activity is within normal range.'}
    
    **2. FINANCIAL & CULTURAL HEALTH**
    Debt-to-Equity is {factors['debt']}. {'Combined with low EBITDA margins, this creates pressure to manipulate earnings.' if factors['debt'] > 2.0 and factors['ebitda'] < 10 else 'Financial pressure appears manageable.'}
    Internal culture score is {factors['culture']}/5.0. {'Toxic environments often precede whistleblower events.' if factors['culture'] < 3.0 else 'Strong culture acts as a fraud deterrent.'}
    
    **3. RECOMMENDATION**
    {'Initiate forensic accounting review immediately.' if prob > 60 else 'Continue standard monitoring protocols.'}
    """
    return txt

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ExecuWatch Enterprise", page_icon="🏢", layout="wide")

# Custom CSS for "Report" feel
st.markdown("""
<style>
    .metric-card { background-color: #0E1117; border: 1px solid #262730; padding: 15px; border-radius: 5px; }
    .report-text { font-family: 'Courier New', monospace; font-size: 14px; color: #00FF41; background-color: #000; padding: 15px; }
    .red-flag { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎛️ Enterprise Controls")
    
    st.subheader("1. Subject Profile")
    name = st.text_input("Name", "Elon Musk")
    industry = st.selectbox("Sector", ["Tech", "Finance", "Pharma", "Energy"])
    ind_risk_map = {"Tech": 2, "Finance": 4, "Pharma": 3, "Energy": 3}
    
    st.subheader("2. Financial Indicators")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        debt = st.number_input("Debt/Equity", 0.0, 10.0, 1.2)
        insider = st.number_input("Insider Sell %", 0, 100, 5)
    with col_s2:
        ebitda = st.number_input("EBITDA %", -50, 50, 15)
        tenure = st.number_input("Tenure (Yrs)", 1, 40, 5)

    st.subheader("3. Behavioral & Legal")
    civil = st.slider("Civil Lawsuits", 0, 50, 3)
    criminal = st.checkbox("Criminal Record", False)
    social = st.slider("Social Media Activity", 1, 10, 8, help="1=Silent, 10=Tweet Storms")
    
    st.subheader("4. Governance")
    control = st.slider("Board Control (%)", 0, 100, 40) / 100.0
    culture = st.slider("Culture (Glassdoor)", 1.0, 5.0, 3.2)
    
    run_btn = st.button("GENERATE AUDIT REPORT", type="primary", use_container_width=True)

# --- MAIN PANEL ---
if run_btn:
    st.title(f"🏢 Risk Audit: {name}")
    
    # 1. LIVE DATA & AI PROCESSING
    status = st.status("Processing Intelligence Feed...", expanded=True)
    live_data = get_live_data(name, status)
    
    input_vector = pd.DataFrame({
        'civil': [civil], 'criminal': [1 if criminal else 0], 'control': [control],
        'debt': [debt], 'insider_sell': [insider], 'ebitda': [ebitda],
        'tenure': [tenure], 'social': [social], 'risk_app': [social], # Proxy
        'culture': [culture], 'sentiment': [live_data['sentiment']], 
        'ind_risk': [ind_risk_map[industry]]
    })
    
    prob = model.predict_proba(input_vector)[0][1] * 100
    
    # Determine Archetype
    if insider > 30 and social > 8: archetype = "The Opportunist (High Sell-off)"
    elif control < 0.4 and culture < 2.5: archetype = "The Autocrat (Toxic Control)"
    elif debt > 3.0: archetype = "The Gambler (High Leverage)"
    else: archetype = "The Steward (Balanced)"
    
    status.update(label="Audit Complete", state="complete", expanded=False)

    # 2. KPI METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{prob:.1f}%", delta="CRITICAL" if prob > 65 else "Stable", delta_color="inverse")
    c2.metric("Insider Confidence", f"-{insider}%", "Shares Sold (LTM)", delta_color="off")
    c3.metric("Public Sentiment", f"{live_data['sentiment']:.2f}", "News Polarity")
    c4.metric("Financial Health", f"{ebitda}%", "EBITDA Margin")

    # 3. DEEP DIVE TABS
    tab1, tab2, tab3 = st.tabs(["📈 Financial vs Behavioral", "📰 Live Intelligence", "📝 Final Report"])

    with tab1:
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.subheader("Risk Correlation Matrix")
            # Bubble Chart: X=Financial Health, Y=Culture, Size=Risk
            # We create dummy peers for context
            peers = pd.DataFrame({
                'CEO': [name, 'Competitor A', 'Competitor B', 'Competitor C'],
                'Fin_Health': [ebitda, 20, 10, 35],
                'Culture': [culture, 4.2, 2.5, 3.8],
                'Risk': [prob, 15, 60, 10],
                'Type': ['Subject', 'Peer', 'Peer', 'Peer']
            })
            fig_bubble = px.scatter(peers, x="Fin_Health", y="Culture", size="Risk", color="Type",
                                    title="Subject Position vs Industry Peers (Bubble Size = Fraud Risk)",
                                    labels={"Fin_Health": "EBITDA Margin (%)", "Culture": "Culture Score"},
                                    range_x=[-10, 50], range_y=[1, 5], text="CEO")
            st.plotly_chart(fig_bubble, use_container_width=True)
            
        with col_g2:
            st.subheader("Insider Activity Simulation")
            # Simulated stock drop if insider selling is high
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            stock = [100, 102, 98, 95, 90, 85] if insider > 20 else [100, 105, 110, 112, 115, 118]
            fig_line = px.line(x=months, y=stock, title="6-Month Stock Trajectory (Proj.)")
            st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        if live_data['headlines']:
            st.subheader("🚨 Red Flag Analysis")
            red_flags = ['fraud', 'investigation', 'scandal', 'lawsuit', 'probe', 'misconduct']
            
            for h in live_data['headlines']:
                # Highlight bad words
                highlighted = h
                for rf in red_flags:
                    if rf in h.lower():
                        highlighted = highlighted.replace(rf, f"<span class='red-flag'>{rf.upper()}</span>").replace(rf.title(), f"<span class='red-flag'>{rf.upper()}</span>")
                
                st.markdown(f"> {highlighted}", unsafe_allow_html=True)
        else:
            st.warning("No live news signals detected. Analysis based on manual financial/behavioral inputs.")

    with tab3:
        st.subheader("Official Risk Narrative")
        report_text = generate_report(name, prob, archetype, {'insider_sell': insider, 'debt': debt, 'ebitda': ebitda, 'culture': culture})
        st.markdown(report_text)
        
        st.download_button("Download Report PDF", report_text, file_name=f"{name}_Audit.txt")

else:
    st.info("👈 Configure the Enterprise Controls in the sidebar and click 'Generate Audit Report'.")
