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

# ==========================================
# 1. ADVANCED AI ENGINE (Multi-Factor)
# ==========================================
@st.cache_resource
def load_expert_model():
    """
    Trains a robust Random Forest model on 8 distinct risk factors.
    """
    np.random.seed(42)
    n = 5000
    
    # --- LEGAL & COMPLIANCE ---
    civil = np.random.randint(0, 50, n)
    criminal = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    
    # --- GOVERNANCE & BEHAVIOR ---
    control = np.random.uniform(0, 1, n) # Board Control (0=Weak, 1=Strong)
    risk_appetite = np.random.randint(1, 10, n)
    
    # --- FINANCIAL & CULTURE ---
    debt_ratio = np.random.uniform(0.1, 5.0, n) 
    employee_score = np.random.uniform(1.0, 5.0, n) 
    industry_risk = np.random.choice([1, 2, 3], size=n) 
    regulatory_pressure = np.random.choice([0, 1], size=n) 
    
    # --- SENTIMENT ---
    sentiment = np.random.uniform(-1.0, 1.0, n)

    # --- LOGIC ---
    risk_score = (
        (civil * 0.3) + (criminal * 40) + (risk_appetite * 2) + 
        (debt_ratio * 4) + (industry_risk * 5) + (regulatory_pressure * 5) - 
        (control * 10) - (employee_score * 5) - (sentiment * 10)
    )
    
    risk_score += np.random.normal(0, 5, n)
    threshold = np.percentile(risk_score, 85)
    is_fraud = (risk_score > threshold).astype(int)
    
    # DataFrame
    df = pd.DataFrame({
        'civil': civil, 'criminal': criminal, 'control': control, 
        'risk_appetite': risk_appetite, 'debt_ratio': debt_ratio, 
        'employee_score': employee_score, 'industry_risk': industry_risk, 
        'regulatory_pressure': regulatory_pressure, 'sentiment': sentiment, 
        'is_fraud': is_fraud
    })
    
    # Train
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    
    return model

model = load_expert_model()

# ==========================================
# 2. DATA FETCHING (With Dummy Protection)
# ==========================================

def is_dummy_search(name):
    dummies = ['test', 'dummy', 'sample', 'example', 'none', 'john doe']
    return any(d in name.lower() for d in dummies)

def get_live_data(name):
    if is_dummy_search(name):
        return {
            "bio": "SIMULATION MODE: No live data fetched.", "url": "#",
            "sentiment": 0.0, "subjectivity": 0.0, "headlines": [],
            "polarities": [], "is_live": False
        }

    try:
        page = wikipedia.page(name, auto_suggest=False)
        bio = page.summary[:500] + "..."
        url = page.url
    except:
        bio = "No Wikipedia page found."; url = "#"

    try:
        results = DDGS().text(f"{name} CEO controversy fraud news", max_results=15)
        headlines = [r['title'] for r in results]
    except:
        headlines = []

    if headlines:
        polarities = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_sent = np.mean(polarities)
    else:
        polarities = []
        avg_sent = 0.0
        
    return {
        "bio": bio, "url": url, "sentiment": avg_sent,
        "headlines": headlines, "polarities": polarities, "is_live": True
    }

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ExecuWatch 360", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .metric-container { background-color: #111; padding: 15px; border-radius: 8px; border: 1px solid #333; }
    .sim-banner { background-color: #FFA500; color: black; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("1. Profile Identity")
    name = st.text_input("Executive Name", "Test CEO")
    industry = st.selectbox("Industry Sector", ["Technology (Med Risk)", "Finance (High Risk)", "Manufacturing (Low Risk)", "Crypto (Extreme Risk)"])
    ind_val = {"Technology (Med Risk)": 2, "Finance (High Risk)": 3, "Manufacturing (Low Risk)": 1, "Crypto (Extreme Risk)": 4}[industry]

    st.subheader("2. Legal History")
    civil = st.slider("Civil Lawsuits", 0, 50, 5)
    criminal = st.checkbox("Criminal Record?", False)
    
    st.subheader("3. Financial Health")
    debt = st.slider("Debt-to-Equity Ratio", 0.0, 5.0, 1.5)
    
    st.subheader("4. Governance & Culture")
    control = st.slider("Board Independence (%)", 0, 100, 60) / 100.0
    culture = st.slider("Glassdoor Rating (1-5)", 1.0, 5.0, 3.8)
    
    st.subheader("5. Psychometrics")
    risk_appetite = st.slider("Risk Appetite (1-10)", 1, 10, 6)
    reg_pressure = 1 if st.radio("Regulatory Environment", ["Low", "High"]) == "High" else 0

    btn_analyze = st.button("RUN 360° ANALYSIS", type="primary")

# --- MAIN DASHBOARD ---
if btn_analyze:
    st.title(f"360° Risk Assessment: {name}")
    
    with st.spinner("Aggregating live data and manual variables..."):
        data = get_live_data(name)
        time.sleep(1)

    # PREDICT
    input_df = pd.DataFrame({
        'civil': [civil], 'criminal': [1 if criminal else 0], 
        'control': [control], 'risk_appetite': [risk_appetite],
        'debt_ratio': [debt], 'employee_score': [culture],
        'industry_risk': [ind_val], 'regulatory_pressure': [reg_pressure],
        'sentiment': [data['sentiment']]
    })
    
    prob = model.predict_proba(input_df)[0][1] * 100

    if not data['is_live']:
        st.markdown("<div class='sim-banner'>⚠️ SIMULATION MODE: Using manual variables only.</div>", unsafe_allow_html=True)

    # TOP METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Fraud Risk", f"{prob:.1f}%", delta="High" if prob > 50 else "Safe", delta_color="inverse")
    c2.metric("Financial Pressure", f"{debt} D/E", "Critcal" if debt > 3.0 else "Stable", delta_color="inverse")
    c3.metric("Live Sentiment", f"{data['sentiment']:.2f}", "News Polarity")
    c4.metric("Internal Culture", f"{culture}/5.0", "Employee Satisfaction")

    # --- TABS FOR GRAPHS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🕸️ 7-Point Radar", "📰 Live News & Sentiment", "📉 Trends & Benchmarks", "🧠 AI Logic"])

    with tab1:
        # RADAR CHART
        c_radar, c_desc = st.columns([2, 1])
        with c_radar:
            categories = ['Legal Hist.', 'Fin. Pressure', 'Governance', 'Cult. Toxicity', 'Media Sent.', 'Risk Appetite', 'Ind. Risk']
            r_vals = [
                min(civil/2, 10), min(debt*2, 10), (1-control)*10, 
                (5-culture)*2.5, (1-data['sentiment'])*5, risk_appetite, ind_val * 2.5
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name=name, line_color='#FF4B4B'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, title="Risk Dimensions")
            st.plotly_chart(fig, use_container_width=True)
        with c_desc:
            st.info("The wider the red shape, the higher the risk profile.")
            if debt > 3.0: st.warning("🔴 **Financial:** High Debt load.")
            if culture < 3.0: st.warning("🔴 **Culture:** Poor employee reviews.")

    with tab2:
        # NEWS & SENTIMENT HISTOGRAM
        if data['is_live']:
            c_news, c_hist = st.columns([1, 1])
            with c_news:
                st.subheader("Recent Headlines")
                for h in data['headlines'][:5]:
                    st.text(f"• {h}")
            with c_hist:
                st.subheader("Sentiment Distribution")
                if data['polarities']:
                    df_sent = pd.DataFrame({'Sentiment': data['polarities']})
                    fig_hist = px.histogram(df_sent, x="Sentiment", nbins=10, 
                                            title="News Polarity (Left=Neg, Right=Pos)",
                                            color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Live data disabled for dummy/test profiles.")

    with tab3:
        # TREND LINES & BENCHMARKING
        c_trend, c_bench = st.columns(2)
        
        with c_trend:
            st.subheader("12-Month Trust Trend (Simulated)")
            dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
            # Create trend that dips if risk is high
            trend_data = np.linspace(90, 100-prob, 12) + np.random.normal(0, 3, 12)
            fig_line = px.line(x=dates, y=trend_data, labels={'x': 'Date', 'y': 'Trust Index'}, markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
            
        with c_bench:
            st.subheader("Industry Benchmarks")
            comp_data = pd.DataFrame({
                "Metric": ["Fraud Probability", "Financial Leverage", "Cultural Score"],
                f"{name}": [prob, debt*20, culture*20], # scaled to 100
                "Industry Avg": [15, 40, 75]
            })
            fig_bar = px.bar(comp_data, x="Metric", y=[f"{name}", "Industry Avg"], barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        # FEATURE IMPORTANCE (Explainable AI)
        st.subheader("Why did the AI give this score?")
        importances = model.feature_importances_
        features = input_df.columns
        df_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Factor Contribution to Risk Model")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("👈 Configure variables in the Sidebar and click 'RUN 360° ANALYSIS'")
    st.markdown("**Test:** Enter 'Elon Musk' for live data, or 'Test User' for dummy mode.")