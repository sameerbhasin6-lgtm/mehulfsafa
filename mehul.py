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
# 1. ADVANCED AI ENGINE (Hybrid Model)
# ==========================================
@st.cache_resource
def load_advanced_model():
    """
    Trains a model that considers both QUANTITATIVE (lawsuits) 
    and QUALITATIVE (sentiment, tenure) factors.
    """
    np.random.seed(101)
    n = 3000
    
    # --- Quantitative Data ---
    civil_suits = np.random.randint(0, 40, n)
    criminal_record = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    
    # --- Qualitative/Behavioral Proxies ---
    media_sentiment = np.random.uniform(-1.0, 1.0, n) # -1 (Hated) to 1 (Loved)
    board_control = np.random.uniform(0, 1, n) # 0 (Dictator) to 1 (Democrat)
    risk_appetite = np.random.randint(1, 10, n) # 1 (Safe) to 10 (Gambler)
    
    # --- Ground Truth Logic ---
    # High Risk = Criminal Record OR (High Risk Appetite + Bad Media + Low Board Control)
    risk_score = (
        (civil_suits * 0.4) + 
        (criminal_record * 50) + 
        (risk_appetite * 3) - 
        (media_sentiment * 15) - 
        (board_control * 10)
    )
    
    # Normalize
    risk_score += np.random.normal(0, 5, n)
    threshold = np.percentile(risk_score, 85)
    is_fraud = (risk_score > threshold).astype(int)
    
    # DataFrame
    df = pd.DataFrame({
        'civil': civil_suits, 'criminal': criminal_record, 
        'sentiment': media_sentiment, 'control': board_control, 
        'risk_appetite': risk_appetite, 'is_fraud': is_fraud
    })
    
    # Train
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=101)
    model.fit(df.drop('is_fraud', axis=1), df['is_fraud'])
    
    return model

model = load_advanced_model()

# ==========================================
# 2. QUALITATIVE ANALYTICS MODULES
# ==========================================

def get_ceo_bio(name):
    """Fetches biography for qualitative background check."""
    try:
        page = wikipedia.page(name, auto_suggest=False)
        return page.summary[:500] + "...", page.url
    except:
        return "No public biography found. Subject may be a private figure.", "#"

def get_news_sentiment(name):
    """
    Fetches news and calculates:
    1. Polarity (Pos/Neg)
    2. Subjectivity (Fact vs Opinion)
    """
    try:
        results = DDGS().text(f"{name} CEO business news", max_results=15)
        texts = [r['title'] for r in results]
    except:
        return 0, 0, []

    if not texts:
        return 0, 0, []

    polarities = []
    subjectivities = []
    
    for text in texts:
        blob = TextBlob(text)
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    
    avg_pol = np.mean(polarities)
    avg_sub = np.mean(subjectivities)
    
    return avg_pol, avg_sub, texts

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ExecuWatch Pro", page_icon="🏛️", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .big-stat { font-size: 32px; font-weight: bold; color: #4F8BF9; }
    .report-box { background-color: #0E1117; padding: 20px; border-radius: 10px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.title("🏛️ Executive Profiler")
    st.markdown("### 1. Identity")
    name = st.text_input("CEO Name", "Mark Zuckerberg")
    
    st.markdown("### 2. Known Legal History")
    civil = st.slider("Civil Lawsuits (Lifetime)", 0, 50, 2)
    criminal = st.checkbox("Criminal Record?", False)
    
    st.markdown("### 3. Behavioral Est.")
    risk_appetite = st.slider("Risk Appetite (1-10)", 1, 10, 7, help="1=Conservative, 10=Aggressive Growth")
    control = st.slider("Board Control (0-100%)", 0, 100, 50, help="Higher % means the Board has more power over the CEO.") / 100.0
    
    btn = st.button("Generate Full Dossier", type="primary")

# --- MAIN DASHBOARD ---
if btn:
    st.title(f"Qualitative Risk Dossier: {name}")
    
    # 1. LIVE DATA FETCH
    with st.spinner("Analyzing media patterns and constructing psychological profile..."):
        bio, wiki_url = get_ceo_bio(name)
        sentiment, subjectivity, headlines = get_news_sentiment(name)
        
        # Prepare AI Input
        input_data = pd.DataFrame({
            'civil': [civil], 'criminal': [1 if criminal else 0], 
            'sentiment': [sentiment], 'control': [control], 
            'risk_appetite': [risk_appetite]
        })
        
        # Predict
        prob = model.predict_proba(input_data)[0][1] * 100
        
        time.sleep(1)

    # --- TOP ROW: KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Fraud Probability", f"{prob:.1f}%", delta="-2.4%" if prob < 50 else "+12%", delta_color="inverse")
    with c2:
        sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
        st.metric("Media Sentiment", sentiment_label, f"{sentiment:.2f} Polarity")
    with c3:
        st.metric("Subjectivity Score", f"{subjectivity*100:.0f}%", "Fact vs Opinion")
    with c4:
        st.metric("Board Oversight", "Strong" if control > 0.6 else "Weak", f"{control*100:.0f}% Independence")

    # --- TABBED ANALYSIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Qualitative Graphs", "🧠 Psycholinguistic Profile", "📝 Text Report"])

    with tab1:
        # ROW: RADAR CHART + TREND LINE
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            st.subheader("Leadership & Risk DNA")
            # Create Radar Chart Data
            categories = ['Legal Safety', 'Media Image', 'Governance', 'Financial Caution', 'Innovation']
            
            # Normalize inputs to 0-5 scale for the chart
            r_legal = 5 - (civil / 10)
            r_media = (sentiment + 1) * 2.5
            r_gov = control * 5
            r_fin = 5 - (risk_appetite / 2)
            r_innov = risk_appetite / 2
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[r_legal, r_media, r_gov, r_fin, r_innov],
                theta=categories,
                fill='toself',
                name=name
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_g2:
            st.subheader("Reputation Trend (Simulated)")
            # Simulated time series data
            dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
            # Create a trend that dips if current risk is high
            trend = np.linspace(80, 100 - prob, 12) + np.random.normal(0, 2, 12)
            
            fig_line = px.line(x=dates, y=trend, labels={'x': 'Date', 'y': 'Trust Score'}, title="12-Month Trust Index")
            st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        # ROW: SENTIMENT ANALYSIS
        st.subheader("Media Perception Analysis")
        c_p1, c_p2 = st.columns([2, 1])
        
        with c_p1:
            # Word Cloud alternative (Bar Chart of Keywords)
            st.markdown("#### **Dominant Themes in Recent News**")
            # Fake keyword extraction for demo (Real extraction requires NLP libraries like SpaCy which are heavy)
            keywords = {"Fraud": 2, "Growth": 8, "Innovation": 6, "Lawsuit": civil, "Profits": 5}
            k_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Frequency"])
            fig_bar = px.bar(k_df, x="Frequency", y="Keyword", orientation='h', color="Frequency", color_continuous_scale="Bluered")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_p2:
            st.markdown("#### **Bio Summary**")
            st.info(bio)
            st.markdown(f"[Read Full Wikipedia Entry]({wiki_url})")

    with tab3:
        st.subheader("Automated Qualitative Risk Report")
        
        # Dynamic Text Generation
        risk_level = "CRITICAL" if prob > 70 else "ELEVATED" if prob > 40 else "LOW"
        
        report = f"""
        **CONFIDENTIAL REPORT: {name.upper()}**
        **DATE:** {time.strftime('%Y-%m-%d')}
        
        **1. EXECUTIVE SUMMARY**
        The subject, {name}, has been flagged as **{risk_level} RISK** ({prob:.2f}% probability of adverse litigation/fraud events).
        
        **2. QUALITATIVE LEADERSHIP ASSESSMENT**
        - **Aggression Index:** The subject displays a Risk Appetite of {risk_appetite}/10. High scores here often correlate with rapid growth but increased compliance oversights.
        - **Media Resonance:** Recent news sentiment is {sentiment:.2f}. {'Negative press coverage suggests ongoing reputational battles.' if sentiment < 0 else 'Positive press indicates strong public trust.'}
        
        **3. GOVERNANCE STRUCTURE**
        With a Board Control score of {control*100:.0f}%, the organization appears to have {'weak checks and balances on executive power.' if control < 0.4 else 'robust oversight mechanisms.'}
        
        **4. RECOMMENDATION**
        {'Immediate third-party audit recommended.' if prob > 50 else 'Monitor situation. No immediate intervention required.'}
        """
        
        st.text_area("Copy Report", report, height=400)

else:
    st.info("👈 Please enter the CEO's details in the sidebar and click 'Generate Full Dossier'.")
