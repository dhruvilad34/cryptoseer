import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CryptoSeer · Price Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# ENHANCED CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;900&display=swap');

/* ── Force sidebar ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
    display: block !important; visibility: visible !important;
    min-width: 19rem !important; transform: translateX(0) !important;
}

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #e8eaf0; }
#MainMenu, footer, header   { visibility: hidden; }

.stApp {
    background: #060b18;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,180,255,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(123,97,255,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(255,107,107,0.04) 0%, transparent 60%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #050d1a 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}
section[data-testid="stSidebar"] * { color: #b8cce0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #6a8aaa !important; font-size:0.75rem !important; letter-spacing:0.08em; text-transform:uppercase; }

/* ── Hero ── */
.hero-wrap {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(0,212,255,0.08);
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #00d4ff;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 3.8rem;
    line-height: 1;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 40%, #7b61ff 70%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-desc {
    font-size: 1rem;
    color: #4a6a8a;
    font-weight: 300;
    max-width: 600px;
}

/* ── Section headers ── */
.sec-hdr {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #00d4ff;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; margin-bottom: 0.5rem; }
.kpi-card {
    background: linear-gradient(135deg, rgba(13,31,60,0.9), rgba(10,22,48,0.9));
    border: 1px solid rgba(26,58,92,0.8);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
}
.kpi-card:hover { border-color: rgba(0,212,255,0.3); transform: translateY(-2px); }
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #3a5a7a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.kpi-val       { font-weight: 700; font-size: 1.5rem; color: #00d4ff; }
.kpi-val.green { color: #00ff99; }
.kpi-val.red   { color: #ff4466; }
.kpi-val.gold  { color: #ffcc44; }
.kpi-val.white { color: #e8eaf0; }

/* ── Prediction box ── */
.pred-outer {
    background: linear-gradient(135deg, #071830, #060f22);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 24px;
    padding: 0.3rem;
    margin-top: 0.5rem;
}
.pred-inner {
    background: linear-gradient(135deg, rgba(0,212,255,0.03), rgba(123,97,255,0.03));
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.pred-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #00d4ff;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 100px;
    display: inline-block;
    padding: 0.2rem 1rem;
    margin-bottom: 1.5rem;
}
.pred-row { display: flex; justify-content: center; align-items: center; gap: 3rem; flex-wrap: wrap; margin: 1rem 0 1.5rem 0; }
.pred-block { text-align: center; }
.pred-lbl   { font-size: 0.75rem; color: #3a5a7a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.pred-amt   { font-weight: 900; font-size: 2.6rem; color: #ffffff; line-height: 1; }
.pred-amt.muted { font-size: 1.8rem; color: #2a4a6a; }
.pred-arrow { font-size: 2rem; color: rgba(0,212,255,0.3); }
.pred-chg-up   { font-size: 1.4rem; font-weight: 700; color: #00ff99; }
.pred-chg-down { font-size: 1.4rem; font-weight: 700; color: #ff4466; }

/* ── How it works cards ── */
.how-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; }
.how-card {
    background: linear-gradient(135deg, rgba(13,31,60,0.6), rgba(10,22,48,0.6));
    border: 1px solid rgba(26,58,92,0.5);
    border-radius: 16px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    transition: all 0.3s;
}
.how-card:hover { border-color: rgba(0,212,255,0.25); background: rgba(13,31,60,0.9); }
.how-icon  { font-size: 2.2rem; margin-bottom: 0.8rem; }
.how-title { font-weight: 700; color: #c8d8e8; font-size: 1rem; margin-bottom: 0.3rem; }
.how-desc  { font-size: 0.78rem; color: #3a5a7a; }

/* ── Sidebar brand ── */
.sb-brand {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.4rem;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sb-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a4a6a !important;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.12)) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: #00d4ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(123,97,255,0.25)) !important;
    border-color: rgba(0,212,255,0.6) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(0,255,100,0.06) !important; border-left: 3px solid #00ff88 !important; border-radius: 10px !important; }
.stInfo    { background: rgba(0,212,255,0.06) !important; border-left: 3px solid #00d4ff !important; border-radius: 10px !important; }
.stWarning { background: rgba(255,200,0,0.06) !important; border-left: 3px solid #ffcc00 !important; border-radius: 10px !important; }

/* ── Disclaimer ── */
.disclaimer {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #1a3050;
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    letter-spacing: 0.05em;
    border-top: 1px solid rgba(26,48,80,0.4);
    margin-top: 2rem;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
COIN_EMOJI = {
    'BTC':'₿','ETH':'Ξ','BNB':'◈','ADA':'₳','SOL':'◎',
    'XRP':'✕','DOT':'●','DOGE':'Ð','AVAX':'▲','MATIC':'⬡',
    'LTC':'Ł','LINK':'⬡','UNI':'🦄','ATOM':'⚛','FIL':'⬡'
}

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain/(loss+1e-10)))

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def add_features(df):
    df = df.copy()
    df['MA_7']  = df['Close'].rolling(7).mean()
    df['MA_14'] = df['Close'].rolling(14).mean()
    df['MA_30'] = df['Close'].rolling(30).mean()
    df['MA_90'] = df['Close'].rolling(90).mean()
    df['Daily_Return']  = df['Close'].pct_change()*100
    df['Volatility_7']  = df['Daily_Return'].rolling(7).std()
    df['Volatility_14'] = df['Daily_Return'].rolling(14).std()
    df['Price_Range']   = df['High'] - df['Low']
    df['RSI_14']        = compute_rsi(df['Close'])
    for lag in [1,2,3,5,7]: df[f'Lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def dark_fig(fig, height=420):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(6,11,24,0.6)',
        height=height,
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family='Outfit, sans-serif', color='#8aabcc'),
        legend=dict(bgcolor='rgba(7,17,31,0.8)', bordercolor='rgba(0,212,255,0.15)',
                    borderwidth=1, font=dict(size=11))
    )
    fig.update_xaxes(gridcolor='rgba(26,58,92,0.4)', zeroline=False,
                     tickfont=dict(size=10), linecolor='rgba(26,58,92,0.6)')
    fig.update_yaxes(gridcolor='rgba(26,58,92,0.4)', zeroline=False,
                     tickfont=dict(size=10), linecolor='rgba(26,58,92,0.6)')
    return fig

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">🔮 CryptoSeer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tagline">ML PRICE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload Dataset (CSV)", type=['csv'],
                                help="Upload crypto50_combined.csv from Kaggle")
    st.markdown("---")
    st.markdown("**⚙️ Configuration**")

    all_coins = ['BTC','ETH','BNB','ADA','SOL','XRP','DOT','DOGE','AVAX','MATIC','LTC','LINK']

    app_mode = st.radio("📌 Mode", ["🔮 Single Coin", "📊 Multi-Coin Compare"], index=0)
    st.markdown("")

    if "Single" in app_mode:
        coin = st.selectbox("🪙 Select Coin", all_coins, index=0)
        compare_coins = []
    else:
        coin = None
        compare_coins = st.multiselect(
            "🪙 Select Coins to Compare",
            all_coins,
            default=['BTC','ETH','SOL'],
            help="Pick 2–5 coins for best results"
        )

    model_choice = st.selectbox("🤖 Prediction Model", [
        'Random Forest ⚡ (Fast)',
        'Gradient Boosting 🎯 (Accurate)',
        'Linear Regression 📐 (Simple)'
    ])

    st.markdown("---")
    run_btn = st.button(
        "🚀  Run Prediction" if "Single" in app_mode else "📊  Compare Coins",
        use_container_width=True
    )
    st.markdown("---")

    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#1a3050;line-height:1.6'>
    ⚠️ For educational use only.<br>
    Not financial advice.<br>
    Crypto markets are volatile.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🔮 AI-Powered · Real Data · 2014–2026</div>
    <div class="hero-title">CryptoSeer</div>
    <div class="hero-desc">
        Machine learning price prediction for the top 50 cryptocurrencies.
        Upload your dataset, pick a coin, and get tomorrow's forecast in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# NO FILE — LANDING
# ─────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
    border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.5rem;
    font-family:Space Mono,monospace;font-size:0.8rem;color:#4a8aaa;'>
    👈 &nbsp; Upload <strong style="color:#00d4ff">crypto50_combined.csv</strong> in the sidebar to begin
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="how-icon">📂</div>
            <div class="how-title">Upload Data</div>
            <div class="how-desc">Drop in the Kaggle CSV with 10+ years of crypto history</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🧮</div>
            <div class="how-title">Auto Features</div>
            <div class="how-desc">RSI · Moving Averages · Volatility · Lag features</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🤖</div>
            <div class="how-title">Train Model</div>
            <div class="how-desc">Random Forest or Gradient Boosting trained in seconds</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🔮</div>
            <div class="how-title">Get Forecast</div>
            <div class="how-desc">Tomorrow's predicted price with expected % change</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# LOAD & FILTER
# ─────────────────────────────────────────────────────────────────
df_raw = load_data(uploaded)

if 'Symbol' not in df_raw.columns:
    st.error("❌ 'Symbol' column not found. Please check your CSV.")
    st.stop()

available = sorted(df_raw['Symbol'].unique())

# ─────────────────────────────────────────────────────────────────
# MULTI-COIN COMPARISON MODE
# ─────────────────────────────────────────────────────────────────
if "Multi" in app_mode:
    if len(compare_coins) < 2:
        st.warning("👈 Please select at least 2 coins in the sidebar to compare!")
        st.stop()

    COIN_COLORS = ['#00d4ff','#ff6b6b','#00ff99','#ffd700','#7b61ff','#ff9944']
    FEATS = ['Open','High','Low','Volume','MA_7','MA_14','MA_30',
             'RSI_14','Daily_Return','Volatility_7','Price_Range',
             'Lag_1','Lag_2','Lag_3','Lag_5','Lag_7']

    st.markdown('<div class="sec-hdr">📊 MULTI-COIN COMPARISON</div>', unsafe_allow_html=True)

    # ── Normalized price chart (% growth from common start date) ──
    st.markdown('<div class="sec-hdr">📈 NORMALIZED PRICE GROWTH (% from common start)</div>', unsafe_allow_html=True)
    fig_norm = go.Figure()
    for i, c in enumerate(compare_coins):
        if c not in available: continue
        df_c = df_raw[df_raw['Symbol']==c].sort_values('Date').reset_index(drop=True)
        base = df_c['Close'].iloc[0]
        norm = (df_c['Close'] / base) * 100
        fig_norm.add_trace(go.Scatter(
            x=df_c['Date'], y=norm,
            name=c, line=dict(color=COIN_COLORS[i % len(COIN_COLORS)], width=2)
        ))
    fig_norm.add_hline(y=100, line_dash='dash', line_color='rgba(255,255,255,0.2)', line_width=1)
    fig_norm.update_layout(yaxis_title='Growth Index (100 = start)')
    st.plotly_chart(dark_fig(fig_norm, 420), use_container_width=True)

    # ── Side-by-side recent price ──
    st.markdown('<div class="sec-hdr">💹 RECENT PRICE (Last 365 Days)</div>', unsafe_allow_html=True)
    cols = st.columns(len(compare_coins))
    for i, c in enumerate(compare_coins):
        if c not in available: continue
        df_c = df_raw[df_raw['Symbol']==c].sort_values('Date').reset_index(drop=True).tail(365)
        with cols[i]:
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(
                x=df_c['Date'], y=df_c['Close'],
                name=c, line=dict(color=COIN_COLORS[i % len(COIN_COLORS)], width=2),
                fill='tozeroy', fillcolor=f'rgba({int(COIN_COLORS[i%len(COIN_COLORS)][1:3],16)},{int(COIN_COLORS[i%len(COIN_COLORS)][3:5],16)},{int(COIN_COLORS[i%len(COIN_COLORS)][5:],16)},0.06)'
            ))
            fig_mini.update_layout(
                title=dict(text=f"{COIN_EMOJI.get(c,'◆')} {c}", font=dict(size=13, color='#c8d8e8')),
                showlegend=False
            )
            st.plotly_chart(dark_fig(fig_mini, 220), use_container_width=True)

    # ── Correlation heatmap ──
    st.markdown('<div class="sec-hdr">🔗 PRICE CORRELATION HEATMAP</div>', unsafe_allow_html=True)
    price_df = pd.DataFrame()
    for c in compare_coins:
        if c not in available: continue
        df_c = df_raw[df_raw['Symbol']==c].sort_values('Date').set_index('Date')['Close']
        price_df[c] = df_c
    price_df.dropna(inplace=True)
    corr = price_df.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,'#0a1628'],[0.5,'#00d4ff'],[1,'#7b61ff']],
        text=[[f'{v:.2f}' for v in row] for row in corr.values],
        texttemplate='%{text}', textfont=dict(size=13),
        zmin=-1, zmax=1
    ))
    st.plotly_chart(dark_fig(fig_corr, 380), use_container_width=True)

    # ── Run predictions for all coins ──
    if not run_btn:
        st.markdown("""<div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);
        border-radius:14px;padding:1rem 1.5rem;
        font-family:Space Mono,monospace;font-size:0.78rem;color:#3a7a9a;'>
        👈 Click <strong style="color:#00d4ff">📊 Compare Coins</strong> to run predictions for all selected coins!
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="sec-hdr">🔮 PREDICTIONS FOR ALL COINS</div>', unsafe_allow_html=True)

    results = []
    model_name = model_choice.split('(')[0].strip()

    for c in compare_coins:
        if c not in available: continue
        df_c = df_raw[df_raw['Symbol']==c].sort_values('Date').reset_index(drop=True)
        df_f = add_features(df_c)
        if len(df_f) < 100: continue

        X = df_f[FEATS].values
        y = df_f['Close'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        with st.spinner(f"Training {model_name} for {c}..."):
            if 'Random Forest' in model_choice:
                mdl = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            elif 'Gradient Boosting' in model_choice:
                mdl = GradientBoostingRegressor(n_estimators=150, random_state=42)
            else:
                mdl = LinearRegression()
            mdl.fit(X_tr_s, y_tr)

        last_close_c   = df_f['Close'].iloc[-1]
        last_row_c     = df_f[FEATS].iloc[-1].values.reshape(1,-1)
        tomorrow_c     = mdl.predict(scaler.transform(last_row_c))[0]
        chg_c          = ((tomorrow_c - last_close_c)/last_close_c)*100
        mae_c          = mean_absolute_error(y_te, mdl.predict(X_te_s))
        r2_c           = r2_score(y_te, mdl.predict(X_te_s))
        acc_c          = max(0, 100-(mae_c/last_close_c*100))

        results.append({
            'coin': c,
            'emoji': COIN_EMOJI.get(c,'◆'),
            'last_close': last_close_c,
            'tomorrow': tomorrow_c,
            'change_pct': chg_c,
            'mae': mae_c,
            'r2': r2_c,
            'accuracy': acc_c,
            'signal': '📈 BULLISH' if chg_c >= 0 else '📉 BEARISH'
        })

    # ── Summary table ──
    st.markdown('<div class="sec-hdr">🏆 COMPARISON TABLE</div>', unsafe_allow_html=True)
    result_cols = st.columns(len(results))
    for i, r in enumerate(results):
        with result_cols[i]:
            chg_color = '#00ff99' if r['change_pct'] >= 0 else '#ff4466'
            st.markdown(f"""
            <div class="kpi-card" style="border-color:rgba({int(COIN_COLORS[i%len(COIN_COLORS)][1:3],16)},{int(COIN_COLORS[i%len(COIN_COLORS)][3:5],16)},{int(COIN_COLORS[i%len(COIN_COLORS)][5:],16)},0.3)">
                <div style="font-size:1.6rem;margin-bottom:0.3rem">{r['emoji']}</div>
                <div style="font-weight:900;font-size:1.1rem;color:{COIN_COLORS[i%len(COIN_COLORS)]};margin-bottom:0.8rem">{r['coin']}</div>
                <div class="kpi-label">Last Close</div>
                <div style="font-weight:700;color:#8aabcc;margin-bottom:0.6rem">${r['last_close']:,.2f}</div>
                <div class="kpi-label">Predicted Tomorrow</div>
                <div style="font-weight:900;font-size:1.3rem;color:#ffffff;margin-bottom:0.4rem">${r['tomorrow']:,.2f}</div>
                <div style="font-weight:700;font-size:1rem;color:{chg_color};margin-bottom:0.6rem">{r['change_pct']:+.2f}%</div>
                <div style="font-size:0.75rem;color:{chg_color};font-weight:700">{r['signal']}</div>
                <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.05)">
                    <div class="kpi-label">Accuracy</div>
                    <div style="font-weight:700;color:#00d4ff">{r['accuracy']:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Bar chart comparison ──
    if results:
        st.markdown('<div class="sec-hdr">📊 PREDICTED % CHANGE COMPARISON</div>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=[r['coin'] for r in results],
            y=[r['change_pct'] for r in results],
            marker_color=[('#00ff99' if r['change_pct']>=0 else '#ff4466') for r in results],
            marker_line_width=0,
            text=[f"{r['change_pct']:+.2f}%" for r in results],
            textposition='outside',
            textfont=dict(size=13, color='#c8d8e8')
        ))
        fig_bar.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1)
        fig_bar.update_layout(yaxis_title='Expected % Change Tomorrow', showlegend=False)
        st.plotly_chart(dark_fig(fig_bar, 360), use_container_width=True)

        # ── Best pick ──
        best = max(results, key=lambda x: x['change_pct'])
        worst = min(results, key=lambda x: x['change_pct'])
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem">
            <div class="pred-outer">
                <div class="pred-inner">
                    <div class="pred-tag">🏆 MOST BULLISH SIGNAL</div>
                    <div style="font-size:2.5rem;margin:0.5rem 0">{best['emoji']}</div>
                    <div style="font-weight:900;font-size:2rem;color:#00ff99">{best['coin']}</div>
                    <div style="font-size:1.1rem;color:#6080a0;margin:0.3rem 0">${best['last_close']:,.2f} → <strong style="color:#fff">${best['tomorrow']:,.2f}</strong></div>
                    <div class="pred-chg-up">📈 {best['change_pct']:+.2f}% · BULLISH</div>
                </div>
            </div>
            <div class="pred-outer">
                <div class="pred-inner">
                    <div class="pred-tag">⚠️ MOST BEARISH SIGNAL</div>
                    <div style="font-size:2.5rem;margin:0.5rem 0">{worst['emoji']}</div>
                    <div style="font-weight:900;font-size:2rem;color:#ff4466">{worst['coin']}</div>
                    <div style="font-size:1.1rem;color:#6080a0;margin:0.3rem 0">${worst['last_close']:,.2f} → <strong style="color:#fff">${worst['tomorrow']:,.2f}</strong></div>
                    <div class="pred-chg-down">📉 {worst['change_pct']:+.2f}% · BEARISH</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer">
    ⚠️ EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE · CRYPTO MARKETS ARE HIGHLY VOLATILE
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SINGLE COIN MODE — original flow continues below
# ─────────────────────────────────────────────────────────────────
if coin not in available:
    st.warning(f"⚠️ {coin} not found. Available coins: {available}")
    st.stop()

df_coin = df_raw[df_raw['Symbol']==coin].sort_values('Date').reset_index(drop=True)
emoji   = COIN_EMOJI.get(coin, '◆')

# ─────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────
last_close = df_coin['Close'].iloc[-1]
prev_close = df_coin['Close'].iloc[-2]
daily_chg  = ((last_close - prev_close)/prev_close)*100
ath        = df_coin['High'].max()
atl        = df_coin['Low'].min()
total_days = len(df_coin)

st.markdown(f'<div class="sec-hdr">{emoji} {coin} · MARKET OVERVIEW</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Last Close</div>
    <div class="kpi-val white">${last_close:,.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">24h Change</div>
    <div class="kpi-val {'green' if daily_chg>=0 else 'red'}">{daily_chg:+.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time High</div>
    <div class="kpi-val gold">${ath:,.0f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time Low</div>
    <div class="kpi-val">${atl:,.2f}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Days of Data</div>
    <div class="kpi-val white">{total_days:,}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PRICE CHART
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 PRICE HISTORY · CANDLESTICK</div>', unsafe_allow_html=True)

fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      row_heights=[0.75,0.25], vertical_spacing=0.01)
fig_c.add_trace(go.Candlestick(
    x=df_coin['Date'], open=df_coin['Open'], high=df_coin['High'],
    low=df_coin['Low'], close=df_coin['Close'], name='OHLC',
    increasing=dict(line=dict(color='#00ff99',width=1), fillcolor='rgba(0,255,153,0.7)'),
    decreasing=dict(line=dict(color='#ff4466',width=1), fillcolor='rgba(255,68,102,0.7)')
), row=1, col=1)
fig_c.add_trace(go.Bar(
    x=df_coin['Date'], y=df_coin['Volume'], name='Volume',
    marker_color='rgba(0,212,255,0.18)', marker_line_width=0
), row=2, col=1)
fig_c.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(dark_fig(fig_c, 500), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────
df = add_features(df_coin)

col_ma, col_rsi = st.columns([3,1])

with col_ma:
    st.markdown('<div class="sec-hdr">📊 MOVING AVERAGES</div>', unsafe_allow_html=True)
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Date'],y=df['Close'], name='Close', line=dict(color='rgba(255,255,255,0.5)',width=1)))
    fig_ma.add_trace(go.Scatter(x=df['Date'],y=df['MA_7'],  name='MA 7',  line=dict(color='#ffd700',width=2)))
    fig_ma.add_trace(go.Scatter(x=df['Date'],y=df['MA_30'], name='MA 30', line=dict(color='#00d4ff',width=2)))
    fig_ma.add_trace(go.Scatter(x=df['Date'],y=df['MA_90'], name='MA 90', line=dict(color='#7b61ff',width=2)))
    st.plotly_chart(dark_fig(fig_ma, 320), use_container_width=True)

with col_rsi:
    st.markdown('<div class="sec-hdr">⚡ RSI (14)</div>', unsafe_allow_html=True)
    last_30 = df.tail(120)
    fig_rsi = go.Figure()
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(255,68,102,0.07)', line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,255,153,0.07)',  line_width=0)
    fig_rsi.add_hline(y=70, line_dash='dash', line_color='rgba(255,68,102,0.4)', line_width=1)
    fig_rsi.add_hline(y=30, line_dash='dash', line_color='rgba(0,255,153,0.4)', line_width=1)
    fig_rsi.add_trace(go.Scatter(
        x=last_30['Date'], y=last_30['RSI_14'], name='RSI',
        line=dict(color='#ff9944', width=2),
        fill='tozeroy', fillcolor='rgba(255,153,68,0.05)'
    ))
    fig_rsi.update_yaxes(range=[0,100])
    st.plotly_chart(dark_fig(fig_rsi, 320), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# WAIT FOR RUN
# ─────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);
    border-radius:14px;padding:1rem 1.5rem;
    font-family:Space Mono,monospace;font-size:0.78rem;color:#3a7a9a;'>
    👈 &nbsp; Click <strong style="color:#00d4ff">🚀 Run Prediction</strong> in the sidebar to train the model!
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────────
FEATS = ['Open','High','Low','Volume','MA_7','MA_14','MA_30',
         'RSI_14','Daily_Return','Volatility_7','Price_Range',
         'Lag_1','Lag_2','Lag_3','Lag_5','Lag_7']

X = df[FEATS].values
y = df['Close'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler   = MinMaxScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)
te_dates = df['Date'].values[len(y_tr):]

model_name = model_choice.split('(')[0].strip()
with st.spinner(f"⚡ Training {model_name}... (5–15 seconds)"):
    if 'Random Forest' in model_choice:
        mdl = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    elif 'Gradient Boosting' in model_choice:
        mdl = GradientBoostingRegressor(n_estimators=150, random_state=42, learning_rate=0.08)
    else:
        mdl = LinearRegression()
    mdl.fit(X_tr_s, y_tr)
    y_pred = mdl.predict(X_te_s)

mae  = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
r2   = r2_score(y_te, y_pred)
acc  = max(0, 100 - (mae/last_close*100))

st.success(f"✅ **{model_name}** trained successfully on {len(y_tr):,} data points!")

# ─────────────────────────────────────────────────────────────────
# MODEL METRICS
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🏆 MODEL PERFORMANCE</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
for col,lbl,val,cls in zip([c1,c2,c3,c4],
    ["Mean Abs Error","Root Mean Sq Err","R² Score","Approx Accuracy"],
    [f"${mae:,.0f}", f"${rmse:,.0f}", f"{r2:.4f}", f"{acc:.1f}%"],
    ["red","red","green","green"]):
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-val {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PREDICTION CHART + FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
col_pred, col_feat = st.columns([3,2])

with col_pred:
    st.markdown('<div class="sec-hdr">🎯 ACTUAL vs PREDICTED</div>', unsafe_allow_html=True)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=te_dates, y=y_te,   name='Actual',
                               line=dict(color='#00d4ff',width=2.5)))
    fig_p.add_trace(go.Scatter(x=te_dates, y=y_pred, name='Predicted',
                               line=dict(color='#ff6b6b',width=2,dash='dot')))
    fig_p.add_trace(go.Scatter(
        x=list(te_dates)+list(te_dates[::-1]),
        y=list(y_te)+list(y_pred[::-1]),
        fill='toself', fillcolor='rgba(0,212,255,0.04)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, name='Range'
    ))
    st.plotly_chart(dark_fig(fig_p, 380), use_container_width=True)

with col_feat:
    st.markdown('<div class="sec-hdr">🧠 FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    if hasattr(mdl, 'feature_importances_'):
        imp = pd.DataFrame({'Feature':FEATS,'Importance':mdl.feature_importances_})
        imp = imp.sort_values('Importance').tail(10)
        colors = ['rgba(0,212,255,0.5)' if i < 7 else '#00d4ff' for i in range(len(imp))]
        fig_f = go.Figure(go.Bar(
            x=imp['Importance'], y=imp['Feature'], orientation='h',
            marker_color=colors, marker_line_width=0
        ))
        fig_f.update_layout(xaxis_title='Importance Score')
        st.plotly_chart(dark_fig(fig_f, 380), use_container_width=True)
    else:
        # For Linear Regression show coefficients
        coef = pd.DataFrame({'Feature':FEATS,'Coefficient':np.abs(mdl.coef_)})
        coef = coef.sort_values('Coefficient').tail(10)
        fig_f = go.Figure(go.Bar(
            x=coef['Coefficient'], y=coef['Feature'], orientation='h',
            marker_color='#7b61ff', marker_line_width=0
        ))
        st.plotly_chart(dark_fig(fig_f, 380), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# TOMORROW'S PREDICTION
# ─────────────────────────────────────────────────────────────────
last_row       = df[FEATS].iloc[-1].values.reshape(1,-1)
tomorrow_price = mdl.predict(scaler.transform(last_row))[0]
chg_pct        = ((tomorrow_price - last_close)/last_close)*100
is_up          = chg_pct >= 0
chg_cls        = "pred-chg-up" if is_up else "pred-chg-down"
chg_icon       = "📈" if is_up else "📉"
chg_word       = "BULLISH" if is_up else "BEARISH"

st.markdown('<div class="sec-hdr">🔮 TOMORROW\'S PRICE FORECAST</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="pred-outer">
  <div class="pred-inner">
    <div class="pred-tag">🔮 {coin} · {model_name.upper()} FORECAST</div>
    <div class="pred-row">
      <div class="pred-block">
        <div class="pred-lbl">Last Close</div>
        <div class="pred-amt muted">${last_close:,.2f}</div>
      </div>
      <div class="pred-arrow">→</div>
      <div class="pred-block">
        <div class="pred-lbl">Predicted Tomorrow</div>
        <div class="pred-amt">${tomorrow_price:,.2f}</div>
      </div>
    </div>
    <div class="{chg_cls}">{chg_icon} &nbsp; {chg_pct:+.2f}% &nbsp;·&nbsp; {chg_word} SIGNAL</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ EDUCATIONAL PURPOSES ONLY &nbsp;·&nbsp; NOT FINANCIAL ADVICE &nbsp;·&nbsp;
CRYPTO MARKETS ARE HIGHLY VOLATILE &nbsp;·&nbsp; NEVER INVEST BASED ON MODEL PREDICTIONS ALONE
</div>
""", unsafe_allow_html=True)
