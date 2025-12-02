import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.linear_model import LinearRegression

# --- App Configuration ---
st.set_page_config(page_title="Pro Trader Dashboard", layout="wide")
st.title("📊 Pro Asset Analysis: Seasonality, Technicals & AI Forecast")

# --- Ticker Mapping ---
TICKER_MAP = {
    "SPY (S&P 500 ETF)": "SPY",
    "QQQ (Nasdaq 100 ETF)": "QQQ",
    "IWM (Russell 2000 ETF)": "IWM",
    "XAUUSD (Gold Futures)": "GC=F",
    "XAGUSD (Silver Futures)": "SI=F",
    "BTCUSD (Bitcoin)": "BTC-USD",
    "ETHUSD (Ethereum)": "ETH-USD"
}

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
selected_label = st.sidebar.selectbox("Select Asset", options=list(TICKER_MAP.keys()))
ticker = TICKER_MAP[selected_label]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01")) # Longer history for better seasonality
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
forecast_period = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 90)

# --- Helper Functions ---

@st.cache_data
def load_data(symbol, start, end):
    """Fetches and cleans historical stock data."""
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Ensure Date is timezone-naive
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    
    # Calculate Daily Returns
    data['Return'] = data['Close'].pct_change()
    
    return data

def calculate_technicals(df):
    """Calculates technical indicators and generates a signal."""
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def generate_rating(last_row):
    """Generates a Buy/Sell/Neutral rating based on indicators."""
    score = 0
    reasons = []
    
    # SMA Trend
    if last_row['Close'] > last_row['SMA_50'] > last_row['SMA_200']:
        score += 2
        reasons.append("Price in strong uptrend (Above SMA 50 & 200)")
    elif last_row['Close'] < last_row['SMA_50'] < last_row['SMA_200']:
        score -= 2
        reasons.append("Price in strong downtrend (Below SMA 50 & 200)")
    
    # RSI
    if last_row['RSI'] < 30:
        score += 1
        reasons.append("RSI Oversold (<30)")
    elif last_row['RSI'] > 70:
        score -= 1
        reasons.append("RSI Overbought (>70)")
        
    # MACD
    if last_row['MACD'] > last_row['Signal_Line']:
        score += 1
        reasons.append("MACD Bullish Crossover")
    else:
        score -= 1
        reasons.append("MACD Bearish Crossover")
        
    # Final Rating
    if score >= 2:
        rating = "STRONG BUY"
        color = "green"
    elif score == 1:
        rating = "BUY"
        color = "green"
    elif score == 0:
        rating = "NEUTRAL"
        color = "gray"
    elif score == -1:
        rating = "SELL"
        color = "red"
    else:
        rating = "STRONG SELL"
        color = "red"
        
    return rating, color, reasons, score

# --- Main Logic ---

data_load_state = st.text(f'Loading data for {ticker}...')
try:
    data = load_data(ticker, start_date, end_date)
    data = calculate_technicals(data)
    data_load_state.empty()
    
    if data.empty:
        st.error("No data found.")
        st.stop()

    # Get latest data for metrics
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # --- Top Level Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${latest['Close']:.2f}", f"{(latest['Close'] - prev['Close']):.2f}")
    col2.metric("RSI (14)", f"{latest['RSI']:.1f}", delta=None)
    
    rating, r_color, r_reasons, r_score = generate_rating(latest)
    col3.markdown(f"### Signal: :{r_color}[{rating}]")
    col4.markdown(f"**Score:** {r_score}/4")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Summary", "📅 Historical Seasonality", "🔮 Prophet Forecast", "📏 Linear Trend"])
    
    # --- TAB 1: TECHNICAL SUMMARY ---
    with tab1:
        st.subheader("Technical Analysis Breakdown")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Indicator Signals")
            for r in r_reasons:
                if "uptrend" in r or "Bullish" in r or "Oversold" in r:
                    st.success(r)
                else:
                    st.error(r)
            
            st.divider()
            st.dataframe(pd.DataFrame({
                'Metric': ['SMA 50', 'SMA 200', 'MACD', 'Signal Line'],
                'Value': [f"{latest['SMA_50']:.2f}", f"{latest['SMA_200']:.2f}", f"{latest['MACD']:.4f}", f"{latest['Signal_Line']:.4f}"]
            }), hide_index=True)

        with c2:
            # Plot Price + SMAs
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='blue')))
            fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name="SMA 50", line=dict(color='orange')))
            fig_tech.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name="SMA 200", line=dict(color='red')))
            fig_tech.update_layout(title="Price vs Moving Averages", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_tech, use_container_width=True)

    # --- TAB 2: HISTORICAL SEASONALITY ---
    with tab2:
        st.subheader("Historical Monthly Performance")
        st.markdown("How does this asset perform on average during specific months?")
        
        # Prepare Monthly Data
        df_season = data.copy()
        df_season['Month'] = df_season['Date'].dt.month
        df_season['Month_Name'] = df_season['Date'].dt.strftime('%b')
        df_season['Year'] = df_season['Date'].dt.year
        
        # 1. Average Return by Month
        monthly_avg = df_season.groupby('Month')['Return'].mean() * 100
        monthly_names = [pd.to_datetime(str(m), format='%m').strftime('%b') for m in monthly_avg.index]
        
        # Color logic for bars
        colors = ['green' if val > 0 else 'red' for val in monthly_avg.values]

        fig_bar = go.Figure(data=[
            go.Bar(x=monthly_names, y=monthly_avg.values, marker_color=colors)
        ])
        fig_bar.update_layout(title="Average Monthly Return (%)", yaxis_title="Avg Return %")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2. Monthly Heatmap
        st.subheader("Monthly Returns Heatmap (Year over Year)")
        
        pivot_table = df_season.pivot_table(index='Year', columns='Month', values='Return', aggfunc=lambda x: (x+1).prod()-1)
        pivot_table = pivot_table * 100 # Convert to %
        
        # Reorder columns to be Jan-Dec
        pivot_table.columns = [pd.to_datetime(str(m), format='%m').strftime('%b') for m in pivot_table.columns]
        
        fig_heat = px.imshow(
            pivot_table, 
            labels=dict(x="Month", y="Year", color="Return %"),
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- TAB 3: PROPHET FORECAST ---
    with tab3:
        st.subheader(f"AI Price Prediction ({forecast_period} days)")
        
        df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=forecast_period)
        forecast = m.predict(future)
        
        fig_forecast = plot_plotly(m, forecast)
        fig_forecast.update_layout(height=500)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        with st.expander("See detailed components"):
            st.pyplot(m.plot_components(forecast))

    # --- TAB 4: LINEAR REGRESSION ---
    with tab4:
        st.subheader("Linear Trend Line")
        
        df_reg = data.copy()
        df_reg['Date_Ordinal'] = df_reg['Date'].map(pd.Timestamp.toordinal)
        X = df_reg[['Date_Ordinal']].values
        y = df_reg['Close'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        trend = lr.predict(X)
        
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close", line=dict(color='blue', width=1)))
        fig_lr.add_trace(go.Scatter(x=data['Date'], y=trend, name="Trend", line=dict(color='red', width=3, dash='dot')))
        st.plotly_chart(fig_lr, use_container_width=True)
        
        slope = lr.coef_[0]
        st.info(f"Linear Slope: {slope:.4f}")

except Exception as e:
    st.error(f"An error occurred: {e}")
