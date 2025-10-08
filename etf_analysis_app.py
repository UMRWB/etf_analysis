import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta import others
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Page configuration
st.set_page_config(
    page_title="ETF Monthly Return Analysis & Recommender",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 ETF Monthly Return Analysis & Investment Recommender")
st.markdown("Analyze monthly return patterns and get ETF recommendations based on historical performance")

# Sidebar configuration
st.sidebar.header("Configuration")

# ETF selections
BROAD_CORES = ["VOO", "VTI", "VT", "VXUS", "VEA"]
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
COMMODITIES = ["GLD", "SLV"]

period = st.sidebar.selectbox("Time Period", ["5y", "10y", "max"], index=1)
interval = "1mo"

# Asset class selection
asset_classes = st.sidebar.multiselect(
    "Select Asset Classes",
    ["Broad Core ETFs", "Sector ETFs", "Commodity ETFs"],
    default=["Broad Core ETFs", "Sector ETFs", "Commodity ETFs"]
)

# Recommender settings
st.sidebar.header("Recommender Settings")
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)
weight_prob = st.sidebar.slider("Probability weight", 0.0, 1.0, 0.5, 0.1)
weight_return = 1.0 - weight_prob
st.sidebar.caption(f"Return weight: {weight_return:.1f}")

# Cache data loading
@st.cache_data(ttl=3600)
def load_etf_data(tickers, period, interval):
    """Load ETF data from Yahoo Finance"""
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"Loading {ticker}... ({idx+1}/{len(tickers)})")
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if not df.empty:
                df["pct_change"] = df["Close"].pct_change() * 100
                df["cumulative_return"] = others.cumulative_return(df["Close"])
                data_dict[ticker] = df
        except Exception as e:
            st.warning(f"Error loading {ticker}: {str(e)}")

        progress_bar.progress((idx + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()
    return data_dict

# Helper functions
def calculate_annual_sharpe(pct_return: pd.Series, risk_free_rate: float = 0.05):
    """Calculate annualized Sharpe ratio"""
    if pct_return.std() == 0:
        return np.nan
    return ((pct_return.mean() - risk_free_rate) / pct_return.std()) * np.sqrt(252)

def calc_positive_prob(column):
    """Calculate probability of positive returns"""
    valid_values = column.dropna()
    if len(valid_values) == 0:
        return np.nan
    positive_count = (valid_values > 0).sum()
    total_count = len(valid_values)
    return (positive_count / total_count) * 100

def calculate_monthly_stats_for_etfs(etf_dict, etf_list):
    """Calculate monthly statistics for all ETFs"""
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']

    prob_results = {}
    avg_return_results = {}

    for etf in etf_list:
        if etf not in etf_dict or etf_dict[etf].empty:
            continue

        df_temp = etf_dict[etf][["pct_change"]].copy()
        df_temp['Year'] = df_temp.index.year
        df_temp['Month'] = df_temp.index.month_name()

        pivot = df_temp.pivot_table(
            values='pct_change',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        pivot = pivot.reindex(columns=month_order)

        prob_results[etf] = pivot.apply(calc_positive_prob, axis=0)
        avg_return_results[etf] = pivot.mean(axis=0)

    return pd.DataFrame(prob_results).T, pd.DataFrame(avg_return_results).T

def style_dataframe_prob(df):
    """Apply consistent styling to probability dataframes"""
    return df.style.background_gradient(
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        axis=None
    ).format("{:.2f}")

def style_dataframe_returns(df):
    """Apply consistent styling to returns dataframes"""
    # Find global min and max for consistent coloring
    vmin = df.min().min()
    vmax = df.max().max()

    return df.style.background_gradient(
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        axis=None
    ).format("{:.2f}")

def get_recommendations(prob_df, avg_df, month_name, top_n=5, weight_prob=0.5, weight_return=0.5):
    """
    Get ETF recommendations for a specific month based on probability and returns

    Parameters:
    - prob_df: DataFrame with probability of positive returns
    - avg_df: DataFrame with average returns
    - month_name: Name of the month to get recommendations for
    - top_n: Number of recommendations to return
    - weight_prob: Weight for probability (0-1)
    - weight_return: Weight for average return (0-1)
    """
    if month_name not in prob_df.columns or month_name not in avg_df.columns:
        return None

    # Normalize probability and returns to 0-100 scale
    prob_scores = prob_df[month_name].fillna(0)

    # Normalize returns (scale to 0-100)
    returns = avg_df[month_name].fillna(avg_df[month_name].min())
    return_min = returns.min()
    return_max = returns.max()
    if return_max != return_min:
        return_scores = ((returns - return_min) / (return_max - return_min)) * 100
    else:
        return_scores = pd.Series(50, index=returns.index)

    # Calculate composite score
    composite_score = (weight_prob * prob_scores) + (weight_return * return_scores)

    # Create recommendation dataframe
    recommendations = pd.DataFrame({
        'ETF': composite_score.index,
        'Composite_Score': composite_score.values,
        'Probability_%': prob_scores.values,
        'Avg_Return_%': avg_df[month_name].values,
        'Sharpe_Ratio': [np.nan] * len(composite_score)  # Will be filled later
    })

    recommendations = recommendations.sort_values('Composite_Score', ascending=False)
    recommendations['Rank'] = range(1, len(recommendations) + 1)

    return recommendations.head(top_n)

def plot_cumulative_return(data_dict, title="Cumulative Return"):
    """Create plotly cumulative return chart"""
    fig = go.Figure()

    for ticker in data_dict:
        if not data_dict[ticker].empty:
            fig.add_trace(
                go.Scatter(
                    x=data_dict[ticker].index, 
                    y=data_dict[ticker]["cumulative_return"], 
                    name=ticker,
                    mode='lines'
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        showlegend=True,
        height=500,
        hovermode='x unified'
    )

    return fig

# Main app logic
# Determine which ETFs to load
selected_tickers = []
if "Broad Core ETFs" in asset_classes:
    selected_tickers.extend(BROAD_CORES)
if "Sector ETFs" in asset_classes:
    selected_tickers.extend(SECTORS)
if "Commodity ETFs" in asset_classes:
    selected_tickers.extend(COMMODITIES)

if not selected_tickers:
    st.warning("Please select at least one asset class from the sidebar.")
    st.stop()

# Load data
with st.spinner("Loading ETF data..."):
    all_data = load_etf_data(selected_tickers, period, interval)

if not all_data:
    st.error("No data loaded. Please check your selections.")
    st.stop()

# Calculate Sharpe ratios for all ETFs
sharpe_results = {}
for ticker, data in all_data.items():
    if not data.empty and 'pct_change' in data.columns:
        sharpe = calculate_annual_sharpe(data['pct_change'].dropna())
        sharpe_results[ticker] = sharpe

# Calculate monthly statistics
all_prob, all_avg = calculate_monthly_stats_for_etfs(all_data, list(all_data.keys()))

# Get current month and next month
current_date = datetime.now()
current_month = current_date.strftime("%B")
next_month_date = current_date + relativedelta(months=1)
next_month = next_month_date.strftime("%B")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Recommendations", 
    "📊 Overview", 
    "📈 Cumulative Returns", 
    "🎲 Monthly Probability", 
    "💰 Average Returns",
    "📉 Sharpe Ratios"
])

with tab1:
    st.header("🎯 ETF Investment Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📅 Recommendations for {current_month}")
        st.caption(f"Current Month: {current_month} {current_date.year}")

        current_recs = get_recommendations(
            all_prob, all_avg, current_month, 
            top_n=top_n, 
            weight_prob=weight_prob, 
            weight_return=weight_return
        )

        if current_recs is not None:
            # Add Sharpe ratios
            current_recs['Sharpe_Ratio'] = current_recs['ETF'].map(sharpe_results)

            # Display recommendations
            for idx, row in current_recs.iterrows():
                with st.container():
                    st.markdown(f"### #{int(row['Rank'])} - **{row['ETF']}**")

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Score", f"{row['Composite_Score']:.1f}")
                    with metric_col2:
                        st.metric("Win Probability", f"{row['Probability_%']:.1f}%")
                    with metric_col3:
                        st.metric("Avg Return", f"{row['Avg_Return_%']:.2f}%")
                    with metric_col4:
                        if not pd.isna(row['Sharpe_Ratio']):
                            st.metric("Sharpe Ratio", f"{row['Sharpe_Ratio']:.2f}")
                        else:
                            st.metric("Sharpe Ratio", "N/A")

                    st.markdown("---")

            # Download button
            csv = current_recs.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"Download {current_month} Recommendations",
                csv,
                f"recommendations_{current_month.lower()}.csv",
                "text/csv"
            )
        else:
            st.warning(f"No data available for {current_month}")

    with col2:
        st.subheader(f"📅 Recommendations for {next_month}")
        st.caption(f"Next Month: {next_month} {next_month_date.year}")

        next_recs = get_recommendations(
            all_prob, all_avg, next_month, 
            top_n=top_n, 
            weight_prob=weight_prob, 
            weight_return=weight_return
        )

        if next_recs is not None:
            # Add Sharpe ratios
            next_recs['Sharpe_Ratio'] = next_recs['ETF'].map(sharpe_results)

            # Display recommendations
            for idx, row in next_recs.iterrows():
                with st.container():
                    st.markdown(f"### #{int(row['Rank'])} - **{row['ETF']}**")

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Score", f"{row['Composite_Score']:.1f}")
                    with metric_col2:
                        st.metric("Win Probability", f"{row['Probability_%']:.1f}%")
                    with metric_col3:
                        st.metric("Avg Return", f"{row['Avg_Return_%']:.2f}%")
                    with metric_col4:
                        if not pd.isna(row['Sharpe_Ratio']):
                            st.metric("Sharpe Ratio", f"{row['Sharpe_Ratio']:.2f}")
                        else:
                            st.metric("Sharpe Ratio", "N/A")

                    st.markdown("---")

            # Download button
            csv = next_recs.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"Download {next_month} Recommendations",
                csv,
                f"recommendations_{next_month.lower()}.csv",
                "text/csv"
            )
        else:
            st.warning(f"No data available for {next_month}")

    # Methodology explanation
    st.markdown("---")
    st.subheader("📋 Methodology")
    st.markdown(f"""
    **How recommendations are calculated:**

    1. **Composite Score** = ({weight_prob:.1f} × Probability) + ({weight_return:.1f} × Normalized Return)
    2. **Probability**: Historical percentage of positive returns for this month
    3. **Average Return**: Mean percentage return for this month across all years
    4. **Sharpe Ratio**: Overall risk-adjusted return measure

    **Note**: Past performance does not guarantee future results. These recommendations are based solely on historical patterns.
    """)

with tab2:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total ETFs Analyzed", len(all_data))
    with col2:
        st.metric("Time Period", period.upper())
    with col3:
        st.metric("Current Month", current_month)

    # Separate by asset class
    broad_cores_dict = {k: v for k, v in all_data.items() if k in BROAD_CORES}
    sectors_dict = {k: v for k, v in all_data.items() if k in SECTORS}
    commodities_dict = {k: v for k, v in all_data.items() if k in COMMODITIES}

    if broad_cores_dict:
        st.write("**Broad Core ETFs:**", ", ".join(broad_cores_dict.keys()))
    if sectors_dict:
        st.write("**Sector ETFs:**", ", ".join(sectors_dict.keys()))
    if commodities_dict:
        st.write("**Commodity ETFs:**", ", ".join(commodities_dict.keys()))

with tab3:
    st.header("Cumulative Returns")

    # Separate by asset class
    broad_cores_dict = {k: v for k, v in all_data.items() if k in BROAD_CORES}
    sectors_dict = {k: v for k, v in all_data.items() if k in SECTORS}
    commodities_dict = {k: v for k, v in all_data.items() if k in COMMODITIES}

    if broad_cores_dict:
        st.subheader("Broad Core ETFs")
        fig = plot_cumulative_return(broad_cores_dict, "Cumulative Return - Broad Core ETFs")
        st.plotly_chart(fig, use_container_width=True)

    if sectors_dict:
        st.subheader("Sector ETFs")
        fig = plot_cumulative_return(sectors_dict, "Cumulative Return - Sector ETFs")
        st.plotly_chart(fig, use_container_width=True)

    if commodities_dict:
        st.subheader("Commodity ETFs")
        fig = plot_cumulative_return(commodities_dict, "Cumulative Return - Commodity ETFs")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Monthly Probability of Positive Returns")

    st.subheader("All ETFs - Full Year")
    st.dataframe(style_dataframe_prob(all_prob), use_container_width=True)

    # Highlight current and next month
    st.subheader(f"Focus: {current_month} & {next_month}")

    focus_months = [current_month, next_month]
    focus_prob = all_prob[focus_months]
    st.dataframe(style_dataframe_prob(focus_prob), use_container_width=True)

    csv = all_prob.to_csv().encode('utf-8')
    st.download_button(
        "Download All Probability Data",
        csv,
        "monthly_probability.csv",
        "text/csv"
    )

with tab5:
    st.header("Average Monthly Returns")

    st.subheader("All ETFs - Full Year")
    st.dataframe(style_dataframe_returns(all_avg), use_container_width=True)

    # Highlight current and next month
    st.subheader(f"Focus: {current_month} & {next_month}")

    focus_months = [current_month, next_month]
    focus_avg = all_avg[focus_months]
    st.dataframe(style_dataframe_returns(focus_avg), use_container_width=True)

    csv = all_avg.to_csv().encode('utf-8')
    st.download_button(
        "Download All Average Returns",
        csv,
        "monthly_avg_returns.csv",
        "text/csv"
    )

with tab6:
    st.header("Sharpe Ratios")

    if sharpe_results:
        sharpe_df = pd.DataFrame.from_dict(sharpe_results, orient='index', columns=['Sharpe Ratio'])
        sharpe_df = sharpe_df.sort_values('Sharpe Ratio', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure(data=[
                go.Bar(x=sharpe_df.index, y=sharpe_df['Sharpe Ratio'],
                       marker_color='steelblue')
            ])
            fig.update_layout(
                title="Annualized Sharpe Ratios",
                xaxis_title="ETF",
                yaxis_title="Sharpe Ratio",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Rankings")
            st.dataframe(sharpe_df.round(3), use_container_width=True)

            csv = sharpe_df.to_csv().encode('utf-8')
            st.download_button(
                "Download Sharpe Ratios",
                csv,
                "sharpe_ratios.csv",
                "text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data source: Yahoo Finance via yfinance")
st.sidebar.caption("⚠️ Disclaimer: Past performance does not guarantee future results. These recommendations are for informational purposes only and should not be considered financial advice.")
