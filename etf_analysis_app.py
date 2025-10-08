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
    page_title="ETF Complete Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# BACKTEST CLASS
# ============================================================================

class ETFBacktester:
    """Backtest the ETF recommendation strategy"""

    def __init__(self, tickers, start_date, end_date, lookback_years=5):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.lookback_years = lookback_years
        self.data = {}
        self.results = []
        self.benchmark_data = None

    def load_data(self, progress_callback=None):
        """Load historical data for all tickers"""
        lookback_start = self.start_date - relativedelta(years=self.lookback_years + 1)
        loaded = 0

        for ticker in self.tickers:
            try:
                df = yf.Ticker(ticker).history(start=lookback_start, end=self.end_date, interval='1mo')
                if not df.empty:
                    df['pct_change'] = df['Close'].pct_change() * 100
                    df.index = pd.to_datetime(df.index, utc=True)
                    self.data[ticker] = df
                loaded += 1
                if progress_callback:
                    progress_callback(loaded / len(self.tickers))
            except:
                pass

    def load_benchmark(self, benchmark_ticker='VOO'):
        """Load benchmark data"""
        try:
            df = yf.Ticker(benchmark_ticker).history(start=self.start_date, end=self.end_date, interval='1mo')
            if not df.empty:
                df['pct_change'] = df['Close'].pct_change() * 100
                self.benchmark_data = df
                return True
        except:
            return False
        return False

    def calc_positive_prob(self, column):
        valid_values = column.dropna()
        if len(valid_values) == 0:
            return np.nan
        return ((valid_values > 0).sum() / len(valid_values)) * 100

    def get_monthly_stats(self, ticker, end_date):
        if ticker not in self.data:
            return None, None

        end_date = pd.to_datetime(end_date, utc=True)

        df = self.data[ticker][self.data[ticker].index <= end_date].copy()
        if len(df) < 12:
            return None, None

        df['Year'] = df.index.year
        df['Month'] = df.index.month_name()

        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']

        pivot = df.pivot_table(values='pct_change', index='Year', columns='Month', aggfunc='first')
        pivot = pivot.reindex(columns=month_order)

        prob = pivot.apply(self.calc_positive_prob, axis=0)
        avg_return = pivot.mean(axis=0)

        return prob, avg_return

    def get_recommendation(self, month_name, as_of_date, weight_prob=0.5, weight_return=0.5, top_n=1):
        all_probs = {}
        all_avgs = {}

        for ticker in self.tickers:
            prob, avg = self.get_monthly_stats(ticker, as_of_date)
            if prob is not None and month_name in prob.index:
                all_probs[ticker] = prob[month_name]
                all_avgs[ticker] = avg[month_name]

        if not all_probs:
            return None

        prob_series = pd.Series(all_probs).fillna(0)
        avg_series = pd.Series(all_avgs)

        return_min = avg_series.min()
        return_max = avg_series.max()
        if return_max != return_min:
            return_scores = ((avg_series - return_min) / (return_max - return_min)) * 100
        else:
            return_scores = pd.Series(50, index=avg_series.index)

        composite_score = (weight_prob * prob_series) + (weight_return * return_scores)
        top_etfs = composite_score.nlargest(top_n)

        return top_etfs.index.tolist()

    def run_backtest(self, weight_prob=0.5, weight_return=0.5, top_n=1, progress_callback=None):
        current_date = self.start_date
        portfolio_value = 100000
        benchmark_value = 100000
        results = []
        months_processed = 0
        total_months = (self.end_date.year - self.start_date.year) * 12 + (self.end_date.month - self.start_date.month) + 1

        while current_date <= self.end_date:
            month_name = current_date.strftime("%B")
            decision_date = current_date - relativedelta(months=1)

            recommended_etfs = self.get_recommendation(month_name, decision_date, weight_prob, weight_return, top_n)

            if recommended_etfs is None or len(recommended_etfs) == 0:
                current_date += relativedelta(months=1)
                continue

            # Calculate strategy returns
            month_returns = {}
            for ticker in recommended_etfs:
                if ticker in self.data:
                    ticker_data = self.data[ticker]
                    month_data = ticker_data[(ticker_data.index.year == current_date.year) & 
                                            (ticker_data.index.month == current_date.month)]

                    if not month_data.empty and 'pct_change' in month_data.columns:
                        ret = month_data['pct_change'].iloc[0]
                        if not pd.isna(ret):
                            month_returns[ticker] = ret

            if month_returns:
                portfolio_return = np.mean(list(month_returns.values()))
                portfolio_value *= (1 + portfolio_return / 100)
            else:
                portfolio_return = 0

            # Calculate benchmark returns
            benchmark_return = 0
            if self.benchmark_data is not None:
                bench_month = self.benchmark_data[(self.benchmark_data.index.year == current_date.year) & 
                                                 (self.benchmark_data.index.month == current_date.month)]
                if not bench_month.empty and 'pct_change' in bench_month.columns:
                    bench_ret = bench_month['pct_change'].iloc[0]
                    if not pd.isna(bench_ret):
                        benchmark_return = bench_ret
                        benchmark_value *= (1 + benchmark_return / 100)

            results.append({
                'Date': current_date,
                'Month': month_name,
                'Year': current_date.year,
                'Holdings': ', '.join(recommended_etfs),
                'Returns_%': portfolio_return,
                'Portfolio_Value': portfolio_value,
                'Benchmark_Returns_%': benchmark_return,
                'Benchmark_Value': benchmark_value
            })

            current_date += relativedelta(months=1)
            months_processed += 1
            if progress_callback:
                progress_callback(months_processed / total_months)

        self.results = pd.DataFrame(results)
        return self.results

    def calculate_metrics(self):
        if self.results.empty:
            return None, None

        returns = self.results['Returns_%'].values
        benchmark_returns = self.results['Benchmark_Returns_%'].values

        def annualized_return(rets):
            total_return = np.prod(1 + rets / 100) - 1
            n_years = len(rets) / 12
            if n_years > 0:
                return (((1 + total_return) ** (1 / n_years)) - 1) * 100
            return 0

        def sharpe_ratio(rets):
            if len(rets) == 0 or rets.std() == 0:
                return 0
            excess_return = rets.mean() - (0.05 / 12)
            return (excess_return / rets.std()) * np.sqrt(12)

        def max_drawdown(values):
            peak = values[0]
            max_dd = 0
            for value in values:
                if value > peak:
                    peak = value
                dd = ((peak - value) / peak) * 100
                if dd > max_dd:
                    max_dd = dd
            return max_dd

        strategy_metrics = {
            'Total Return (%)': ((self.results['Portfolio_Value'].iloc[-1] / 100000) - 1) * 100,
            'Annualized Return (%)': annualized_return(returns),
            'Total Months': len(returns),
            'Winning Months': (returns > 0).sum(),
            'Losing Months': (returns < 0).sum(),
            'Win Rate (%)': ((returns > 0).sum() / len(returns)) * 100,
            'Average Return (%)': returns.mean(),
            'Std Dev (%)': returns.std(),
            'Best Month (%)': returns.max(),
            'Worst Month (%)': returns.min(),
            'Sharpe Ratio': sharpe_ratio(returns),
            'Max Drawdown (%)': max_drawdown(self.results['Portfolio_Value'].values)
        }

        benchmark_metrics = {
            'Total Return (%)': ((self.results['Benchmark_Value'].iloc[-1] / 100000) - 1) * 100,
            'Annualized Return (%)': annualized_return(benchmark_returns),
            'Total Months': len(benchmark_returns),
            'Winning Months': (benchmark_returns > 0).sum(),
            'Losing Months': (benchmark_returns < 0).sum(),
            'Win Rate (%)': ((benchmark_returns > 0).sum() / len(benchmark_returns)) * 100,
            'Average Return (%)': benchmark_returns.mean(),
            'Std Dev (%)': benchmark_returns.std(),
            'Best Month (%)': benchmark_returns.max(),
            'Worst Month (%)': benchmark_returns.min(),
            'Sharpe Ratio': sharpe_ratio(benchmark_returns),
            'Max Drawdown (%)': max_drawdown(self.results['Benchmark_Value'].values)
        }

        return strategy_metrics, benchmark_metrics

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    vmin = df.min().min()
    vmax = df.max().max()

    return df.style.background_gradient(
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        axis=None
    ).format("{:.2f}")

def get_recommendations(prob_df, avg_df, month_name, top_n=5, weight_prob=0.5, weight_return=0.5):
    """Get ETF recommendations for a specific month"""
    if month_name not in prob_df.columns or month_name not in avg_df.columns:
        return None

    prob_scores = prob_df[month_name].fillna(0)
    returns = avg_df[month_name].fillna(avg_df[month_name].min())

    return_min = returns.min()
    return_max = returns.max()
    if return_max != return_min:
        return_scores = ((returns - return_min) / (return_max - return_min)) * 100
    else:
        return_scores = pd.Series(50, index=returns.index)

    composite_score = (weight_prob * prob_scores) + (weight_return * return_scores)

    recommendations = pd.DataFrame({
        'ETF': composite_score.index,
        'Composite_Score': composite_score.values,
        'Probability_%': prob_scores.values,
        'Avg_Return_%': avg_df[month_name].values,
        'Sharpe_Ratio': [np.nan] * len(composite_score)
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

# ============================================================================
# MAIN APP
# ============================================================================

st.title("📈 ETF Complete Analysis Platform")
st.markdown("Comprehensive ETF analysis with recommendations, backtesting, and performance metrics")

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

# Calculate Sharpe ratios
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

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Recommendations",
    "🔬 Backtest",
    "📊 Overview",
    "📈 Cumulative Returns",
    "🎲 Monthly Probability",
    "💰 Average Returns",
    "📉 Sharpe Ratios"
])

# TAB 1: RECOMMENDATIONS
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
            current_recs['Sharpe_Ratio'] = current_recs['ETF'].map(sharpe_results)

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
            next_recs['Sharpe_Ratio'] = next_recs['ETF'].map(sharpe_results)

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

            csv = next_recs.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"Download {next_month} Recommendations",
                csv,
                f"recommendations_{next_month.lower()}.csv",
                "text/csv"
            )
        else:
            st.warning(f"No data available for {next_month}")

    st.markdown("---")
    st.subheader("📋 Methodology")
    st.markdown(f"""
    **How recommendations are calculated:**

    1. **Composite Score** = ({weight_prob:.1f} × Probability) + ({weight_return:.1f} × Normalized Return)
    2. **Probability**: Historical percentage of positive returns for this month
    3. **Average Return**: Mean percentage return for this month across all years
    4. **Sharpe Ratio**: Overall risk-adjusted return measure

    **Note**: Past performance does not guarantee future results.
    """)

# TAB 2: BACKTEST
with tab2:
    st.header("🔬 Strategy Backtest")
    st.markdown("Test the historical performance of the recommendation strategy vs. VOO (S&P 500)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Backtest Parameters")

        bt_start = st.date_input("Start Date", value=datetime(2020, 1, 1))
        bt_end = st.date_input("End Date", value=datetime(2024, 12, 31))
        bt_lookback = st.slider("Lookback Years", 3, 10, 5)

        st.markdown("---")
        st.subheader("Strategy Parameters")
        bt_top_n = st.slider("Number of ETFs to hold", 1, 5, 1)
        bt_weight_prob = st.slider("Probability Weight", 0.0, 1.0, 0.5, 0.1)
        bt_weight_return = 1.0 - bt_weight_prob
        st.caption(f"Return Weight: {bt_weight_return:.1f}")

        run_backtest = st.button("🚀 Run Backtest", type="primary")

    with col2:
        st.subheader("Asset Classes")
        bt_include_cores = st.checkbox("Broad Core ETFs", value=True)
        bt_include_sectors = st.checkbox("Sector ETFs", value=True)
        bt_include_commodities = st.checkbox("Commodity ETFs", value=True)

        bt_tickers = []
        if bt_include_cores:
            bt_tickers.extend(BROAD_CORES)
        if bt_include_sectors:
            bt_tickers.extend(SECTORS)
        if bt_include_commodities:
            bt_tickers.extend(COMMODITIES)

        st.info(f"Total ETFs: {len(bt_tickers)}")
        st.info("📊 Benchmark: VOO (S&P 500)")

    if run_backtest:
        if not bt_tickers:
            st.error("Please select at least one asset class")
        else:
            with st.spinner("Running backtest..."):
                backtester = ETFBacktester(
                    tickers=bt_tickers,
                    start_date=bt_start,
                    end_date=bt_end,
                    lookback_years=bt_lookback
                )

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Loading historical data...")
                backtester.load_data(lambda p: progress_bar.progress(p * 0.5))

                status_text.text("Loading VOO benchmark data...")
                benchmark_loaded = backtester.load_benchmark('VOO')
                progress_bar.progress(0.6)

                if not benchmark_loaded:
                    st.warning("Could not load VOO benchmark data. Continuing without benchmark comparison.")

                status_text.text("Running backtest simulation...")
                results = backtester.run_backtest(
                    weight_prob=bt_weight_prob,
                    weight_return=bt_weight_return,
                    top_n=bt_top_n,
                    progress_callback=lambda p: progress_bar.progress(0.6 + p * 0.4)
                )

                progress_bar.empty()
                status_text.empty()

                strategy_metrics, benchmark_metrics = backtester.calculate_metrics()

                st.success("Backtest complete!")

                # Performance comparison
                st.subheader("📊 Performance Comparison: Strategy vs. VOO")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### Strategy")
                    st.metric("Total Return", f"{strategy_metrics['Total Return (%)']:.2f}%")
                    st.metric("Annualized Return", f"{strategy_metrics['Annualized Return (%)']:.2f}%")
                    st.metric("Sharpe Ratio", f"{strategy_metrics['Sharpe Ratio']:.2f}")
                    st.metric("Max Drawdown", f"{strategy_metrics['Max Drawdown (%)']:.2f}%")

                with col2:
                    st.markdown("### VOO (Benchmark)")
                    st.metric("Total Return", f"{benchmark_metrics['Total Return (%)']:.2f}%")
                    st.metric("Annualized Return", f"{benchmark_metrics['Annualized Return (%)']:.2f}%")
                    st.metric("Sharpe Ratio", f"{benchmark_metrics['Sharpe Ratio']:.2f}")
                    st.metric("Max Drawdown", f"{benchmark_metrics['Max Drawdown (%)']:.2f}%")

                with col3:
                    st.markdown("### Outperformance")
                    outperf_total = strategy_metrics['Total Return (%)'] - benchmark_metrics['Total Return (%)']
                    outperf_annual = strategy_metrics['Annualized Return (%)'] - benchmark_metrics['Annualized Return (%)']
                    outperf_sharpe = strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']
                    outperf_dd = benchmark_metrics['Max Drawdown (%)'] - strategy_metrics['Max Drawdown (%)']

                    st.metric("Total Return", f"{outperf_total:+.2f}%", 
                             delta=f"{outperf_total:.2f}%")
                    st.metric("Annualized Return", f"{outperf_annual:+.2f}%",
                             delta=f"{outperf_annual:.2f}%")
                    st.metric("Sharpe Ratio", f"{outperf_sharpe:+.2f}",
                             delta=f"{outperf_sharpe:.2f}")
                    st.metric("DD Improvement", f"{outperf_dd:+.2f}%",
                             delta=f"{outperf_dd:.2f}%", delta_color="normal")

                # Detailed metrics
                st.markdown("---")
                st.subheader("📈 Detailed Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Strategy Performance")
                    metrics_df = pd.DataFrame({
                        'Metric': list(strategy_metrics.keys()),
                        'Value': [f"{v:.2f}" if isinstance(v, (int, float)) else v for v in strategy_metrics.values()]
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("#### VOO Benchmark Performance")
                    bench_df = pd.DataFrame({
                        'Metric': list(benchmark_metrics.keys()),
                        'Value': [f"{v:.2f}" if isinstance(v, (int, float)) else v for v in benchmark_metrics.values()]
                    })
                    st.dataframe(bench_df, use_container_width=True, hide_index=True)

                # Portfolio value chart with benchmark
                st.subheader("📈 Portfolio Value Over Time: Strategy vs. VOO")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['Date'],
                    y=results['Portfolio_Value'],
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=results['Date'],
                    y=results['Benchmark_Value'],
                    name='VOO (Benchmark)',
                    line=dict(color='gray', width=2, dash='dash')
                ))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Monthly returns comparison
                st.subheader("📊 Monthly Returns: Strategy vs. VOO")

                fig2 = go.Figure()

                fig2.add_trace(go.Bar(
                    x=results['Date'],
                    y=results['Returns_%'],
                    name='Strategy',
                    marker_color='lightblue',
                    opacity=0.7
                ))

                fig2.add_trace(go.Bar(
                    x=results['Date'],
                    y=results['Benchmark_Returns_%'],
                    name='VOO',
                    marker_color='lightgray',
                    opacity=0.7
                ))

                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    barmode='group',
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig2, use_container_width=True)

                # Outperformance by month
                st.subheader("📊 Monthly Outperformance (Strategy - VOO)")

                results['Outperformance_%'] = results['Returns_%'] - results['Benchmark_Returns_%']

                fig3 = go.Figure()
                colors = ['green' if r > 0 else 'red' for r in results['Outperformance_%']]

                fig3.add_trace(go.Bar(
                    x=results['Date'],
                    y=results['Outperformance_%'],
                    marker_color=colors,
                    name='Outperformance'
                ))

                fig3.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

                fig3.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Outperformance (%)",
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig3, use_container_width=True)

                # Detailed results table
                st.subheader("📋 Detailed Results")
                st.dataframe(
                    results[['Date', 'Month', 'Holdings', 'Returns_%', 'Benchmark_Returns_%', 'Outperformance_%', 'Portfolio_Value', 'Benchmark_Value']].style.format({
                        'Returns_%': '{:.2f}',
                        'Benchmark_Returns_%': '{:.2f}',
                        'Outperformance_%': '{:.2f}',
                        'Portfolio_Value': '${:,.2f}',
                        'Benchmark_Value': '${:,.2f}'
                    }),
                    use_container_width=True
                )

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Full Results (CSV)",
                    csv,
                    "backtest_results_with_benchmark.csv",
                    "text/csv"
                )

# TAB 3: OVERVIEW
with tab3:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total ETFs Analyzed", len(all_data))
    with col2:
        st.metric("Time Period", period.upper())
    with col3:
        st.metric("Current Month", current_month)

    broad_cores_dict = {k: v for k, v in all_data.items() if k in BROAD_CORES}
    sectors_dict = {k: v for k, v in all_data.items() if k in SECTORS}
    commodities_dict = {k: v for k, v in all_data.items() if k in COMMODITIES}

    if broad_cores_dict:
        st.write("**Broad Core ETFs:**", ", ".join(broad_cores_dict.keys()))
    if sectors_dict:
        st.write("**Sector ETFs:**", ", ".join(sectors_dict.keys()))
    if commodities_dict:
        st.write("**Commodity ETFs:**", ", ".join(commodities_dict.keys()))

# TAB 4: CUMULATIVE RETURNS
with tab4:
    st.header("Cumulative Returns")

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

# TAB 5: MONTHLY PROBABILITY
with tab5:
    st.header("Monthly Probability of Positive Returns")

    st.subheader("All ETFs - Full Year")
    st.dataframe(style_dataframe_prob(all_prob), use_container_width=True)

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

# TAB 6: AVERAGE RETURNS
with tab6:
    st.header("Average Monthly Returns")

    st.subheader("All ETFs - Full Year")
    st.dataframe(style_dataframe_returns(all_avg), use_container_width=True)

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

# TAB 7: SHARPE RATIOS
with tab7:
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
st.sidebar.info("💡 Data source: Yahoo Finance via yfinance")
st.sidebar.caption("⚠️ Disclaimer: Past performance does not guarantee future results. These recommendations are for informational purposes only and should not be considered financial advice.")
