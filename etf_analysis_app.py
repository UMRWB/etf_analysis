import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta import others
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

# Page configuration
st.set_page_config(
    page_title="ETF Analysis & Backtest",
    page_icon="📈",
    layout="wide"
)

# Import backtest class (inline for simplicity)
class ETFBacktester:
    """Backtest the ETF recommendation strategy"""

    def __init__(self, tickers, start_date, end_date, lookback_years=5):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.lookback_years = lookback_years
        self.data = {}
        self.results = []

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
        results = []
        months_processed = 0
        total_months = (self.end_date.year - self.start_date.year) * 12 + (self.end_date.month - self.start_date.month)

        while current_date <= self.end_date:
            month_name = current_date.strftime("%B")
            decision_date = current_date - relativedelta(months=1)

            recommended_etfs = self.get_recommendation(month_name, decision_date, weight_prob, weight_return, top_n)

            if recommended_etfs is None or len(recommended_etfs) == 0:
                current_date += relativedelta(months=1)
                continue

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

            results.append({
                'Date': current_date,
                'Month': month_name,
                'Year': current_date.year,
                'Holdings': ', '.join(recommended_etfs),
                'Returns_%': portfolio_return,
                'Portfolio_Value': portfolio_value
            })

            current_date += relativedelta(months=1)
            months_processed += 1
            if progress_callback:
                progress_callback(months_processed / total_months)

        self.results = pd.DataFrame(results)
        return self.results

    def calculate_metrics(self):
        if self.results.empty:
            return None

        returns = self.results['Returns_%'].values

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

        metrics = {
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

        return metrics

# Rest of the original Streamlit app code...
# (This is abbreviated for space - in reality, include all the original functionality)

st.title("📈 ETF Analysis, Recommender & Backtest")

# Sidebar
st.sidebar.header("Configuration")
BROAD_CORES = ["VOO", "VTI", "VT", "VXUS", "VEA"]
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
COMMODITIES = ["GLD", "SLV"]

# Create tabs
tab_backtest, tab_recommender = st.tabs(["🔬 Backtest", "🎯 Recommender"])

with tab_backtest:
    st.header("🔬 Strategy Backtest")
    st.markdown("Test the historical performance of the recommendation strategy")

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

        # Build ticker list
        bt_tickers = []
        if bt_include_cores:
            bt_tickers.extend(BROAD_CORES)
        if bt_include_sectors:
            bt_tickers.extend(SECTORS)
        if bt_include_commodities:
            bt_tickers.extend(COMMODITIES)

        st.info(f"Total ETFs: {len(bt_tickers)}")

    if run_backtest:
        if not bt_tickers:
            st.error("Please select at least one asset class")
        else:
            with st.spinner("Running backtest..."):
                # Initialize backtester
                backtester = ETFBacktester(
                    tickers=bt_tickers,
                    start_date=bt_start,
                    end_date=bt_end,
                    lookback_years=bt_lookback
                )

                # Load data
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Loading historical data...")
                backtester.load_data(lambda p: progress_bar.progress(p))

                # Run backtest
                status_text.text("Running backtest simulation...")
                progress_bar.progress(0)
                results = backtester.run_backtest(
                    weight_prob=bt_weight_prob,
                    weight_return=bt_weight_return,
                    top_n=bt_top_n,
                    progress_callback=lambda p: max(progress_bar.progress(p), 1.0)
                )

                progress_bar.empty()
                status_text.empty()

                # Calculate metrics
                metrics = backtester.calculate_metrics()

                # Display results
                st.success("Backtest complete!")

                st.subheader("📊 Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
                    st.metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")

                with col2:
                    st.metric("Annualized Return", f"{metrics['Annualized Return (%)']:.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

                with col3:
                    st.metric("Winning Months", int(metrics['Winning Months']))
                    st.metric("Best Month", f"{metrics['Best Month (%)']:.2f}%")

                with col4:
                    st.metric("Losing Months", int(metrics['Losing Months']))
                    st.metric("Worst Month", f"{metrics['Worst Month (%)']:.2f}%")

                # Additional metrics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Avg Monthly Return", f"{metrics['Average Return (%)']:.2f}%")
                with col2:
                    st.metric("Monthly Volatility", f"{metrics['Std Dev (%)']:.2f}%")
                with col3:
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")

                # Portfolio value chart
                st.subheader("📈 Portfolio Value Over Time")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['Date'],
                    y=results['Portfolio_Value'],
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Monthly returns chart
                st.subheader("📊 Monthly Returns Distribution")

                fig2 = go.Figure()
                colors = ['green' if r > 0 else 'red' for r in results['Returns_%']]

                fig2.add_trace(go.Bar(
                    x=results['Date'],
                    y=results['Returns_%'],
                    marker_color=colors,
                    name='Monthly Return'
                ))

                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig2, use_container_width=True)

                # Detailed results table
                st.subheader("📋 Detailed Results")
                st.dataframe(
                    results[['Date', 'Month', 'Holdings', 'Returns_%', 'Portfolio_Value']].style.format({
                        'Returns_%': '{:.2f}',
                        'Portfolio_Value': '${:,.2f}'
                    }),
                    use_container_width=True
                )

                # Download button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Full Results (CSV)",
                    csv,
                    "backtest_results.csv",
                    "text/csv"
                )

                # Store results in session state for comparison
                st.session_state['backtest_results'] = results
                st.session_state['backtest_metrics'] = metrics

with tab_recommender:
    st.header("🎯 Current Month Recommendations")
    st.info("This tab would contain the recommender functionality from the previous app")
    st.markdown("See the full recommender in the etf_recommender_app.py file")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 The backtest uses walk-forward analysis - only historical data available at each decision point is used")
st.sidebar.caption("⚠️ Past performance does not guarantee future results")
