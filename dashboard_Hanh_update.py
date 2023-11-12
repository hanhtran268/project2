# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:28:16 2023

@author: Source
"""
# ==============================================================================
# Initiating
# ==============================================================================

# Libraries
import urllib
import requests
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import time

st.set_page_config(layout="wide")

# ==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
# ==============================================================================


class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

# ==============================================================================
# Header
# ==============================================================================


def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """


# Add dashboard title
st.markdown("<h1 style='text-align: center; color: black;font-size: 48px;'>MY FINANCIAL DASHBOARD</h1>",
            unsafe_allow_html=True)

col_container = st.container()
with col_container:
    st.write("<div style='font-size: 16px; text-align: center;'>Data source: YahooFinance</div>",
             unsafe_allow_html=True)
    st.write("<div style='font-size: 16px; text-align: center;'>Created by: TRAN Hanh</div>",
             unsafe_allow_html=True)

# Using CSS to adjust the mine spacing
st.markdown(
    """
    <style>
    .column-container {
        margin-bottom: 20px; 
    }
    </style>
    """,
    unsafe_allow_html=True)

# Get the list of stock tickers from S&P500
ticker_list = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Select box for 1st tab
col1, col2, col3 = st.columns([1, 1, 1])

# Ticker name
global ticker
with col1:
    ticker = st.selectbox("Stock Name", ticker_list)

# Create download and Refresh button



# Data to download
def GetStockData(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)  # Drop the indexes
    stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
    return stock_df
if ticker != '':
    # Create start date and end date for download
    with col2:
        start_date_to_download = st.date_input(
            "Start date to Download", datetime.today().date() - timedelta(days=30))
    with col3:
        end_date_to_download = st.date_input(
            "End date to Download", datetime.today().date())
    stock_price = GetStockData(
        ticker, start_date_to_download, end_date_to_download)

# Convert the DataFrame to CSV and create download button
csv_data = stock_price.to_csv(index=False)
col1, col2 = st.columns([1, 3])
with col1:
    st.download_button("Download", data=csv_data,
                       file_name='stock_data.csv', key='csv')

# Progress button
with col2:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

# ==============================================================================
# Tab 1
# ==============================================================================


def render_tab1():

    # GET THE COMPANY INFOR
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return YFinance(ticker).info

    # CREATE COLUMN STRUCTURE
    col1, col2 = st.columns([1, 2])

    # If the ticker is already selected
    if ticker != '':

        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
    # If the ticker is already selected,get data for "Earning Date", "Ex-Dividend Date"
    if ticker != '':

        # Get the company information in list format
        stock = yf.Ticker(ticker)
        #earnings_info = stock.earnings
        dividends_info = stock.dividends
        

        # SHOW SOME STATISTICS AS DATAFRAME

        with col1:
            # Statistic in yfinance.info
            st.subheader("Data Summary")
            info_keys = {'previousClose': 'Previous Close',
                         'open': 'Open',
                         'bid': 'Bid',
                         'ask': 'Ask',
                         '52WeekChange': '52 Week Change',
                         'volume': 'Volume',
                         'averageVolume': 'Avg. Volume',
                         'marketCap': 'Market Cap',
                         'beta': 'Beta (5Y Monthly)',
                         'trailingPE': 'PE Ratio (TTM)',
                         'trailingEps': 'EPS (TTM)',
                         'dividendRate': 'Forward Dividend & Yield',
                         'targetMeanPrice': '1y Target Est',
                         'Earnings Date' : 'Earnings Date'}
            # Create a new key for 'Days Range', '52 Week Range'
            fifty_range_low = info.get('fiftyTwoWeekLow', 'N/A')
            fifty_range_high = info.get('fiftyTwoWeekHigh', 'N/A')
            day_range_low = info.get('dayLow', 'N/A')
            day_range_high = info.get('dayHigh', 'N/A')
            
            # Create a new key for "Earning Date", "Ex-Dividend Date"
            #earnings_date = info.iloc[0]['Earnings Date'] if not info.empty else 'N/A'
            ex_dividend_date = dividends_info.index[-1].strftime('%Y-%m-%d') if not dividends_info.empty else 'N/A'

            company_stats = {}  # Dictionary

            # Fill the company_stats dictionary
            for key, value in info_keys.items():
                company_stats.update({value: info.get(key, 'N/A')})
            
            # Add new values to the dictionary
            company_stats["52 Week Range"] = f"{fifty_range_low} - {fifty_range_high}"
            company_stats["Day's Range"] = f"{day_range_low} - {day_range_high}"
            company_stats["Ex-Dividend Date"] = f"{ex_dividend_date}"
            #company_stats["Earnings Date"] = f"{earnings_date}"

            # Convert the dictionary to a DataFrame
            company_stats_df = pd.DataFrame({'Statistic': list(
                company_stats.keys()), 'Value': list(company_stats.values())})
            company_stats_df.columns = ["Statistic", "Value"]
         
            # Display the DataFrame
            st.dataframe(company_stats_df,width=0, height=430, use_container_width=True,hide_index=True)

    # AREA CHART FOR STOCK PRICE
    with col2:
        # Create duration filter
        st.subheader("Stock Price")
        duration = st.selectbox(
            "Select Duration", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])

        # Function to get data for each duration
        @st.cache_data
        def GetStockData(ticker, duration):
            end_date = datetime.now()
            if duration == "1M":
                start_date = end_date - timedelta(days=30)
            elif duration == "3M":
                start_date = end_date - timedelta(days=90)
            elif duration == "6M":
                start_date = end_date - timedelta(days=180)
            elif duration == "YTD":
                start_date = datetime(end_date.year, 1, 1)
            elif duration == "1Y":
                start_date = end_date - timedelta(days=365)
            elif duration == "3Y":
                start_date = end_date - timedelta(days=3*365)
            elif duration == "5Y":
                start_date = end_date - timedelta(days=5*365)
            else:
                start_date = None
            stock_df = yf.Ticker(ticker).history(
                start=start_date, end=end_date, interval="1d")
            stock_df.reset_index(inplace=True)
            stock_df['Date'] = stock_df['Date'].dt.date
            return stock_df

        # If the ticker name is selected, show chart
        if ticker != '':
            stock_price = GetStockData(ticker, duration)
            st.area_chart(data=stock_price,
                          x='Date',
                          y='Close',
                          color="#29AB87",
                          width=0, height=400,
                          use_container_width=True)

    # COMPANY PROFILE

    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        # General Information
        st.subheader("General Information")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div style="text-align: justify;">' +
                        info['address1'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        info['city'] + ' ' +
                        info['state'] + ' ' +
                        info['zip'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        info['country'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        info['phone'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        info['website'] +
                        '</div><br>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="text-align: justify;">' +
                        'Sector(s): ' +
                        info['sector'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        'Industry: ' +
                        info['industry'] +
                        '</div><br>',
                        unsafe_allow_html=True)
            st.markdown('<div style="text-align: justify;">' +
                        'Full Time Employees: ' +
                        str(info['sector']) +
                        '</div><br>',
                        unsafe_allow_html=True)
        # Key Executives
        st.subheader("Key Executives")
        company_officers = info['companyOfficers']

        col1, col2, col3, col4, col5 = st.columns(([2, 3, 1, 1, 1]))
        col1.markdown(
            '<div style="font-size: 16px; color: grey; border-bottom: 1px solid grey; padding-bottom: 5px;">Name</div>',
            unsafe_allow_html=True)
        col2.markdown(
            '<div style="font-size: 16px; color: grey; border-bottom: 1px solid grey; padding-bottom: 5px;">Title</div>',
            unsafe_allow_html=True)
        col3.markdown(
            '<div style="font-size: 16px; color: grey; border-bottom: 1px solid grey; padding-bottom: 5px;">Total Pay</div>',
            unsafe_allow_html=True)
        col4.markdown(
            '<div style="font-size: 16px; color: grey;border-bottom: 1px solid grey; padding-bottom: 5px;">Exercised</div>',
            unsafe_allow_html=True)
        col5.markdown(
            '<div style="font-size: 16px; color: grey;border-bottom: 1px solid grey; padding-bottom: 5px;">Year Born</div>',
            unsafe_allow_html=True)

        for officer in company_officers:
            with col1:
                st.write(officer.get('name', 'N/A'))
            with col2:
                st.write(officer.get('title', 'N/A'))
            with col3:
                total_pay_data = officer.get('totalPay', {})
                total_pay_raw = str(total_pay_data.get('raw', 'N/A'))
                st.write(total_pay_raw)
            with col4:
                exercisedValue_data = officer.get('exercisedValue', {})
                exercisedValue_raw = str(exercisedValue_data.get('raw', 'N/A'))
                st.write(exercisedValue_raw)
            with col5:
                st.write(str(officer.get('yearBorn', 'N/A')))

    # COMPANY DESCRIPTION
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        st.subheader("Company Description")
        st.markdown('<div style="text-align: justify;">' +
                    info['longBusinessSummary'] +
                    '</div><br>',
                    unsafe_allow_html=True)

    # MAJOR HOLDERS
    # If the ticker is already selected
    if ticker != '':
        # Get the company information about their shareholders
        major_holder = pd.DataFrame(yf.Ticker(ticker).major_holders)
        major_holder.columns = ["Percentage", "Statistic"]
        

        # Set a title
        st.subheader("Major Holders")

        # Convert to list
        col1s, col2s = major_holder.columns
        list1 = major_holder[col1s].to_list()
        list2 = major_holder[col2s].to_list()
        list3 = [{col1s: val1, col2s: val2}
                 for val1, val2 in zip(list1, list2)]

        col1, col2 = st.columns(([1, 1]))
        col1.markdown(
            '<div style="font-size: 16px; color: grey; border-bottom: 1px solid grey; padding-bottom: 5px;">Percentage</div>',
            unsafe_allow_html=True)
        col2.markdown(
            '<div style="font-size: 16px; color: grey; border-bottom: 1px solid grey; padding-bottom: 5px;">Statistic</div>',
            unsafe_allow_html=True)
        for line in list3:
            with col1:
                st.write(line.get(col1s, 'N/A'))
            with col2:
                st.write(line.get(col2s, 'N/A'))

# ==============================================================================
# Tab 2
# ==============================================================================


def render_tab2():

    # Create filters
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    start_date = col1.date_input(
        "Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    duration = col3.selectbox(
        "Duration", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
    time_interval = col4.selectbox("Interval", ["1d", "1mo", "1y"])
    plot_type = col5.selectbox("Plot Type", ["Line Plot", "Candle Plot"])

    # Function to get data for each filter
    @st.cache_data
    def GetStockData(ticker, start_date, end_date, duration, time_interval):
        if duration == "1M":
            start_date = end_date - timedelta(days=30)
        elif duration == "3M":
            start_date = end_date - timedelta(days=90)
        elif duration == "6M":
            start_date = end_date - timedelta(days=180)
        elif duration == "YTD":
            start_date = datetime(end_date.year, 1, 1)
        elif duration == "1Y":
            start_date = end_date - timedelta(days=365)
        elif duration == "3Y":
            start_date = end_date - timedelta(days=3*365)
        elif duration == "5Y":
            start_date = end_date - timedelta(days=5*365)

        stock_df = yf.Ticker(ticker).history(
            start=start_date, end=end_date, interval=time_interval)
        stock_df.reset_index(inplace=True)
        stock_df['Date'] = stock_df['Date'].dt.date
        return stock_df

    # If the ticker name is selected, get data
    if ticker != '':
        stock_price = GetStockData(
            ticker, start_date, end_date, duration, time_interval)

    # Plotting the selected chart type for the Stock price
    from plotly.subplots import make_subplots

    if stock_price is not None and not stock_price.empty:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        if plot_type == "Line Plot":
            price = go.Scatter(
                x=stock_price['Date'],
                y=stock_price['Close'],
                mode='lines', name='Stock Price')
        elif plot_type == "Candle Plot":
            price = go.Candlestick(
                x=stock_price['Date'],
                open=stock_price['Open'],
                high=stock_price['High'],
                low=stock_price['Low'],
                close=stock_price['Close'], name='Stock Price')

    # Determine bar colors based on volume changes
        stock_price['VolumeChange'] = stock_price['Volume'].diff()
        bar_colors = np.where(stock_price['VolumeChange'] > 0, 'green', 'red')

    # Plotting the bar chart for the Volume
        volume = go.Bar(
            x=stock_price['Date'],
            y=stock_price['Volume']/1000000,
            name='Volume',
            marker_color=bar_colors)

    # Add traces to the subplot grid
        fig.add_trace(price, row=1, col=1)
        fig.add_trace(volume, row=1, col=1, secondary_y=False)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_yaxes(dtick=20, range=[
                         0, stock_price['Close'].max()], row=1, col=1)
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_layout(height=650)

    # Add another subplot: line chart for MA
        fig.add_trace(go.Scatter(
            x=stock_price['Date'],
            y=stock_price['Close'].rolling(window=50, min_periods=1).mean(),
            mode='lines',
            name='Moving Average', line=dict(color='purple')), row=1, col=1)

    # Display the plot
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Tab 3
# ==============================================================================


def render_tab3():

    # Create filters for Financial Infor
    col1, col2 = st.columns([1,5])
    with col1:
        fin_statement = st.radio('Statement:', [
                                     'Income Statement', 'Balance Sheet', 'Cash Flow'])
        fin_period = st.radio('Period:', ['Annual', 'Quarterly'])
        # If the ticker name is selected, get data
        if ticker != '':
            stock = yf.Ticker(ticker)

    # Function to get data for each statement
    @st.cache_data
    def GetFinancialInfo(_stock, fin_statement, period):
        if period == 'Annual':
            if fin_statement == 'Income Statement':
                fin_data = _stock.financials
            elif fin_statement == 'Balance Sheet':
                fin_data = _stock.balance_sheet
            elif fin_statement == 'Cash Flow':
                fin_data = _stock.cashflow
            else:
                fin_data = None
        if period == 'Quarterly':
            if fin_statement == 'Income Statement':
                fin_data = _stock.quarterly_financials
            elif fin_statement == 'Balance Sheet':
                fin_data = _stock.quarterly_balance_sheet
            elif fin_statement == 'Cash Flow':
                fin_data = _stock.quarterly_cashflow
            else:
                fin_data = None
        return fin_data

    # Get the data for selected fin_statement
    fin_data = GetFinancialInfo(stock, fin_statement, fin_period)
    fin_data2 = GetFinancialInfo(stock, fin_statement, 'Quarterly')

    # Create TTM column
    last_four_quarters2 = fin_data2.iloc[:, -4:]
    ttm_values = last_four_quarters2.sum(axis=1)

    # Display the dataframe
    with col2:
        if fin_data is not None and not fin_data.empty:
            fin_data.columns = fin_data.columns.date
            fin_data['TTM'] = ttm_values
            st.dataframe(fin_data,width=0, height=600,use_container_width=True)
        else:
            st.write(f'No data available for {fin_statement} of {ticker}')

# ==============================================================================
# Tab 4
# ==============================================================================


def render_tab4():

    # MONTE CARLO SIMULATION
    def MC_simulation(ticker, n_simulations, time_horizon):
        stock_price = yf.Ticker(ticker).history(
            start=datetime.now() + timedelta(days=-time_horizon), end=datetime.now())
        close_price = stock_price['Close']
        daily_return = close_price.pct_change()
        daily_volatility = daily_return.std()

        simulated_prices = []
        simulated_df = pd.DataFrame()
        for rnd in range(n_simulations):
            current_price = stock_price['Close'][-1]
            simulated_price = []
            for i in range(time_horizon):
                simulated_daily_return = np.random.normal(0, daily_volatility)
                future_price = current_price * (1 + simulated_daily_return)
                simulated_price.append(future_price)
                current_price = future_price

            simulated_prices.append(simulated_price)
            simulated_df.loc[:, rnd] = pd.Series(simulated_price)
    # Create simulation traces
        traces = []
        for i in range(n_simulations):
            trace = go.Scatter(x=list(range(
                time_horizon)), y=simulated_prices[i], mode='lines', name=f'Simulation {i+1}')
            traces.append(trace)

    # Create and show the plot
        st.subheader("Monte Carlo simulation")
        df = go.Figure(data=traces)
        df.update_layout(xaxis_title='Days', yaxis_title='Price', height=600)
        st.plotly_chart(df,use_container_width=True)

    # VALUE AT RISK
    # Plotting the histogram
        # Create histogram trace
        histogram_trace = go.Histogram(x=simulated_df.iloc[-1, ], nbinsx=30, opacity=0.7, name='Simulated Prices')

        # Create vertical line for median
        median_trace = go.Scatter(x=[simulated_df.iloc[-1, ].median(), simulated_df.iloc[-1, ].median()],
                                  y=[0, 1], mode='lines', line=dict(color='green', dash='dash'), name='Median')

        # Create vertical line for VaR (5%)
        var_trace = go.Scatter(x=[np.percentile(simulated_df.iloc[-1, ], 5), np.percentile(simulated_df.iloc[-1, ], 5)],
                              y=[0, 1], mode='lines', line=dict(color='red', dash='dash'), name='VaR (5%)')

        # Create layout
        layout = go.Layout(title='Value at Risk (VaR) at 95%',
                           xaxis=dict(title='Stock Price'),
                           yaxis=dict(title='Frequency'))
        # Create figure
        fig = go.Figure(data=[histogram_trace, median_trace, var_trace], layout=layout)

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

    # Dropdown widgets for selecting number of simulations and time horizon
    col1, col2 = st.columns([1,1])
    n_simulations = col1.selectbox(
        'Number of Simulations:', [200, 500, 1000])
    time_horizon = col2.selectbox('Time Horizon (days):', [30, 60, 90])

    # Function to update the simulation plot based on dropdown values
    if ticker != '':
        MC_simulation(ticker, n_simulations, time_horizon)


# ==============================================================================
# Main body
# ==============================================================================

# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Company profile", "Chart", "Financial", "Monte Carlo Simulation"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()


# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #FFFFFF;
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# END
###############################################################################
