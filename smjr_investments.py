import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import math
import sys
from datetime import timedelta #pip
import mplfinance as mpf #pip
import seaborn as sns #don't need but pip anyway
from matplotlib.dates import DateFormatter 

# Page Configuration
st.set_page_config(
    page_title="SMJR Investments", # Page title
    layout="wide", # Making layout wide
    initial_sidebar_state="expanded" # Sidebar setting
)

# Design using markdown
st.markdown(
    """
    <style>
    /* Force light grey background */
    body, .main {
        background-color: #f0f0f0 !important; /* Forcing a light grey background */
        color: #333333 !important; /* Forcing text to be dark grey */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; /* Font input */
    }

    /* Sidebar Design */
    section[data-testid="stSidebar"] {
        background-color: #d3d3d3 !important; /* Forcing a grey background */
        color: #333333 !important; /* Dark grey text */
    }

    /* Sidebar Text */
    section[data-testid="stSidebar"] label, /* Label text in the sidebar */
    section[data-testid="stSidebar"] .css-1cpxqw2 { /* General sidebar text */
        color: #333333 !important; /* Dark grey text in sidebar for labels and general text */
    }

    /* Sidebar Input Boxes */
    section[data-testid="stSidebar"] .css-1x8cf1d, /* Input box container */
    section[data-testid="stSidebar"] input { /* Input fields */
        background-color: #ffffff !important; /* White background for input fields */
        color: #333333 !important; /* Dark grey text */
        border: 1px solid #cccccc !important; /* Light grey border */
        border-radius: 5px !important; /* Rounded corners */
    }

    /* Sidebar Dropdown Styling */
    section[data-testid="stSidebar"] select { /* Dropdown menu */
        background-color: #ffffff !important; /* White background */
        color: #333333 !important; /* Dark grey text */
        border: 1px solid #cccccc !important; /* Light grey border */
    }
    section[data-testid="stSidebar"] option { /* Options within dropdowns */
        background-color: #ffffff !important; /* White background for options */
        color: #333333 !important; /* Dark grey text */
    }

    /* Header Colors */
    h1, h2, h3, h4 { 
        color: #333333 !important; /* Forcing headers to be dark grey */
    }

    /* Metric Container Styling */
    div[data-testid="metric-container"] {
        background-color: #f5f5f5 !important; /* Light grey background */
        border-radius: 8px !important; /* Rounded corners */
        padding: 10px !important; /* Add spacing within the metric container */
    }

    /* Metric Value Styling */
    div[data-testid="metric-container"] > div:nth-child(2) { /* Metric value */
        color: #2ca02c !important; /* Green for positive values */
        font-size: 18px !important; /* Larger font size for emphasis */
        font-weight: bold !important; /* Bold font */
    }

    /* Metric Delta Styling */
    div[data-testid="metric-container"] > div:nth-child(3) { /* Metric delta */
        color: #ff5733 !important; /* Use red for deltas (differences) to indicate change */
        font-size: 14px !important; /* Medium font size */
    }

    /* DataFrame Styling */
    .stDataFrame {
        background-color: #ffffff !important; /* White background for tables */
        border-radius: 10px !important; /* Slightly rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("<h1 style='text-align: center;'>SMJR Investments - Portfolio Analysis</h1>", unsafe_allow_html=True) # Title with heading setting
st.markdown("---") # Line underneath for presentation
st.markdown("> #### <span style='color: #00bfae;'>*Please click on the sidebar to start your portfolio analysis!*</span>", unsafe_allow_html=True) # 4th biggest size with instructions


#Formatting numbers to round easier
def format_number(num):
    """This function formats any numbers up to the billions,
    If the number can be perfectly divided (no remainder) by a billion/million/thousand then we simply return the number divided by a billion/million/thousand and concatenate its corresponding letter (i.e B for Billion)
    If the number cannot be perfectly divided (remainder present) by a billion/million/thousand then we simply return that number, divided by a billion/million/thousand rounded to 2 numbers/dcps and concatenate its corresponding letter (i.e B for Billion)
    If the number is less or equal to a thousand we simply return the number rounded by two decimal places
    """
    if num >= 1000000000:
        if not num % 1000000000:
            return f"{num // 1000000000} B"
        return f"{round(num/ 1000000000,2)} B"
    elif (num >= 1000000) and (num < 1000000000):
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 2)} M'
    elif (num >= 1000) and (num < 1000000):
        if not num % 1000:
            return f'{num // 1000} K'
        return f'{round(num/1000,2)} K'
    elif num <= 1000:
        return round(num,2)
    
first_graph = False #Want to display portfolio graph before individual analysis graph


def weight_sum_validator(weight_list): #Weight Validator that will be used later
        """This function adds up all the items (assumed numeric as streamlit number func only allows numeric) of a list,
        These items are between 0 and 1.
        It then returns True if the sum of these items is 1
        It returns False if the sum of these items is not 1 as well as writing to the sidebar that the corresponding weight_list does not sum to 1, with the individual items shown.
        """
        if math.isclose(sum(weight_lst), 1.0, abs_tol=1e-9): #Combats binary form of numbers
            return True
        if sum(weight_list) != 1:
            st.sidebar.write(f"{weight_list} does not sum to 1")
            return False
        

        
ticker_list = [] #Initializing empty list for appending later...
df = pd.DataFrame()

tickers_csv = pd.read_csv("nasdaq_screener_1732110032769.csv",dtype='object') #CSV File with almost all tickers on yfinance


for item in tickers_csv['Symbol']:
    ticker_list.append(item)
    


    
#Model Assumptions:
#Tickers are entered in correctly
#Users correctly enters weights that add up to 1
#User adds in correct number of weights corresponding to number of stocks (i.e., does not leave blank)




cols = st.columns([2,1],gap="medium")
with cols[0]:
    st.markdown("### Portfolio Analysis")
#Enter initial investment

st.sidebar.markdown("### Inputs")
initial_investment = st.sidebar.number_input("Initial Investment: ",min_value=0.0,max_value=10000000000.0)


#This part takes in the user's stocks and their respective weights
no_stocks = st.sidebar.number_input("Number of stocks in portfolio: ",min_value=0,step=1, max_value=15)
stock_lst =[] #Could call these back to the user so they know the exact stocks
weight_lst=[]
invalid_tickers = []

if (initial_investment != 0) and (no_stocks != 0):
    for num in range(no_stocks): #This part here is a bit new
        ky1 =  f"ticker_{num}" #Each input requires a unique key associated with it, so by convenience I am using the same number as the iteration
        ky2 = f"weight_{num}"
        c1,c2 = st.sidebar.columns(2) #c1 and c2 are going to be associated with two columns of size one by one side by side
        with c1: 
            option = c1.text_input(f"Ticker",key = ky1).upper() #Relate it to c1 (first) and specify input type and key
            if (option != '') and (option not in ticker_list): ##Making sure that inputted tickers are valid, then appending them to a removal list for later
                invalid_tickers.append(option)
                st.sidebar.write(f"{option} is an invalid ticker.")
                break
            else:
                stock_lst.append(option)
            
        with c2:
            weight = c2.number_input(f"Weighting",min_value=0.0,max_value=1.0,key = ky2,) #Note if max_value = 1, then it assumes integer
            if weight != 0:
                #Weights are rounded to 2dcps
                weight_lst.append(round(weight,2))
            
        
        for item in stock_lst: ##Removing all invalid items from stock_lst
            if item in invalid_tickers:
                stock_lst.remove(item)
        


    

    if len(weight_lst) != 0 and (len(weight_lst) == no_stocks): ##Using weight_sum_validator() if and only if there have been weights inputted
    ## The if statement below is quite confusing but step by step it is,
    #Step 1: Make sure through the use of the weight_sum_validator that the sum of weights equals 1
    #Step 2: Make sure that none of the Ticker entries are blank 
    #Step 3: Make sure the length of the stock_lst with all valid stocks is equal to the user's inputted number of stocks
        if (weight_sum_validator(weight_lst) == True) and (stock_lst.count('') == 0) and (len(stock_lst) == no_stocks):
            if len(set(stock_lst)) != len(stock_lst):
                st.sidebar.write("You have two identical tickers inputted")
            else:
                date_lst = []
                #Below code iterates over each stock chosen, finds the oldest date we can get info for it from yfinance, appends said date to a list
                for stock in stock_lst:
                        all_data = yf.Ticker(stock).history(period = 'max')
                        try:
                            min_date = all_data.iloc[0].name.date()
                            date_lst.append(min_date)
                            success = True
                        except IndexError:
                            st.sidebar.write(f"{stock} possibly delisted, please change to a valid stock")
                            success = False
                            break
                                

                        
                    
                
                if success is True:
                    if all_data.index.tz is not None:  # If the index has timezone info
                        all_data.index = all_data.index.tz_localize(None)
                
                    #We then sort the dates for which we can start retrieving info from stocks on yfinance (in descending order), we then pick the newest date and make this the minimum value
                    sorted_dates = sorted(date_lst,reverse=True)
                    st.sidebar.write("Please Note the Minimum Period is 3 Months")
                    
                    end_date = st.sidebar.date_input("End Date: ",max_value=datetime.datetime.now())
                    start_date = st.sidebar.date_input("Start Date: ",min_value=sorted_dates[0],max_value=end_date-timedelta(days=100)) 
                    
                    if start_date != end_date:
                        if (start_date.year == end_date.year) and ((end_date.month - start_date.month < 3) or (end_date.month - start_date.month == 3 and start_date.day > end_date.day)): ##Making sure end-dates are minimum of 3 months away, have to make sure the months are atleast 3 away and then that the days are the same or less
                            st.sidebar.write("Period too short")

                        else:
                            stock_df = pd.DataFrame()
                            weighted_rs = pd.DataFrame()
                            statistics_df = pd.DataFrame()
                            

                        #Getting Columns for your return if you just put all of initial_investment into either stocks
                            for stock in stock_lst: 
                                price_1 = yf.download(stock,start_date,end_date,interval="1wk")['Adj Close'] # 1.) First get Adj Close for each stock via yf
                                statistics_df[stock] = price_1
                                price = price_1/price_1.iloc[0] # Find the normed return of each stock by dividing it by its initial value 
                                price = price*initial_investment
                                stock_df[stock] = price # 2.) Add each of these to stock_df dataframe

                                

                        # Getting the weighted returns of the stocks using the zip function
                        # Zip function here essentially assigns each stock to its given weight
                            for stock,weight in zip(stock_lst,weight_lst): 
                                weighted_rs[f"{stock} Weighted Returns"] = stock_df[stock]*weight

                        #Now using the iloc functions like .iloc[:,columns] first ':' is just essentially saying 'pick all rows' then second argument are our particular columns
                        #I am setting the index to the number of stocks as the first 'n' columns in our df will just be the individual prices of the stocks, we only want the columns that have the weighted returns
                        #Then i am picking all these weighted returns columns and summing them up for the total portfolio value
                            index = no_stocks
                            stock_df["Total Portfolio Value"] = weighted_rs.sum(axis=1)
                            

                            
                            overlay = st.sidebar.checkbox("Graph Returns if Invested in Individual Stocks")
                            SandP = st.sidebar.checkbox("Compare against the S&P500")
                            
                            sp = yf.download("^SPX",start_date,end_date,interval="1wk")['Adj Close']
                            sp = sp/sp.iloc[0]
                            sp = sp*initial_investment
                            stock_df["S&P 500"] = sp
                            sp_return = ((stock_df["S&P 500"].iloc[-1] - stock_df["S&P 500"].iloc[0])/stock_df["S&P 500"][0])*100


                            #Defining a create_plot() function to avoid repetition, takes in a dataframe and a legend, all graphs go to the graph_container
                            def create_plot(df,legend):
                                """This function creates a base graph that is used for showing portfolio performance, with the date range as the x-axis and the share price which is the portfolio value as the y axis
                                It takes in a dataframe (which will be a pandas df in our case) as the first argument
                                It takes a legend as its second argument (which is positioned in the upper left)
                                Its primary use is to avoid repetition
                                It then plots a pyplot onto streamlit
                                """
                                fig2,ax6 = plt.subplots(figsize=(18,10))
                                sns.lineplot(data=df,linewidth=3.0,ax=ax6)
                                plt.title("Portfolio Value Over Time",fontsize=24)
                                ax6.set_xlabel('Date Range',fontsize=14)
                                ax6.set_ylabel('Share Price',fontsize=14)
                                for x in ax6.lines:
                                    x.set_linestyle('solid')
                                plt.legend(legend, loc='upper left', fontsize=25)
                                plt.xticks(rotation=0)
                                ax6.set_yticklabels([f'${label:.0f}' for label in ax6.get_yticks()],fontsize=10)
                                st.pyplot(fig2)
                                    

                            with cols[0]:
                                graph_container = st.container()

                                if overlay and SandP:
                                    #Here, since they want to both overlay invdividual stocks and s&p we can just plot the entire df
                                    create_plot(stock_df,stock_df.columns)
                                    first_graph=True
                                if not overlay and SandP:
                                    #Here since they don't want to overlay but want s&p (which is the very last col of df), we use iloc[:,no_stocks:] to get all rows and all cols after the individual stock columns
                                    create_plot(stock_df.iloc[:,no_stocks:],["Total Portfolio Value","S&P 500"])
                                    first_graph=True
                                if overlay and not SandP:
                                    #Here our chosen columns are every one apart from the last one as this is the one with S&P
                                    column_names = stock_df.columns[:-1]
                                    create_plot(stock_df.iloc[:,:-1],column_names)
                                    first_graph=True
                                if not overlay and not SandP:
                                    #Simply plotting TPV column with the legend being the single element of the "Total Portfolio Value"
                                    create_plot(stock_df['Total Portfolio Value'],['Total Portfolio Value'])
                                    first_graph=True
                                inner_cols = st.columns([1,1,1])
                                final=stock_df["Total Portfolio Value"].iloc[-1]  
                                start=stock_df["Total Portfolio Value"].iloc[0]
                                portfolio_return = (final)
                                portfolio_return_pct = ((portfolio_return-start)/start)*100
                                with inner_cols[0]:
                                    st.metric(label="Portfolio Return",value=f"${format_number(portfolio_return)}",delta=f"{round(portfolio_return_pct,2)}%")
                                
                        
                                
                                #Here I am initiating an empty beta list, then getting the beta's of the stocks from yfinance and appending them to the list
                                beta_lst = []
                                for stock in stock_lst:
                                    info = yf.Ticker(stock).info
                                    beta = info.get('beta')
                                    if beta is not None:
                                        beta_lst.append(beta)
                                    else:
                                        st.write(f"{stock} does not have a known beta, unable to find portfolio beta")
                                        

                                
                                #np.dot() function just gets the dot product of the two lists
                                #I.e in the case of two stocks,
                                #portfolio_beta = w1*b1 + w2*b2 which is dot product
                                if len(beta_lst) == no_stocks:
                                    portfolio_beta = np.dot(weight_lst,beta_lst)
                                    with inner_cols[1]:
                                        st.metric(label="Portfolio Beta",value=f"{portfolio_beta:.2f}")

                                #Getting pct_returns for stocks for finding variance in percentage(squared)
                                for stock in stock_lst:
                                    statistics_df[stock] = statistics_df[stock].pct_change()
                                
                                #Getting rid of first row of pct_change data as it will be NaN
                                statistics_df_ignoredrow = statistics_df.iloc[1:]

                                
                                
                                #Using a formula where variance of portfolio = weights.transposed * cov_array * weights
                                weight_arr = np.array(weight_lst)
                                cov_arr = np.array(statistics_df_ignoredrow.cov())
                                weight_transposed_array = weight_arr.T
                                portfolio_variance = np.dot(weight_transposed_array,np.dot(cov_arr,weight_arr))
                                portfolio_sd = math.sqrt(portfolio_variance)

                                with inner_cols[2]:
                                    st.metric(label="Portfolio Volatility",value=f"{portfolio_sd*100:.2f}%")

                                stock_returns = []
                                weighted_rs_lst = []
                                
                                for stock in stock_lst:
                                    weighted_rs_returns = weighted_rs[f"{stock} Weighted Returns"].iloc[-1]
                                    returns = ((stock_df[stock].iloc[-1] - stock_df[stock].iloc[0])/stock_df[stock].iloc[0])*100
                                    weighted_rs_lst.append(weighted_rs_returns)
                                    stock_returns.append(returns)
                                
                                returns_dict = {k: (v1,v2) for k,v1,v2 in zip(stock_lst,stock_returns,weighted_rs_lst)} 
                                #Above code makes a dict with the keys being the stock tickers, and then the values being a tuple with (stock_returns (the percentages),weighted_rs_returns(actual values associated with growth and weight))

                            
                                
                                sorted_dict = dict(sorted(returns_dict.items(), key=lambda item: item[1],reverse=True))
                                #This sorts our returns_dict, using a lambda function which specifies that we should sort by essentially the first values in each key: (value1,value2) list, i.e sort by highest percentage return
                                
                                item_list = list(sorted_dict.items())
                                #Turning into a list for easier accessing of indices next
                                

                                with cols[1]:
                                    #In these lines of code item_list[0][0] refers to first entry of the highest performing stock(by percentage return), first entry will just be ticker name
                                    #Thus item_list[-1][0] gives WORST performing stock by percentage return's ticker
                                    #Then item_list[0][1][1] gets best performing stock, goes into the second entry of list which is our tuple of the two values, and then goes into the second entry in this tuple (which is dollar value of return)
                                    st.markdown("### Portfolio Stock Rankings")
                                    st.metric(label=f"Top Performer: {item_list[0][0]}",value=f"${format_number(item_list[0][1][1])}",delta=f"{item_list[0][1][0]:.2f}%")
                                    st.metric(label=f"Poorest Performer: {item_list[-1][0]}",value=f"${format_number(item_list[-1][1][1])}",delta=f"{item_list[-1][1][0]:.2f}%")
                                    st.metric(label="Portfolio Return Compared to S&P",value=f"{portfolio_return_pct - sp_return:.2f}%")

                else:
                    st.sidebar.write("Please redo inputs")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        if first_graph is True: #Does not run the Individual Stock Analysis until the conditions of the Total Portfolio have been met. This is to stop the below code being ran before the the Total Portfolio.
            col2,col1 = st.columns([4,8]) #[4,8] for setting ratio of columns
            with col1:
                sns.set_style('darkgrid')
                returns = stock_df['Total Portfolio Value'].dropna().pct_change()*100 #Drops NA values (first) and forms percentage change
                fig1,ax1 = plt.subplots(figsize=(10,4)) 
                sns.lineplot(returns,ax = ax1)
                plt.title('Weekly Returns Over Time')
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Return")
                ax1.set_yticklabels([f'{label:.2f}%' for label in ax1.get_yticks()]) #Each y tick is formatted to 2 decimal places and with a percentage side beside it
                plt.xticks(rotation=45) #Rotates x ticks for aesthetic.
                st.pyplot(fig1) #Requires custom command for streamlit
                
            with col2:
                mean_returns = returns.mean()
                std_returns = returns.std()
                cf_95 = list(map(lambda x: round(x,1),[mean_returns-1.96*std_returns,mean_returns+1.96*std_returns])) #Basic formula for creating confidence interval
                
                st.markdown('<h2>Weekly Return Statistics</h2>',unsafe_allow_html=True)
                st.metric(label = 'Minimum',value = f"{returns.min():.1f}%")
                st.metric(label='Average (Mean)',value=f'{round(mean_returns,2)}%')
                st.metric(label = 'Maximum',value = f"{returns.max():.1f}%")
                st.metric(label='Standard Deviation of Daily Returns',value=f'{round(std_returns,1)}%')
                st.metric(label='95% Confidence Interval',value=f"[{' '.join([f'{x:.1f}%' for x in cf_95])}]")
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)

            st.markdown('<h2>Individual Stock Analysis</h2>',unsafe_allow_html=True)

            def graph(start,interval,style,typey): 
                '''
                The following function stops copying and pasting the same code with minor variations. These minor variations are 
                considered with the function's arguments. The start date of the data frame, time interval (granularity), style of graph
                and type of OHLC. Similarly enough, the kwargs argument shortens down the length of code inside the ylabels.
                Two axes are created, one for the stock price, (index 0) and one for the volume traded during the period, (index of 2)
                axes.set_ylabel[2] rotates the title to be horizontal, which needs to be pushed back as it goes through the y axis. The 
                double star refers to the kwargs dictionary
                axes[0].set_yticklabels see line 412
                Should no data be available, i.e., the NASDAQ has been closed from the start of the day to the current time in EST, an error
                will occur when referencing the dataframe with .iloc[0] /.iloc[-1] In this case, it is not possible to refer to data that doesn't
                exist, so a generic error message will appear, linking the user to NASDAQ opening hours. Note that the default time frame is 7 days,
                as it is highly unlikely that the NASDAQ will be closed for that consecutive period.
                '''
            
                kwargs = {"fontweight":'normal',"fontsize":10}
                try:
                    all_data1 = yf.Ticker(stock_selection).history(start=start,end= datetime.datetime.now(),interval=interval) 
                
                    current_price = all_data1['Close'].iloc[-1]
                    st.metric(label='Current Share Price:',value=f"${current_price:.2f}")
                    fig,axes = mpf.plot(all_data1,type=typey,style= style,volume=True,returnfig=True)
                    axes[0].set_title(f'{stock_selection.upper()} Share Price {time_selection.upper()}',fontsize=15)
                    axes[2].set_xlabel('Time Range',**kwargs)
                    axes[0].set_ylabel('Share Price',**kwargs)
                    axes[2].set_ylabel('Volume',rotation='horizontal',labelpad=30,**kwargs)
                    axes[2].set_yticks([])
                    axes[0].set_yticklabels([f'${label:.2f}' for label in axes[0].get_yticks()],fontsize=10)
                    st.pyplot(fig)
                except: 
                    st.write('The NASDAQ is currently not open. It is open Monday to Friday from 9:30-17:30 EST, the equivalent of 14:30-22:30 GMT. Please refer to the link https://www.nasdaq.com/market-activity/stock-market-holiday-schedule for further information.')
            stock_selection = st.selectbox(label='Stock for Analysis',options=stock_lst)

            if stock_selection:
                col1,col2 = st.columns([3,9])
                with col1:
                    # '''
                    # The first option is 7 days rather than Today for reasons discussed above.
                    # '''
                    list_options = ['7 Days', 'Today','1 Day','3 Days','30 Days']
                    type_selection= style= st.selectbox('Display Type:',options= ['candle','line','ohlc'])
                    style_selection= st.selectbox('Graph Type:',options=['binance','tradingview','yahoo','starsandstripes'])
                    time_selection = st.selectbox(label='Time Frame',options=list_options)
                    st.markdown("<h2>Time Granularity Information</h2>",unsafe_allow_html=True)
                    st.markdown("""<p style = "fontsize:15px;"><b>Today-</b> 1 minute <br>
                            <b>1D-</b> 1 minute <br>
                            <b>3D-</b> 1 hour <br>
                            <b>7D-</b> 1 hour <br>
                            <b>30D-</b>1 day <br>
                        </p>
                        """,unsafe_allow_html= True)
                    
                    
                with col2:
                    # '''
                    # Following code applies the graph function for efficiency. The datetime.datetime.now() object gets the immediate time
                    # By running the end date first, the start date is automatically 100 days before the end date, which the user can adjust.
                    # Note that this means the minimum time period is 100 days, and every time the user wants to adjust the end date they will 
                    # also have to adjust the start. For the special Today selection, the start of the current day (i.e., 00:00:00) EST is created.
                    # The objective of this is to show the trading day share price, rather than a full day's duration from the current time. The 
                    # timedelta object passes a duration, so the start date can be related to the end date (default of current) 
                    # '''
                    if time_selection == 'Today':
                     today = datetime.datetime.today()
                     start_of_today = datetime.datetime(today.year,today.month,today.day) 
                     graph(start_of_today,style=style_selection,typey=type_selection,interval='1m')

                    if time_selection == '1 Day':
                        graph(start=datetime.datetime.now()-timedelta(days=1),style=style_selection,typey=type_selection,interval='1m')
                    elif time_selection == '3 Days':
                        graph(start=datetime.datetime.now()-timedelta(days=3),style=style_selection,typey=type_selection,interval='1h')

                    elif time_selection == '7 Days':
                        graph(start=datetime.datetime.now()-timedelta(days=7),style=style_selection,typey=type_selection,interval='1h')
                    
                    elif time_selection == '30 Days':
                        graph(start=datetime.datetime.now()-timedelta(days=30),style=style_selection,typey=type_selection,interval='1d')
                        
                    
                    
st.markdown('<h2>Application Information</h2>',unsafe_allow_html=True)
st.markdown("""<body> The function of this website is to track the value of your portfolio over time. Information on the stocks' adjusted closing
            price is taken weekly and plotted on a graph, where you are able to view metrics about your portfolio's performance and compare it with 
            the S&P 500, the standard comparison tool for portfolios. The individual stock analysis allows you to analyse recent movements on a specific
            stock in your portfolio on a more in depth level, viewing periods' opening and closing prices, and also highs and lows. The granularity of the data can be found in the list beside the chart.

</b>""",unsafe_allow_html=True)               
st.markdown("<br></br>",unsafe_allow_html=True)
st.markdown("<h2>Who are we?</h2>",unsafe_allow_html=True)
st.markdown("<body>We are a group of four individuals in our second year of studies at UCD, studying Economics & Finance, with a shared interest in financial market analysis and programming.</body>",unsafe_allow_html=True)   
