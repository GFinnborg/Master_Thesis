import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from google.colab import drive ##this is where I stored files, change the ' x '.

######## Import Data ########

#### Data on the Historical prices
stock_data=pd.read_excel('/content/drive/MyDrive/Data - Priser.xlsx')#,header=0,index_col=0,parse_dates=True)

#### Data on the buysignals for the different portfolios, change depending on which portfolio to investigate
df=pd.read_excel('/content/drive/MyDrive/B -- Nästminsta portföljen 5-9,99% täckning.xlsx')

#############################

#### Align the data for the portfolio dataframe - indicate buysignals as bolean values TRUE/FALSE

# convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# create a new dataframe with a date range
date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
new_df = pd.DataFrame({'Date': date_range})

# create a pivot table with names as columns and dates as index
pivot_df = pd.pivot_table(df, values='Stock', index='Date', columns='Stock', aggfunc=lambda x: 'True')

# merge the pivot table with the new dataframe to fill in missing dates
merged_df = pd.merge(new_df, pivot_df, how='left', left_on='Date', right_index=True)

# fill in missing values with 'false'
merged_df = merged_df.fillna('False')

# reset index to make date a column again
merged_df = merged_df.reset_index()

# rename the columns to match the expected output
merged_df = merged_df.rename(columns={'Date': 'Date'})

#############################

# Create dataframe and align under Data column - missing values gets FALSE             
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

mask = merged_df['Date'].isin(stock_data['Date'])

adapted_df = merged_df[mask]
adapted_df = adapted_df.set_index('Date').reindex(stock_data['Date']).reset_index()

adapted_df = adapted_df.fillna(False)

# Remove index column - as it serves no function here 
adapted_df.drop('index', axis=1, inplace=True) 

#### add the stocks that are missing to the Boolean value dataframe with values set to FALSE - aligning the dataframes 1:1
missing_headers = set(stock_data.columns).difference(set(adapted_df.columns))
missing_dict = {header: [False] * len(stock_data) for header in missing_headers}
missing_df = pd.DataFrame(missing_dict)
new_df = pd.concat([adapted_df, missing_df], axis=1)

###### Here alignment can be seen ######
new_df
stock_data

##############################################################################################################################

df1 = new_df
df2 = stock_data

###### Backtesting loop ######

capital = 1000000
portfolio = {}
daily_returns = []
hold_days = 20        ####Change for different holding periods
dates = list(df2.Date)
portfolio_values = []

for i in range(len(df1)):
    buy_list = []
    sell_list = []
    # Get the stocks to buy
    for stock in df1.columns[1:]:
        if df1[stock][i]:
            buy_list.append(stock)
    # Get the stocks to sell
    if i >= hold_days:
        for stock in portfolio.keys():
            if i - portfolio[stock]['buy_day'] == hold_days:
                sell_list.append(stock)
    # Sell the stocks and update the portfolio
    for stock in sell_list:
        sell_price = df2[stock][i]
        buy_price = portfolio[stock]['buy_price']
        num_shares = portfolio[stock]['shares']
        sell_value = num_shares * sell_price
        buy_value = num_shares * buy_price
        pl = (sell_value - buy_value) / buy_value
        daily_returns.append(pl)
        capital += sell_value
        del portfolio[stock]
    # Buy the stocks and update the portfolio
    if len(portfolio) < len(df1.columns[1:]) and len(buy_list) > 0:
        num_stocks = sum([1 for stock in buy_list if not pd.isna(df2[stock][i])])
        total_capital_used = 0
        for stock in buy_list:
            if pd.isna(df2[stock][i]):
                continue
            if stock not in portfolio.keys():
                try:
                    num_shares = int((capital * 0.1) / df2[stock][i])
                except (ValueError, ZeroDivisionError):
                    num_shares = 0
                if num_shares == 0:
                    continue
                portfolio[stock] = {'buy_price': df2[stock][i], 'shares': num_shares, 'buy_day': i}
                total_capital_used += num_shares * df2[stock][i]
            elif i - portfolio[stock]['buy_day'] >= hold_days:
                num_shares = int((capital * 0.1) / df2[stock][i])
                if num_shares == 0:
                    continue
                portfolio[stock]['shares'] += num_shares
                portfolio[stock]['buy_price'] = (portfolio[stock]['buy_price'] + df2[stock][i]) / 2
                portfolio[stock]['buy_day'] = i
                total_capital_used += num_shares * df2[stock][i]
            if total_capital_used >= capital:
                break
        capital -= total_capital_used
    else:
            # Rebalance the portfolio if it is not empty
            if len(portfolio) > 0:
                total_value = capital + sum([df2[stock][i] * portfolio[stock]['shares'] for stock in portfolio.keys() if not pd.isna(df2[stock][i]) and df2[stock][i] != 0])
                target_value_per_stock = total_value / len(portfolio)
                for stock in buy_list:
                    if pd.isna(df2[stock][i]) or df2[stock][i] == 0:
                          num_shares = 0
                    else:
                         if stock not in portfolio.keys():
                             # Buy the stock if not in the portfolio
                             num_shares = int(target_value_per_stock / df2[stock][i])
                             if num_shares * df2[stock][i] > capital:
                                 # If insufficient capital, buy as many shares as possible
                                 num_shares = int(capital / df2[stock][i])
                         else:
                             # Rebalance the stock in the portfolio
                              target_value = target_value_per_stock * portfolio[stock]['shares'] / df2[stock][i]
                              num_shares = int(target_value / df2[stock][i])
                              if num_shares > portfolio[stock]['shares']:
                                 # Buy more shares if needed
                                 num_shares_to_buy = num_shares - portfolio[stock]['shares']

# Update the portfolio value
    portfolio_value = 0
    for stock in portfolio.keys():
        if not pd.isna(df2[stock][i]):
            stock_value = portfolio[stock]['shares'] * df2[stock][i]
            portfolio_value += stock_value
    portfolio_value += capital
    if i > 0:
        prev_portfolio_value = portfolio_values[-1]
        daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        if daily_return < -100.0 or daily_return > 100.0:                        #used for testing purposes
            daily_returns.append(0)
        else:
            daily_returns.append(daily_return)
    else:
        daily_returns.append((portfolio_value - 1000000) / 1000000)

    # Add current day's portfolio value to the list of portfolio values
    portfolio_values.append(portfolio_value)

# Create dictionary with dates and daily returns
date_returns_dict = {dates[i]: daily_returns[i] for i in range(len(dates))}

# Set daily returns outside the 1st to 99th percentile range to 0
for date, ret in date_returns_dict.items():
    ret_percentile_05 = np.percentile(list(date_returns_dict.values()), 1.0)
    ret_percentile_95 = np.percentile(list(date_returns_dict.values()), 99.0)
    if ret < ret_percentile_05 or ret > ret_percentile_95:
        date_returns_dict[date] = 0

# Create a Pandas DataFrame from the dictionary
df_returns = pd.DataFrame.from_dict(date_returns_dict, orient='index', columns=['daily_returns'])

# Print the DataFrame
print(df_returns)# remove rows with 0 values in the 'daily_returns' column
df_returns = df_returns[df_returns['daily_returns'] != 0]



##############################################################################################################################
#renaming
df3 = df_returns
df3.reset_index(inplace=True)  # reset index to default integer index
df3.rename(columns={'index': 'Date', 'daily_returns': 'Daily Returns'}, inplace=True)


####################### BENCHMARK #######################

df4=pd.read_excel('/content/drive/MyDrive/EW_Factor_models_python.xlsx')  

#merge df3 and df4 so dataframes alignes
merg_df = pd.merge(df3, df4, on='Date', how='inner')

#create variable ret_rf
merg_df['ret_rf'] = merg_df['Daily Returns'] - merg_df['rf']

# calculate statistics
mean = merg_df['Daily Returns'].mean()
median = merg_df['Daily Returns'].median()
std_dev = merg_df['Daily Returns'].std()
min_val = merg_df['Daily Returns'].min()
max_val = merg_df['Daily Returns'].max()
n_obs = merg_df['Daily Returns'].count()

# create new dataframe
df_stats = pd.DataFrame({'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Number of Observations'],
                         'Value': [mean, median, std_dev, min_val, max_val, n_obs]})

#plot observations
sns.set_style('darkgrid')
plt.figure(figsize=(12, 8))
sns.distplot(merg_df['Daily Returns'], hist=True, kde=True, bins=150, color='blue')
plt.xlabel('Daily Returns', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Daily Returns Distribution', fontsize=16)
plt.show()


######################## CARHART'S 4 Factor ##########################

#Y range
Y = merg_df['ret_rf']

#X range
X = merg_df[['rm_rf', 'smb', 'hml', 'mom']]

#lägg till konstant i X
X = sm.add_constant(X)

#Regression model
CARHART = sm.OLS(Y, X).fit()

print(CARHART.summary())


######################## Fama and French 3 Factor ##########################

#Y range
Y = merg_df['ret_rf']

#X range
X = merg_df[['rm_rf', 'smb', 'hml']]

#lägg till konstant i X
X = sm.add_constant(X)

#Regression model
FF3 = sm.OLS(Y, X).fit()

print(FF3.summary())


######################## CAPM ##########################

#Y range
Y = merg_df['ret_rf']

#X range
X = merg_df[['rm_rf']]

#lägg till konstant i X
X = sm.add_constant(X)

#Regression model
CAPM = sm.OLS(Y, X).fit()

print(CAPM.summary())










