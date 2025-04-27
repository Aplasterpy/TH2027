import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C:/Users/devst/Desktop/fileee/data/TH2027/CSV.csv")

# Ensure the 'Volume' column is numeric, handling commas as necessary
df['Volume'] = df['Volume'].replace({',': ''}, regex=True).astype(float)

# Ensure other columns are numeric as needed
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)

# Function to calculate the Exponential Moving Average (EMA)
def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Function to calculate the Volume Weighted Moving Average (VWMA)
def calculate_vwma(data, window):
    return (data['Close'] * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()

# Function to calculate the Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Calculate technical indicators
df['EMA10'] = calculate_ema(df, 10)
df['EMA20'] = calculate_ema(df, 20)
df['VWMA20'] = calculate_vwma(df, 20)
df['RSI14'] = calculate_rsi(df, 14)

# Create buy and sell signals
df['buy_signal'] = (df['EMA10'] > df['VWMA20']) & (df['RSI14'] < 30)
df['sell_signal'] = (df['EMA10'] < df['EMA20']) & (df['RSI14'] > 70)

# Print out the indicators and the signal conditions for the last 10 rows to inspect the calculations
print(df[['EMA10', 'EMA20', 'VWMA20', 'RSI14']].tail(10))
print("Signal Conditions (buy_signal, sell_signal):")
print(df[['EMA10', 'EMA20', 'VWMA20', 'RSI14', 'buy_signal', 'sell_signal']].tail(10))

# Reset the index temporarily to access the 'Date' column
df_reset = df.reset_index()

# Print out the signals with the 'Date' column
print(df_reset[['Date', 'buy_signal', 'sell_signal']].tail(10))

# Backtesting strategy with a stop loss of 5% and take profit of 10%
initial_capital = 10000
capital = initial_capital
capital_history = []
position = None
entry_price = 0
entry_date = None
exit_date = None
exit_price = 0
trade_history = []

# Simulate trading based on signals
for i in range(1, len(df)):
    # Check for buy signal
    if df['buy_signal'].iloc[i] and position is None:
        position = capital / df['Close'].iloc[i]
        entry_price = df['Close'].iloc[i]
        entry_date = df.index[i]
        print(f"Buying at {entry_price} on {entry_date}")
        
    # Check for sell signal
    elif df['sell_signal'].iloc[i] and position is not None:
        exit_price = df['Close'].iloc[i]
        exit_date = df.index[i]
        profit_loss = (exit_price - entry_price) * position - 2  # Subtract transaction fee
        trade_history.append({
            'entry_time': entry_date, 'exit_time': exit_date,
            'profit_loss': profit_loss, 'entry_price': entry_price,
            'exit_price': exit_price, 'stoploss': entry_price * 0.95,
            'take_profit': entry_price * 1.10,
            'hit': 'Profit' if profit_loss > 0 else 'Loss'
        })
        capital += profit_loss
        position = None
        print(f"Selling at {exit_price} on {exit_date}, Profit/Loss: {profit_loss}")

    # Implement stop loss or take profit for active position
    if position is not None:
        stop_loss_price = entry_price * 0.95
        take_profit_price = entry_price * 1.10
        current_price = df['Close'].iloc[i]
        
        if current_price <= stop_loss_price or current_price >= take_profit_price:
            exit_price = current_price
            exit_date = df.index[i]
            profit_loss = (exit_price - entry_price) * position - 2
            trade_history.append({
                'entry_time': entry_date, 'exit_time': exit_date,
                'profit_loss': profit_loss, 'entry_price': entry_price,
                'exit_price': exit_price, 'stoploss': stop_loss_price,
                'take_profit': take_profit_price,
                'hit': 'Stoploss' if current_price <= stop_loss_price else 'Take Profit'
            })
            capital += profit_loss
            position = None
            print(f"Position closed at {exit_price} on {exit_date}, Profit/Loss: {profit_loss}")

# Final statistics
if trade_history:
    trades_df = pd.DataFrame(trade_history)
    total_trades = len(trades_df)
    total_profit_loss = trades_df['profit_loss'].sum()
    profit_positions = len(trades_df[trades_df['profit_loss'] > 0])
    loss_positions = len(trades_df[trades_df['profit_loss'] < 0])
    p_to_loss_ratio = profit_positions / loss_positions if loss_positions != 0 else profit_positions
else:
    total_trades = 0
    total_profit_loss = 0
    profit_positions = 0
    loss_positions = 0
    p_to_loss_ratio = 0

# Output statistics
print(f"Total Trades: {total_trades}")
print(f"Total Profit/Loss: {total_profit_loss}")
print(f"Profit Positions: {profit_positions}")
print(f"Loss Positions: {loss_positions}")
print(f"Profit to Loss Ratio: {p_to_loss_ratio:.2f}")

# Visualization
if total_trades > 0:
    # Plot candlestick chart
    apds = [
        mpf.make_addplot(df['EMA10'], color='blue'),
        mpf.make_addplot(df['EMA20'], color='red'),
        mpf.make_addplot(df['VWMA20'], color='green'),
        mpf.make_addplot(df['RSI14'], panel=1, color='purple')
    ]
    
    # Entry and exit markers
    entry_dates = trades_df['entry_time']
    exit_dates = trades_df['exit_time']
    entry_prices = trades_df['entry_price']
    exit_prices = trades_df['exit_price']
    
    markers = [
        mpf.make_addplot(entry_prices, type='scatter', markersize=100, marker='^', color='g'),
        mpf.make_addplot(exit_prices, type='scatter', markersize=100, marker='v', color='r')
    ]
    
    mpf.plot(df,
             type='candle',
             addplot=apds + markers,
             volume=True,
             style='yahoo',
             title="Trading Strategy Visualization",
             panel_ratios=(3, 1))  # 3:1 ratio for main chart and RSI panel
else:
    print("No trades were executed.")
