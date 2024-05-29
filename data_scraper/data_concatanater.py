import os
import pandas as pd

# Root directory containing symbol directories
root_dir = '/Volumes/SD/NASDAQ_Historical_Data/stock_data'

# Path to store concatenated data
output_dir = '/Volumes/SD/NASDAQ_Historical_Data/concatenated_stock_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each symbol directory in the root directory
for symbol_dir in os.listdir(root_dir):
    symbol_path = os.path.join(root_dir, symbol_dir)
    
    # Check if the path is a directory
    if os.path.isdir(symbol_path):
        # Initialize a list to hold DataFrames for the current symbol
        symbol_data_list = []
        
        # Iterate over each file in the symbol directory
        for file in os.listdir(symbol_path):
            file_path = os.path.join(symbol_path, file)
            
            # Check if the file is a CSV
            if file_path.endswith('.csv'):
                try:
                    # Read CSV file into a DataFrame
                    df_temp = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    # If there's an encoding error, try a different encoding
                    df_temp = pd.read_csv(file_path, encoding='ISO-8859-1')
                
                # Append the DataFrame to the list
                symbol_data_list.append(df_temp)
        
        # Concatenate all DataFrames for the current symbol
        if symbol_data_list:
            symbol_data = pd.concat(symbol_data_list, ignore_index=True)
            
            # Sort the DataFrame by 'timestamp', reset index, and drop the 'Unnamed: 0' column if it exists
            symbol_data = symbol_data.sort_values(by='timestamp').reset_index(drop=True)
            if 'Unnamed: 0' in symbol_data.columns:
                symbol_data = symbol_data.drop(columns=['Unnamed: 0'])
            
            # Define the output file path for the current symbol
            output_file = os.path.join(output_dir, f'{symbol_dir}.csv')
            
            # Save the symbol's concatenated data to the specified output file
            symbol_data.to_csv(output_file, index=False)
            
            print(f"Concatenated data for {symbol_dir} saved to {output_file}")
