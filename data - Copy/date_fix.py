import pandas as pd

# Read the CSV file with 'MM/DD/YY' formatted dates
ftse = pd.read_csv("data/FTSE.csv")

# Convert the 'Date' column to 'DD/MM/YY' format
ftse['Date'] = pd.to_datetime(ftse['Date'], format='%m/%d/%y').dt.strftime('%d/%m/%y')

# Save the DataFrame to a new CSV file with the updated date format
output_file_path = 'FTSE_updated.csv'
ftse.to_csv(output_file_path, index=False)

print(ftse)
