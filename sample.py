import numpy as np
from astropy.io import fits
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.time import Time

# Load the FITS file
hdulist = fits.open('your_virgo_file.fits')
data = hdulist[0].data  # Access the 0th index data

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['time', 'intensity'])

# Convert seconds since reference epoch to datetime
# The reference epoch is '1996-01-23T00:00:04.46' TAI
reference_epoch = Time('1996-01-23T00:00:04.46', format='isot', scale='tai')
reference_epoch_datetime = reference_epoch.datetime

# Convert seconds to datetime
df['datetime'] = df['time'].apply(lambda x: 
    pd.Timestamp(reference_epoch_datetime) + pd.Timedelta(seconds=float(x)))

# Set the datetime column as the index
df.set_index('datetime', inplace=True)

# Resample to monthly data
monthly_data = df.resample('MS').mean()  # 'MS' = month start

# Create a 13-month running average centered on each month
# (6 months before, current month, 6 months after)
monthly_data['13month_avg'] = monthly_data['intensity'].rolling(window=13, center=True).mean()

# Drop NaN values that result from the rolling window
monthly_data_clean = monthly_data.dropna(subset=['13month_avg'])

# Apply Savitzky-Golay filter to the 13-month average
monthly_data_clean['smoothed'] = signal.savgol_filter(
    monthly_data_clean['13month_avg'], 13, 1)

# If you need to match with SOHO sunspot data, you might want to:
# 1. Create a date column in 'YYYY-MM' format for easier matching
monthly_data_clean['year_month'] = monthly_data_clean.index.strftime('%Y-%m')

# 2. Save the processed data
monthly_data_clean.to_csv('virgo_blue_monthly.csv')

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(monthly_data_clean.index, monthly_data_clean['intensity'], 'b-', 
         alpha=0.4, label='Monthly Average')
plt.plot(monthly_data_clean.index, monthly_data_clean['13month_avg'], 'g-', 
         label='13-Month Running Average')
plt.plot(monthly_data_clean.index, monthly_data_clean['smoothed'], 'r-', 
         linewidth=2, label='Savitzky-Golay Filtered')
plt.legend()
plt.title('VIRGO Blue Channel Irradiance (1996-2023)')
plt.xlabel('Time')
plt.ylabel('Intensity (ppm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
