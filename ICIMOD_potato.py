# %%
# Load libraries 

import pandas as pd 
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load ERA5 data and filter according to your need 

precTot_1950_2022_municipality = pd.read_csv("D:/GIIS/ERA5 Hourly Data/Data/Average_grid_data (Fourth)/Municipality/precTot_1950_2022_municipality.csv")
tempMax_1950_2022_municipality = pd.read_csv("D:/GIIS/ERA5 Hourly Data/Data/Average_grid_data (Fourth)/Municipality/tempMax_1950_2022_municipality.csv")
tempMin_1950_2022_municipality = pd.read_csv("D:/GIIS/ERA5 Hourly Data/Data/Average_grid_data (Fourth)/Municipality/tempMin_1950_2022_municipality.csv")

# %%
# Select particular municipalites according to our need 

precTot_1950_2022_municipality = precTot_1950_2022_municipality[["year", "month", "day", "Godawari", "Budhinanda", "Naumule"]]
tempMax_1950_2022_municipality = tempMax_1950_2022_municipality[["year", "month", "day", "Godawari", "Budhinanda", "Naumule"]]
tempMin_1950_2022_municipality = tempMin_1950_2022_municipality[["year", "month", "day", "Godawari", "Budhinanda", "Naumule"]]

tempMax_1950_2022_municipality

# %%
# Take annual mean and annual sum of min, max and precp 

tempMax_1950_2022_annual_mean = tempMax_1950_2022_municipality.groupby("year", as_index=False)[["Godawari", "Budhinanda", "Naumule"]].mean()
tempMin_1950_2022_annual_mean = tempMin_1950_2022_municipality.groupby("year", as_index= False)[["Godawari", "Budhinanda", "Naumule"]].mean()
precTot_1950_2022__annual_sum = precTot_1950_2022_municipality.groupby("year", as_index= False)[["Godawari", "Budhinanda", "Naumule"]].sum()

# %%
# Filter according to year 1980-2022 

tempMax_1980_2022_annual_mean = tempMax_1950_2022_annual_mean[(tempMax_1950_2022_annual_mean["year"] >= 1980) & (tempMax_1950_2022_annual_mean["year"] <= 2022)]
tempMin_1980_2022_annual_mean = tempMin_1950_2022_annual_mean[(tempMin_1950_2022_annual_mean["year"] >= 1980) & (tempMin_1950_2022_municipality["year"] <= 2022)]
precTot_1980_2022_annual_sum = precTot_1950_2022__annual_sum[(precTot_1950_2022__annual_sum["year"] >= 1980) & (precTot_1950_2022__annual_sum["year"] <= 2022)]

# %%
# Take mean of Max and Min temperature 

tempMean_1980_2022_annual_mean = pd.concat([tempMax_1980_2022_annual_mean, tempMin_1980_2022_annual_mean]).groupby(level=0).mean()
tempMean_1980_2022_annual_mean = tempMean_1980_2022_annual_mean[["Godawari", "Budhinanda", "Naumule"]]
tempMean_1980_2022_annual_mean

# %%
year = tempMax_1980_2022_annual_mean[["year"]]
year

# %%
tempMean_1980_2022_annual_mean = pd.merge(year, tempMean_1980_2022_annual_mean, left_index=True, right_index=True, how='inner')
tempMean_1980_2022_annual_mean

# %%
# Now we have 4 dataframes to work with 

#tempMax_1980_2022_annual_mean
#tempMin_1980_2022_annual_mean
#tempMean_1980_2022_annual_mean
#precTot_1980_2022_annual_sum

# %%
tempMin_1980_2022_annual_mean = tempMin_1980_2022_annual_mean.to_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMin_1980_2022_annual_mean.csv")
tempMax_1980_2022_annual_mean = tempMax_1980_2022_annual_mean.to_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMax_1980_2022_annual_mean.csv")
tempMean_1980_2022_annual_mean = tempMean_1980_2022_annual_mean.to_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMean_1980_2022_annual_mean.csv")
precTot_1980_2022_annual_sum = precTot_1980_2022_annual_sum.to_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/precTot_1980_2022_annual_sum.csv")

# %%
tempMin_1980_2022_annual_mean = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMin_1980_2022_annual_mean.csv")
tempMax_1980_2022_annual_mean = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMax_1980_2022_annual_mean.csv")
tempMean_1980_2022_annual_mean = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/tempMean_1980_2022_annual_mean.csv")
precTot_1980_2022_annual_sum = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/max min mean precp (First)/precTot_1980_2022_annual_sum.csv")

# %%
tempMax_1980_2022_annual_mean

# %%
# Creatine a line plots for Maximum Temperature

year = tempMax_1980_2022_annual_mean["year"]
Godawari = tempMax_1980_2022_annual_mean["Godawari"]
Budhinanda = tempMax_1980_2022_annual_mean["Budhinanda"]
Naumule = tempMax_1980_2022_annual_mean["Naumule"]

# Fit linear regression model 
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

plt.figure(figsize=(10,6))
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black', label='Trendline')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Maximum Temperature',fontsize=14)
plt.text(2017,24.55,"Slope = -0.0011\n p value = 0.837")
plt.text(1978.4,24.65,"Godawari (Kailali)")
plt.show()

# Fit linear regression model 
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

plt.figure(figsize=(10,6))
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black', label='Trendline')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Maximum Temperature',fontsize=14)
plt.text(2017,12.9,"Slope = 0.024\n p value = 0.003")
plt.text(1978.4,13,"Budhinanda (Bajura)")

# Fit linear regression model 
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

plt.figure(figsize=(10,6))
plt.plot(year, Naumule, marker = 'o', linestyle='-', color='g', label = "Naumule")
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black', label='Trendline')
plt.xlabel('Year', fontsize = 14)
plt.ylabel('Maximum Temperature', fontsize = 14)
plt.text(2017,15.7,"Slope = 0.013\n p value = 0.029")
plt.text(1978.4,15.8,"Naumule (Dailekh)")

plt.savefig('Godawari_plot.tiff', format='tiff', dpi=300) 


# %%
slope = Godawari_coefficients[0]
slope

# %%
slope = Budhinanda_coefficients[0]
slope

# %%
slope = Naumule_coefficients[0]
slope

# %%
# Calculate slope and p value using linregress function from scipy package. 
import scipy.stats

# Godawari
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Godawari)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value )

# %%
# Budhinanda
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Budhinanda)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value )

# %%
# Naumule
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Naumule)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value )

# %%
# Creatine a line plots for Maximum Temperature

year = tempMax_1980_2022_annual_mean["year"]
Godawari = tempMax_1980_2022_annual_mean["Godawari"]
Budhinanda = tempMax_1980_2022_annual_mean["Budhinanda"]
Naumule = tempMax_1980_2022_annual_mean["Naumule"]

# Fit linear regression model for Godawari
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

# Fit linear regression model for Budhinanda
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

# Fit linear regression model for Naumule
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

# Create subplots
plt.figure(figsize=(13, 18))

# Subplot 1 - Godawari
plt.subplot(3, 1, 1)
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari (Kailali)')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Maximum Temperature(°C)', fontsize=16)
plt.yticks([22,23,24,25], fontsize=16)
plt.text(2017,24.7,"Slope = -0.0011 \np value = 0.837", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 


# Subplot 2 - Budhinanda
plt.subplot(3, 1, 2)
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda (Bajura)')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Maximum Temperature(°C)', fontsize=16)
plt.yticks([9,10,11,12,13], fontsize=16)
plt.text(2017,12.8,"Slope = 0.024 \np value = 0.003", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Subplot 3 - Naumule
plt.subplot(3, 1, 3)
plt.plot(year, Naumule, marker='o', linestyle='-', color='g', label='Naumule (Dailekh)')
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Maximum Temperature(°C)', fontsize=16)
plt.yticks([13,14,15,16], fontsize=16)
plt.text(2017,15.7,"Slope = 0.013 \np value = 0.029",fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Adjust layout
plt.tight_layout()

# Save the figure as a TIFF file
plt.savefig('ICIMOD_potato_tempMax.png', format='png', dpi=300)

# Show the plot
plt.show()


# %%
# Creatine a line plots for Minimum Temperature

year = tempMin_1980_2022_annual_mean["year"]
Godawari = tempMin_1980_2022_annual_mean["Godawari"]
Budhinanda = tempMin_1980_2022_annual_mean["Budhinanda"]
Naumule = tempMin_1980_2022_annual_mean["Naumule"]

# Fit linear regression model for Godawari
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

# Fit linear regression model for Budhinanda
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

# Fit linear regression model for Naumule
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

# Create subplots
plt.figure(figsize=(13, 18))

# Subplot 1 - Godawari
plt.subplot(3, 1, 1)
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari (Kailali)')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Minimum Temperature(°C)', fontsize=16)
plt.yticks([13,14,15,16], fontsize=16)
plt.text(2017,15.7,"Slope = 0.017 \np value = 0.00004", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 


# Subplot 2 - Budhinanda
plt.subplot(3, 1, 2)
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda (Bajura)')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Minimum Temperature(°C)', fontsize=16)
plt.yticks([1,2,3,4], fontsize=16)
plt.text(2017,4.1,"Slope = 0.028 \np value = 0.0002", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Subplot 3 - Naumule
plt.subplot(3, 1, 3)
plt.plot(year, Naumule, marker='o', linestyle='-', color='g', label='Naumule (Dailekh)')
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Minimum Temperature(°C)', fontsize=16)
plt.yticks([3,4,5,6], fontsize=16)
plt.text(2017,3.1,"Slope = 0.020 \np value = 0.001",fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Adjust layout
plt.tight_layout()

# Save the figure as a TIFF file
plt.savefig('ICIMOD_potato_tempMin.png', format='png', dpi=300)

# Show the plot
plt.show()

# %%
# Minimum Temperature Slope and p value

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Godawari)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Budhinanda)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Naumule)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
# Creatine a line plots for Mean Temperature

year = tempMean_1980_2022_annual_mean["year"]
Godawari = tempMean_1980_2022_annual_mean["Godawari"]
Budhinanda = tempMean_1980_2022_annual_mean["Budhinanda"]
Naumule = tempMean_1980_2022_annual_mean["Naumule"]

# Fit linear regression model for Godawari
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

# Fit linear regression model for Budhinanda
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

# Fit linear regression model for Naumule
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

# Create subplots
plt.figure(figsize=(13, 18))

# Subplot 1 - Godawari
plt.subplot(3, 1, 1)
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari (Kailali)')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Mean Temperature(°C)', fontsize=16)
plt.yticks([17,18,19,20], fontsize=16)
plt.text(2017,19.7,"Slope = 0.008 \np value = 0.058", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 


# Subplot 2 - Budhinanda
plt.subplot(3, 1, 2)
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda (Bajura)')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Mean Temperature(°C)', fontsize=16)
plt.yticks([5,6,7,8], fontsize=16)
plt.text(2017,8.4,"Slope = 0.026 \np value = 0.0008", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Subplot 3 - Naumule
plt.subplot(3, 1, 3)
plt.plot(year, Naumule, marker='o', linestyle='-', color='g', label='Naumule (Dailekh)')
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Mean Temperature(°C)', fontsize=16)
plt.yticks([8,9,10,11], fontsize=16)
plt.text(2017,11,"Slope = 0.016 \np value = 0.005",fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Adjust layout
plt.tight_layout()

# Save the figure as a TIFF file
plt.savefig('ICIMOD_potato_tempMean.png', format='png', dpi=300)

# Show the plot
plt.show()

# %%
# Slope and p value of Mean Temperature 

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Godawari)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Budhinanda)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Naumule)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
# Creatine a line plots for Total Precipitation

year = precTot_1980_2022_annual_sum["year"]
Godawari = precTot_1980_2022_annual_sum["Godawari"]
Budhinanda = precTot_1980_2022_annual_sum["Budhinanda"]
Naumule = precTot_1980_2022_annual_sum["Naumule"]

# Fit linear regression model for Godawari
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

# Fit linear regression model for Budhinanda
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

# Fit linear regression model for Naumule
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

# Create subplots
plt.figure(figsize=(13, 18))

# Subplot 1 - Godawari
plt.subplot(3, 1, 1)
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari (Kailali)')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Total Precipitation(mm)', fontsize=16)
plt.yticks([1600,1800,2000,2200,2400,2600,2800], fontsize=16)
plt.text(2017,1580,"Slope = 5.817 \np value = 0.108", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 


# Subplot 2 - Budhinanda
plt.subplot(3, 1, 2)
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda (Bajura)')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Total Precipitation(mm)', fontsize=16)
plt.yticks([1200,1400,1600,1800,2000], fontsize=16)
plt.text(2017,1210,"Slope = 3.478 \np value = 0.109", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Subplot 3 - Naumule
plt.subplot(3, 1, 3)
plt.plot(year, Naumule, marker='o', linestyle='-', color='g', label='Naumule (Dailekh)')
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Total Precipitation(mm)', fontsize=16)
plt.yticks([1600,1800,2000,2200,2400], fontsize=16)
plt.text(2017,1520,"Slope = 4.559 \np value = 0.064",fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Adjust layout
plt.tight_layout()

# Save the figure as a TIFF file
plt.savefig('ICIMOD_potato_precTot.png', format='png', dpi=300)

# Show the plot
plt.show()

# %%
# Slope and p value of Total Precipitation 

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Godawari)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Budhinanda)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Naumule)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
###
# Calculate Growing degree days (GDD)

icimod_gdd = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/Daily min max gdd (Second)/test_delete_later.csv")

# %%
# Take sum of the daily gdd 

icimod_yearly_sum = icimod_gdd.groupby('year', as_index=False)[["Godawari", "Budhinanda", "Naumule"]].sum()
icimod_yearly_sum.to_csv("ICIMOD_GDD_yearly.csv")


# %%
# Load GDD yearly data to plot 

ICIMOD_GDD_yearly = pd.read_csv("C:/GIIS/ICIMOD Potato/Data/Daily min max gdd (Second)/ICIMOD_GDD_yearly.csv")
ICIMOD_GDD_yearly

# %%
# Creatine a line plots for Growing Degree Days (GDD)

year = ICIMOD_GDD_yearly["year"]
Godawari = ICIMOD_GDD_yearly["Godawari"]
Budhinanda = ICIMOD_GDD_yearly["Budhinanda"]
Naumule = ICIMOD_GDD_yearly["Naumule"]

# Fit linear regression model for Godawari
Godawari_coefficients = np.polyfit(year, Godawari, 1)
Godawari_polynomial = np.poly1d(Godawari_coefficients)

# Fit linear regression model for Budhinanda
Budhinanda_coefficients = np.polyfit(year, Budhinanda, 1)
Budhinanda_polynomial = np.poly1d(Budhinanda_coefficients)

# Fit linear regression model for Naumule
Naumule_coefficients = np.polyfit(year, Naumule, 1)
Naumule_polynomial = np.poly1d(Naumule_coefficients)

# Create subplots
plt.figure(figsize=(13, 18))

# Subplot 1 - Godawari
plt.subplot(3, 1, 1)
plt.plot(year, Godawari, marker='o', linestyle='-', color='r', label='Godawari (Kailali)')
plt.plot(year, Godawari_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Growing Degree Days', fontsize=16)
plt.yticks([2800,3000,3200,3400,3600], fontsize=16)
plt.text(2017,3570,"Slope = 3.063 \np value = 0.060", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 


# Subplot 2 - Budhinanda
plt.subplot(3, 1, 2)
plt.plot(year, Budhinanda, marker='o', linestyle='-', color='b', label='Budhinanda (Bajura)')
plt.plot(year, Budhinanda_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Growing Degree Days', fontsize=16)
plt.yticks([-1600,-1400,-1200,-1000,-800,-600], fontsize=16)
plt.text(2017,-550,"Slope = 9.807 \np value = 0.0007", fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Subplot 3 - Naumule
plt.subplot(3, 1, 3)
plt.plot(year, Naumule, marker='o', linestyle='-', color='g', label='Naumule (Dailekh)')
plt.plot(year, Naumule_polynomial(year), linestyle='--', color='black')
plt.xlabel('Year', fontsize=16)
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize=16)
plt.ylabel('Growing Degree Days', fontsize=16)
plt.yticks([-400,-200,0,200,400], fontsize=16)
plt.text(2017,380,"Slope = 6.197 \np value = 0.005",fontsize=14)
plt.legend(loc='upper left', prop={'size': 14}) 

# Adjust layout
plt.tight_layout()

# Save the figure as a TIFF file
plt.savefig('ICIMOD_potato_GDD.png', format='png', dpi=300)

# Show the plot
plt.show()

# %%
# Slope and p value of Growing Degree Days
import scipy.stats

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Godawari)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Budhinanda)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)

# %%
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(year, Naumule)
print("P-value:", p_value)
print("Slope", slope)
print("r_value", r_value)


