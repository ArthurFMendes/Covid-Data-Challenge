#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:19:14 2023

@author: arthurmendes
"""
"""
Challenge Questions:

1.    Is there any missing data? If so, perform missing data treatment of
your choice.

2.    What are the data types? Looks like most of the numeric columns need
to be changed. Convert them to floats and integers as necessary.

"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

# Load the dataset into a DataFrame
df = pd.read_csv("/Users/arthurmendes/Desktop/Data Challenge/covid_worldwide.csv")

# Column names
df.columns

# Displaying the first rows of the DataFrame
print(df.head())

# Dimensions of the DataFrame
df.shape

# Information about each variable
df.info()

# Check for missing values
missing_values = df.isnull().sum()

# Print the number of missing values in each column
print(missing_values)


# Descriptive statistics
df.describe().round(2)


df.sort_values('Total Cases', ascending = False)

# Creating new flags for NAs
for col in df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)

# Returning column types
for col in df.columns:
    col_type = df.dtypes[col]
    print(f"{col} has data type {col_type}")

# Delete a row that is not a country
df = df.drop(df[df['Country'] == 'MS Zaandam'].index)  
    
# Modofy data types
cols_to_convert = ['Total Cases','Total Deaths','Total Recovered',
                   'Active Cases', 'Total Test','Population']

for col in cols_to_convert:
    # Remove commas and replace missing values with NaN
    df[col] = df[col].str.replace(',', '').replace('', np.nan).astype(float)
    # Fill missing values with column median
    df[col].fillna(df[col].median(), inplace=True)
    # Convert data type to integer
    df[col] = df[col].astype(int)

###############################################################################
"""
3.    How many different countries had the virus?
4.    Create a geographical plot of the distribution of deaths from around the world.
5.    What are the top 5 countries in active cases?
6.    What are the top 5 countries in total recoveries?
7.    Create your own question and answer it.
8.    Bonus: Create a dashboard with the important insights you find.
"""

# Count the number of times column 'Total Cases' has a number greater than 0
positive_countries = (df['Total Cases'] > 0).sum()

print(positive_countries)

# Get the top 5 values from the 'Active Cases' column
top_5_ActiveCases = df.nlargest(5, 'Active Cases')

# Print the country name and value of each of the top 5 values
for i, row in top_5_ActiveCases.iterrows():
    print(f"{row['Country']} has a total of {row['Active Cases']} active cases")

# Get the top 5 values from the 'Total Recovered' column
top_5_TotalRecovered = df.nlargest(5, 'Total Recovered')

# Print the country name and value of each of the top 5 values
for i, row in top_5_TotalRecovered.iterrows():
    print(f"{row['Country']} has a total of {row['Total Recovered']} people recovered")

# Creating Death rate
df['Death Rate'] = (df['Total Deaths'] / df['Total Cases']) * 100
df['Death Rate'] = df['Death Rate'].round(1)

# filter out rows where the flag for total deaths is 1
df_filtered = df[df['m_Total Deaths'] == 0]

# get top 5 rows by target column
top_5_by_DeathRate = df_filtered.nlargest(5, 'Death Rate')

# Print the country name and value of each of the top 5 values
for i, row in top_5_by_DeathRate.iterrows():
    print(f"{row['Country']} has a {row['Death Rate']}% death rate ")

# Plotting Top 5 Countries with the most Active Cases

dpc_df = df.sort_values('Active Cases', ascending=False).head(5)

dpc_plot = sns.barplot(data=dpc_df, x='Active Cases', y='Country')

dpc_plot.set_xlabel('Active Cases, million', fontweight='bold')
dpc_plot.set_ylabel('Country', fontweight='bold')
dpc_plot.set_title("Top 5 Countries With the Most Active Covid Cases", fontweight='heavy')

plt.show()

# Plotting Top 5 Countries with the most Active Cases

dpc_df = df.sort_values('Active Cases', ascending=False).head(5)

dpc_plot = sns.barplot(data=dpc_df, x='Active Cases', y='Country')

dpc_plot.set_xlabel('Active Cases, million', fontweight='bold')
dpc_plot.set_ylabel('Country', fontweight='bold')
dpc_plot.set_title("Top 5 Countries With the Most Active Covid Cases", fontweight='heavy')

plt.show()

# Plotting Top 5 Countries with the highest death rate

dpc_df = top_5_by_DeathRate.sort_values('Death Rate', ascending=False).head(5)

dpc_plot = sns.barplot(data=dpc_df, x='Death Rate', y='Country')

dpc_plot.set_xlabel('Death Rate, %', fontweight='bold')
dpc_plot.set_ylabel('Country', fontweight='bold')
dpc_plot.set_title("Top 5 Countries With the Largest Death Rate", fontweight='heavy')

plt.show()

###############################################################################
import pycountry
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Generate Country Code  Based on Country Name 

def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            code=pycountry.countries.get(name=country).alpha_3
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE

# create a column for code 
df['CODE']=alpha3code(df.Country)
df.head()

#Merging covid dataset with the world dataset

dfmerge = df.merge(world, left_on='Country', right_on='name')

dfmerged = GeoDataFrame(dfmerge)

#COVID Cases by Country

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('Covid Cases by Country')
dfmerged.plot(column='Total Deaths', legend=True, cmap='OrRd', ax=ax, 
              legend_kwds={'label': "Total Deaths by Country",
                          'orientation' : 'horizontal'})
plt.show()

