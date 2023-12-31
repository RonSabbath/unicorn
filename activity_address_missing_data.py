# -*- coding: utf-8 -*-
"""Activity_Address missing data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LOE1WfubHhuedzaP-DPvHgJi_g-5uHJp

# Activity: Address missing data

## Introduction

The datasets that data professionals use to solve problems typically contain missing values, which must be dealt with in order to achieve clean, useful data. This is particularly crucial in exploratory data analysis (EDA). In this activity, you will learn how to address missing data.

You are a financial data consultant, and an investor has tasked your team with identifying new business opportunities. To help them decide which future companies to invest in, you will provide a list of current businesses valued at more than $1 billion. These are sometimes referred to as "unicorns." Your client will use this information to learn about profitable businesses in general.

The investor has asked you to provide them with the following data:
- Companies in the `hardware` industry based in either `Beijing`, `San Francisco`, or `London`
- Companies in the `artificial intelligence` industry based in `London`
-  A list of the top 20 countries sorted by sum of company valuations in each country, excluding `United States`, `China`, `India`, and `United Kingdom`
- A global valuation map of all countries with companies that joined the list after 2020
- A global valuation map of all countries except `United States`, `China`, `India`, and `United Kingdom` (a separate map for Europe is also required)

Your dataset includes a list of businesses and data points, such as the year they were founded; their industry; and their city, country, and continent.

## **Step 1: Imports**

### Import libraries

Import the following relevant Python libraries:
* `numpy`
* `pandas`
* `matplotlib.pyplot`
* `plotly.express`
* `seaborn`
"""

# Import libraries and modules.

### YOUR CODE HERE ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

"""### Load the dataset

The dataset is currently in CSV format and in a file named `Unicorn_Companies.csv`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.
"""

# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ###
df_companies = pd.read_csv("Unicorn_Companies.csv")

"""## **Step 2: Data exploration**

Explore the dataset and answer questions that will guide your management of missing values.

### Display top rows

Display the first 10 rows of the data to understand how the dataset is structured.
"""

# Display the first 10 rows of the data.

### YOUR CODE HERE ###
df_companies.head(10)

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Refer to the materials about exploratory data analysis in Python.

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  There is a function in the `pandas` library that allows you to get a specific number of rows from the top of a DataFrame.


</details>

<details>
  <summary><h4><strong>Hint 3</strong></h4></summary>

  Call the `head()` function from the `pandas` library.

</details>

### Statistical properties of the dataset

Use `pandas` library to get a better sense of the data, including range, data types, mean values, and shape.

Review this information about the dataset by using the `pandas` library on the `df_companies` DataFrame and answering the following questions.
"""

# Get the shape of the dataset.

### YOUR CODE HERE ###
df_companies.shape

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Refer to the material about exploratory data analysis in Python.

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  Print the 'shape' of the DataFrame.

</details>

**Question: What is the shape of the dataset?**

The dataset has 1074 rows and 10 columns.
"""

# Get the data types and number of non-null values in the dataset.

### YOUR CODE HERE ###
df_companies.info()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Refer to the material about exploratory data analysis in Python.

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  Use the 'info()' method of the DataFrame.

</details>

**Question: What are the data types of various columns?**
"""

The dataset the data types of the columns are as follows: object (9 columns) and int64 (1 column).

"""**Question: How many columns contain non-null values less than the total rows in the dataset?**

There are two columns in the dataset that contain non-null values less than the total number of rows. These columns are:
1. The "City" column, which has 1,058 non-null values, indicating that there are 16 missing values in this column.
2. The "Select Investors" column, which has 1,073 non-null values, indicating that there is 1 missing value in this column.
"""

# Get the range of different values in the dataset.

### YOUR CODE HERE ###
df_companies.describe()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Refer to the material about exploratory data analysis in Python.

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  There is a function in the `pandas` library that allows you to find descriptive statistics for the numeric columns in a DataFrame.


</details>

<details>
  <summary><h4><strong>Hint 3</strong></h4></summary>

  Call the `describe()` function from the `pandas` library.

</details>

**Question: In what year was the oldest company founded?**

The oldest company funded in the dataset was founded in 1919.

### Data preprocessing

In order to answer the investor's questions, some data preprocessing steps are required. The first step is to add the `Year Joined` column to the dataset.
"""

# Create a new column "Year Joined" from "Date Joined".

### YOUR CODE HERE ###
df_companies['Year Joined'] = pd.to_datetime(df_companies['Date Joined']).dt.year

"""Now, prepare the dataset to create a sum of valuations in each country. Currently, the `Valuation` is a string that starts with a `$` and ends with a `B`. Because this column is not in a numeric datatype, it is impossible to properly sum these values. To convert `Valuation` column to numeric, first remove the `$` and `B` symbols from the column and save the results to a new `Valuation_num` column."""

# Remove the extra characters from the Valuation column.

### YOUR CODE HERE ###
df_companies['Valuation_num'] = df_companies['Valuation'].str.replace('$', '').str.replace('B', '')

# Convert the column to numeric

### YOUR CODE HERE ###
df_companies['Valuation_num'] = pd.to_numeric(df_companies['Valuation_num'])

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Columns in different data types can be converted to numeric data type using `pd.to_numeric()`.

</details>

### Find missing values

The unicorn companies dataset is fairly clean, with few missing values.
"""

# Find the number of missing values in each column in this dataset.

### YOUR CODE HERE ###
missing_values = df_companies.isnull().sum()
missing_values

"""**Question: How many missing values are in each column in the dataset?**

There are 16 mising data in the City Column and 1 missing data in the "Select Investers" column.

### Review rows with missing values

Before dealing with missing values, it's important to understand the nature of the missing value that is being filled. Display all rows with missing values from `df_companies`.
"""

# Filter the DataFrame to only include rows with at least one missing value.
# Assign the filtered results to a variable named "df_rows_missing" and display the contents of the variable.

### YOUR CODE HERE ###
df_rows_missing = df_companies[df_companies.isnull().any(axis=1)]
df_rows_missing

"""**Question: Which column has the most data missing?**

The "City" column has the most missing data

### Context-specific missing values

Sometimes, there may be other types of values that are considered missing, such as empty strings and `-1`, `0`, `NaN`, and `NA`. Using one representation for all these missing values is beneficial. Replace any missing values in the dataset with `np.nan`, accessed from the `numpy` library, to simplify the missing values imputation process.


Without replacing the original DataFrame, replace 'Asia' with `np.nan`. Then, find the number of missing values in the dataset.
"""

# Find the number of missing values after replacing 'Asia' with `np.nan`.

### YOUR CODE HERE ###

# Replace 'Asia' with np.nan
df_missing_values = df_companies.replace('Asia', np.nan)

# Find the number of missing values in the dataset
missing_values_count = df_missing_values.isnull().sum()
missing_values_count

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

 Use `isna().sum()` to get the sum of missing values.

</details>

**Question: How many values went missing after changing 'Asia' with `np.nan`?**

After changing 'Asia' with np.nan, there are a total of 327 missing values in the dataset.

**Question: What steps did you take to find missing data?**

To find missing data in the dataset, I performed the following steps:
1. Replaced the occurrence of 'Asia' with np.nan using the replace() function.
2. Used the isna() function to check for missing values in each column of the DataFrame.
3. Applied the sum() function on the result of step 2 to count the number of missing values in each column.

**Question: What observations can be made about the forms and context of missing data?**

Based on the observations from the missing data, we can make the following observations about the forms and context of missing data:
1. The column 'City' has 16 missing values. This suggests that some companies do not have a specific city associated with them in the dataset.
2. The column 'Continent' has 310 missing values. This indicates that the continent information is missing for a significant number of companies in the dataset.
3. The column 'Select Investors' has 1 missing value. This implies that there is one company for which the information about select investors is missing.
4. The other columns, such as 'Company', 'Valuation', 'Date Joined', 'Industry', 'Country/Region', 'Year Founded', 'Funding', 'Year Joined', and 'Valuation_num', do not have any missing values.
These observations indicate that there are missing values in specific columns, which may be due to incomplete or unavailable data for certain companies. It is important to handle these missing values appropriately before performing further analysis on the dataset.

**Question: What other methods could you use to address missing data?**

There are several methods that can be used to address missing data in a dataset. Some common methods include:

1. Deleting rows or columns: If the missing data is limited to a small number of rows or columns, one approach is to simply remove those rows or columns from the dataset. However, this method should be used with caution as it can result in a loss of valuable information.

2. Imputation: Imputation involves filling in missing values with estimated or predicted values. There are various techniques for imputation, such as mean imputation (replacing missing values with the mean of the available values), median imputation, mode imputation, or using more advanced methods like regression imputation or multiple imputation.

3. Forward or backward filling: This method involves propagating the last known value forward or backward to fill in missing values. This approach is suitable when missing values are expected to follow a pattern over time.

4. Using statistical models: Statistical models can be used to predict missing values based on the available data. This approach requires building a model that can estimate the missing values based on the relationships between variables in the dataset.

5. Multiple imputation: Multiple imputation involves creating multiple plausible imputations for the missing values, incorporating uncertainty into the imputation process. This approach can provide more accurate estimates and account for the variability associated with missing data.

The choice of method depends on the nature and extent of missing data, the specific analysis goals, and the characteristics of the dataset. It is important to carefully consider the implications of each method and choose the most appropriate approach for handling missing data in a given context.

## Step 3: Model building

Think of the model you are building as the completed dataset, which you will then use to inform the questions the investor has asked of you.

### Two ways to address missing values

There are several ways to address missing values, which is critical in EDA. The two primary methods are removing them and missing values imputation. Choosing the proper method depends on the business problem and the value the solution will add or take away from the dataset.

Here, you will try both.

To compare the the effect of different actions, first store the original number of values in a variable.
"""

# Store the total number of values in a variable.

### YOUR CODE HERE ###
total_num_values = df_companies.size
total_num_values

"""Now, remove the missing values and count the total number of values in the dataset. Remove all rows containing missing values and store the total number of cells in a variable called `count_dropna_rows`."""

# Drop the rows containing missing values.

### YOUR CODE HERE ###
df_dropna = df_companies.dropna()
# Count the total number of values after removing missing values
count_dropna_rows = df_dropna.size

# Print the total number of values
print("Total number of values after removing missing values:", count_dropna_rows)

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Use `dropna()` function to drop columns with missing values.

</details>

Now, remove all columns containing missing values and store the total number of cells in a variable called `count_dropna_columns`.
"""

# Drop the columns containing missing values.

### YOUR CODE HERE ###
df_dropna_columns = df_companies.dropna(axis=1)

# Count the total number of values after removing missing values
count_dropna_columns = df_dropna_columns.size

# Print the total number of values
print("Total number of values after removing columns with missing values:", count_dropna_columns)

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Provide `axis=1` to `dropna()` function to drop columns with missing values.

</details>

Next, print the percentage of values removed by each method and compare them.
"""

# Print the percentage of values removed by dropping rows.

### YOUR CODE HERE ###
# Calculate the percentage of values removed by dropping rows
percent_removed_rows = (count_dropna_rows / total_num_values) * 100
print("Percentage of values removed by dropping rows:", percent_removed_rows)

# Calculate the percentage of values removed by dropping columns
percent_removed_columns = (count_dropna_columns / total_num_values) * 100
print("Percentage of values removed by dropping columns:", percent_removed_columns)

# Print the percentage of values removed by dropping columns.

### YOUR CODE HERE ###

"""**Question: Which method was most effective? Why?**

The using the count_dropna_rows yielded 98%

Try the second method: imputation. Begin by filling missing values using the backfill method. Then, show the rows that previously had missing values.
"""

# Fill missing values using 'backfill' method.

### YOUR CODE HERE ###
df_filled = df_companies.fillna(method='bfill')

# Show the rows that previously had missing values
rows_with_missing_values = df_companies.loc[df_companies.isnull().any(axis=1)]
rows_with_missing_values

"""**Question: Do the values that were used to fill in for the missing values make sense?**

[Write your response here. Double-click (or enter) to edit.]

Another option is to fill the values with a certain value, such as 'Unknown'. However, doing so doesn’t add any value to the dataset and could make finding the missing values difficult in the future. Reviewing the missing values in this dataset determines that it is fine to leave the values as they are. This also avoids adding bias to the dataset.

## Step 4: Results and evaluation

Now that you've addressed your missing values, provide your investor with their requested data points.

### Companies in the `Hardware` Industry
Your investor is interested in identifying unicorn companies in the `Hardware` industry and one of the following cities: `Beijing`, `San Francisco`, and `London`. They are also interested in companies in the `artificial intelligence` industry in `London`. This information is provided in the following DataFrame.

You have learned that the `pandas` library can be used to `merge()` DataFrames. Merging is useful when two or more DataFrames with similar columns exist that can be combined to create new DataFrames.

Complete the code by merging this DataFrame with `df_companies` DataFrame and create a new DataFrame called `df_invest`.
"""

# Investing search criteria provided as a DataFrame.

### YOUR CODE HERE ###
import pandas as pd

df_search = pd.DataFrame({
    'City': ['Beijing', 'San Francisco', 'London', 'London'],
    'Industry': ['Hardware', 'Hardware', 'Artificial intelligence', 'Hardware']
})

df_invest = pd.merge(df_search, df_companies, left_on=['City', 'Industry'], right_on=['City', 'Industry'])
print(df_invest[['City', 'Industry', 'Company']])

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Review the material about merging DataFrames.

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  Use `merge()` to merge datasets.

</details>

### List of countries by sum of valuation

Group the data by `Country/Region` and sort them by the sum of 'Valuation_num' column.
"""

#Group the data by`Country/Region`

### YOUR CODE HERE ###

df_companies_sum = df_companies.groupby('Country/Region')['Valuation_num'].sum().sort_values(ascending=False).reset_index()


#Print the top 15 values of the DataFrame.

### YOUR CODE HERE ###
df_companies_sum.head(15)

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Review the related material about merging DataFrames.

</details>

**Question: Which countries have the highest sum of valuation?**

The countries that have the highest sum of valuation are USA, China, India, and United Kingdom

Your investor specified that the 4 countries with the highest sum of valuation should not be included in the list. Start by creating a boxplot to visualize the outliers.
"""

# Create a boxlot to identify outliers.

### YOUR CODE HERE ###
sns.boxplot(data=df_companies_sum, y='Valuation_num')

# Set the title of the plot
plt.title('Boxplot of Valuation_num')

# Show the plot.

### YOUR CODE HERE ###
plt.show()

"""In order to visualize the rest of the data properly, consider United States, China, India, and the United Kingdom outliers and remove them."""

# Remove outlier countries.

### YOUR CODE HERE ###
top_countries = ['United States', 'China', 'India', 'United Kingdom']
df_companies_sum_filtered = df_companies_sum[~df_companies_sum['Country/Region'].isin(top_countries)]

"""Now, the data is ready to reveal the top 20 countries with highest company valuations. A data visualization, `sns.barplot` can be used. Complete the following code to plot the data."""

# Create a barplot to compare the top 20 countries with highest company valuations.

### YOUR CODE HERE ###

# Set the figure size
plt.figure(figsize=(12, 6))

# Filter the top 20 countries with highest valuations
top_20_countries = df_companies_sum_filtered.head(20)

# Create the bar plot
sns.barplot(data=top_20_countries, x='Country/Region', y='Valuation_num')

# Set the title and labels
plt.title('Top 20 Countries with Highest Company Valuations')
plt.xlabel('Country/Region')
plt.ylabel('Valuation')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot

### YOUR CODE HERE ###
plt.show()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Select the top 20 rows in `df_companies_sum_outliers_removed`

</details>

<details>
  <summary><h4><strong>Hint 2</strong></h4></summary>

  Select the top 20 rows in `df_companies_sum_outliers_removed` by using `head(20)` function.

</details>

### Plot maps

Your investor has also asked for:
 - A global valuation map of all countries with companies that joined the list after 2020
 - A global valuation map of all countries except `United States`, `China`, `India`, and `United Kingdom` and a separate map for Europe

To create these, plot the data onto maps.

You have learned about using `scatter_geo()` from `plotly.express` library to create plot data on a map. Create a `scatter_geo()` plot that depicts the countries with valuation of companies joined after 2020.
"""

# Plot the sum of valuations per country.

### YOUR CODE HERE ###

# Filter the data for companies that joined after 2020
df_after_2020 = df_companies[df_companies['Year Joined'] > 2020]

# Group the data by country and calculate the sum of valuations
df_grouped = df_after_2020.groupby('Country/Region')['Valuation_num'].sum().reset_index()

# Create the scatter_geo plot
fig = px.scatter_geo(df_grouped, locations='Country/Region', locationmode='country names', color='Valuation_num',
                     hover_name='Country/Region', size='Valuation_num', projection='natural earth')

# Set the title
fig.update_layout(title='Global Valuation of Companies Joined After 2020')

# Show the plot
fig.show()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Filter the `df_companies` by 'Year_Joined'.

</details>
"""

# Plot the sum of valuations per country.

### YOUR CODE HERE ###


# Group the data by country and calculate the sum of valuations
df_grouped = df_companies.groupby('Country/Region')['Valuation_num'].sum().reset_index()

# Sort the data by the sum of valuations in descending order
df_sorted = df_grouped.sort_values('Valuation_num', ascending=False)

# Select the top 20 countries with the highest sum of valuations
df_top20 = df_sorted.head(20)

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Country/Region', y='Valuation_num', data=df_top20)
plt.xlabel('Country/Region')
plt.ylabel('Sum of Valuations')
plt.title('Top 20 Countries with Highest Company Valuations')
plt.xticks(rotation=90)

# Show the plot.

### YOUR CODE HERE ###
plt.show()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

  Use the code in the previous step to complete this section.

</details>

**Question: How is the valuation sum per country visualized in the plot?**

[Write your response here. Double-click (or enter) to edit.]

To create the same map for `europe` only, update the `fig` object to add a new title and also limit the scope of the map to `europe`.
"""

# Update the figure layout.

### YOUR CODE HERE ###
# Group the data by country and calculate the sum of valuations
df_grouped = df_companies.groupby('Country/Region')['Valuation_num'].sum().reset_index()

# Create the scatter_geo plot
fig = px.scatter_geo(df_grouped, locations='Country/Region', locationmode='country names', color='Valuation_num',
                     hover_name='Country/Region', size='Valuation_num', projection='natural earth')

# Set the title
fig.update_layout(title='Sum of Valuations per Country')

# Limit the scope to Europe
fig.update_geos(visible=True, resolution=110, scope='europe')

# Show the plot again.

### YOUR CODE HERE ###
fig.show()

"""<details>
  <summary><h4><strong>Hint 1</strong></h4></summary>

Enter a new text title as string and enter 'europe' to filter `geo_scope`.

</details>

**Question: What steps could you take to further analyze the data?**

To further analyze the data, you can consider the following steps:

1. Perform a trend analysis: Analyze the trends in company valuations over time to identify patterns and potential growth areas. This can help in identifying emerging markets and industries.

2. Conduct a sector analysis: Explore the distribution of valuations across different industries to understand which sectors are attracting the most investment and have the highest valuations. This can provide insights into the potential profitability of specific industries.

3. Investigate funding patterns: Analyze the funding sources and amounts for unicorn companies to understand the funding landscape and investor preferences. This can help in identifying key investors and potential partnership opportunities.

These steps can provide valuable insights into the unicorn company landscape, identify investment opportunities, and guide decision-making for the investor.

## Conclusion

**What are some key takeaways that you learned during this lab?**

1. Data cleaning and preparation are crucial: Dealing with missing values, converting data types, and preparing the data for analysis are essential steps in working with real-world datasets.

2. Data visualization aids understanding: Visualizations such as bar plots, box plots, and maps provide effective ways to explore and present data, helping to identify patterns, outliers, and trends.

3. Merging and grouping data enable insights: Combining datasets through merging and using groupby operations allow for deeper analysis, such as comparing valuations by country or industry, identifying top performers, and making data-driven decisions.

**How would you present your findings from this lab to others? Consider the information you would provide (and what you would omit), how you would share the various data insights, and how data visualizations could help your presentation.**


When presenting the findings from this lab, I would begin by providing an overview of the dataset and the objectives of the analysis. I would focus on the key insights and actionable information that are relevant to the audience's interests.

To share the various data insights, I would utilize a combination of narrative explanations, data visualizations, and supporting statistics. I would highlight the top countries with the highest company valuations, emphasizing the exclusion of outliers like the United States, China, India, and the United Kingdom. I would showcase the bar plot depicting the top 20 countries to visually demonstrate the variations in valuations.

For the global valuation maps, I would present the scatter_geo plots showing the countries with companies that joined the list after 2020. I would point out any notable trends or patterns in the data, such as concentrations of high valuations in specific regions. Additionally, I would showcase the separate map for Europe, highlighting the valuation landscape in that specific area.

Throughout the presentation, I would aim to keep the information concise, focusing on the most relevant and impactful insights. The data visualizations would serve as visual aids to enhance understanding and engagement, helping the audience grasp the information more easily.

Overall, my goal would be to present a clear narrative, supported by relevant data and visualizations, that effectively communicates the key findings and recommendations derived from the analysis.

**Reference**

[Bhat, M.A. *Unicorn Companies*](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies)
"""