from transformers import pipeline
import streamlit as st
import pandas as pd
import numpy as np
import tqdm

REDO = False
NUMBER_O_ROWS = 1_000


# DATA

# Load Dataframes from CSV
df_1_raw = pd.read_csv("customer_shopping_data.csv", encoding='ISO-8859-1')
df_2_raw = pd.read_csv("Online_Retail.csv", encoding='ISO-8859-1')

# Keep only relevant columns
df_1_clean = df_1_raw[['category', 'quantity', 'price']]
df_2_clean = df_2_raw[['Description', 'Quantity', 'UnitPrice']]

# Rename UnitPrice to price and unify quantity column names
df_2_clean.rename(columns={'UnitPrice': 'price', 'Quantity': 'quantity'}, inplace=True)

# Convert price (df_1: Turkish Lira to df_2: British Pounds)
df_1_clean['price'] = (df_1_clean['price'] * 0.023).round(2)

# Remove rows with empty values for description
df_2_clean = df_2_clean.dropna(subset=['Description'])
df_2_clean = df_2_clean[df_2_clean['Description'].str.strip() != '']

## Description Category Mapping:
# Get label (i.e. 'category'-column from df_1)
labels = df_1_clean['category'].unique()

# Classifier
zero_shot_classifier = pipeline(model="facebook/bart-large-mnli")

# New result dataframe
df_2_result = df_2_clean.copy()[:NUMBER_O_ROWS]

# Add 'category'-column to df_2 with empty values
df_2_result['category'] = ''

if REDO:
    print("Starting classification of descriptions...")
    # Iterate over rows of DF_2 and assign category based on description (Zero-Shot-Classification) with progress bar
    with tqdm.tqdm(total=df_2_result.shape[0], desc="Classifying") as pbar:
        for index, row in df_2_result.iloc[:NUMBER_O_ROWS].iterrows():
            description = row['Description']
            result = zero_shot_classifier(description, labels)
            category = result['labels'][0]
            df_2_result.at[index, 'category'] = category
            pbar.update(1)

    # Save result to CSV
    df_2_result.to_csv('Online_Retail_with_category.csv', index=False, encoding='ISO-8859-1')

else:
    # Read data from last time
    df_2_result = pd.read_csv('Online_Retail_with_category.csv', encoding='ISO-8859-1')

# Drop Description column from df_2
df_2_result_drop = df_2_result.drop(columns=['Description'])

# Truncate df_1
df_1_result = df_1_clean.copy()[:NUMBER_O_ROWS]

# Combine results
df_combined = pd.concat([df_1_result, df_2_result_drop], ignore_index=True)

## DISPLAY
# Settings
st.set_page_config(page_title="Information Integration Project", page_icon="📊",layout="wide")
# Title
st.title("Information Integration Project 📊⚙️📝")

# https://docs.streamlit.io/develop/quick-reference/cheat-sheet
# https://docs.streamlit.io/develop/api-reference
# Scenario
st.markdown("""
            In the wake of a strategic business merger, a British online retailer has successfully acquired a Turkish e-commerce company. 
            This project focuses on integrating and consolidating the distinct databases of both companies to streamline operations, enhance data consistency, and support unified business processes.
            
            ### Business Merge of two online retailer
""")
# Original databases
col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_1_raw)
    st.caption("Customer Shopping Data (Turkey)")

with col2:
    st.dataframe(df_2_raw)
    st.caption("Online Data (UK)")

# Cleaned databases
st.markdown("""
            The managment is interessted in a complete overview of the columns price and quantity. The quantity columns are mergeable, while the price column of one of the data sets just needs to be converted (i.e. from Turkish Liras (₺) to British Pounds (£)). Furthermore, the columns description of dataframe UK and category of dataframe Turkiye should link the two datasts together.

            ### Cleaned Data:
""")

# Cleaned databases
col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_1_clean)
    st.caption("Cleaned Customer Shopping Data (Turkey) with converted prices")

with col2:
    st.dataframe(df_2_clean)
    st.caption("Cleaned Online Data (UK) with renamed columns")


st.markdown("""
            ### Mapping of Descriptions to Categories:
            In order to map the descriptions of the UK data set to the categories of the Turkish data set, a zero-shot-classification model is used. The model is able to classify the descriptions of the UK data set into the respective labels (i.e. the categories of the Turkish data set). Hence, the resulting datasets are mergeable.

""")
# Display all labels
st.write("Labels: ", ", ".join(labels))

# Matched databases
col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_1_result)
    st.caption("Customer Shopping Data (Turkey)")

with col2:
    st.dataframe(df_2_result)
    st.caption("Online Data (UK) with mapped categories and dropped description")

# Combined database
st.markdown("""
            ### Combined Data:
            The combined data set contains all rows of the aforementioned cleaned and mapped data sets. The columns of the two data sets are merged based on the columns 'category', 'quantity' and 'price'. The resulting data set provides a unified view of the items in stock of the two companies.
""")

# Combined databases
_, col2, _ = st.columns([1, 1, 1])

with col2:
    st.dataframe(df_combined)
    st.caption("Combined dataframe of the two online retailers")