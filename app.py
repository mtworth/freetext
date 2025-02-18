import streamlit as st
import pandas as pd
import time
from openai import OpenAI
from pydantic import BaseModel
import os  # For accessing environment variables

# Retrieve OpenAI API key from environment variable
openai_api_key = st.secrets["OpenAI_key"] 

class ThemeExtraction(BaseModel):
    themes: list[str]

class AssignedThemes(BaseModel):
    AssignedThemes: list[str]

def extract_themes_from_column(df_column, num_samples, focus_themes, openai_api_key):
    """
    Extracts themes from a randomly sampled subset of a Pandas DataFrame column using GPT-4o.
    
    :param df_column: Pandas Series (a column from a DataFrame)
    :param num_samples: Integer, number of random rows to sample
    :param focus_themes: List of themes the model should prioritize
    :param openai_api_key: String, your OpenAI API key
    :return: List of themes identified by GPT-4o
    """
    
    if not isinstance(df_column, pd.Series):
        raise ValueError("df_column must be a Pandas Series")
    
    sampled_texts = df_column.dropna().sample(n=min(num_samples, len(df_column)), random_state=42).tolist()
    
    client = OpenAI(api_key=openai_api_key)
    
    messages = [
        {
            "role": "user",
            "content": f"""
            Analyze the following text excerpts and identify key feedback themes. 
            Ensure the following themes are considered: {', '.join(focus_themes)}.
            Try to balance specificity and generality. 
            Provide a list of feedback themes only, without additional explanation.

            Text excerpts:
            {sampled_texts}
            """
        }
    ]
    
    response = client.beta.chat.completions.parse(
        messages=messages,
        model="gpt-4o-mini",
        response_format=ThemeExtraction,
        temperature=0
    )
    
    themes = eval(response.choices[0].message.content)["themes"]
    
    return themes

def tag_themes_to_rows(df, text_column, themes, openai_api_key, progress_bar):
    """
    Tags each row in a DataFrame with one or more relevant themes or 'Other'.

    :param df: Pandas DataFrame, containing the text data
    :param text_column: String, the name of the column containing the text
    :param themes: List, the extracted themes to match the text against
    :param openai_api_key: String, your OpenAI API key
    :param progress_bar: Streamlit progress bar object
    :return: DataFrame, with a new column 'Theme_Tags' containing relevant themes or 'Other' if there are no fits
    """
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
    total_rows = len(df)

    def assign_theme(row_text, row_index):
        messages = [
            {
                "role": "user",
                "content": f"""
                Here is a statement: \"{row_text}\"
                
                Themes to consider: {themes}
                
                Please provide the most suitable theme(s) from the list for this statement.
                If it doesn't fit any theme, write 'Other'.
                """
            }
        ]
        response = client.beta.chat.completions.parse(
                messages=messages,
                model="gpt-4o-mini",
                response_format=AssignedThemes,
                temperature=0
            )
        Assignedthemes = eval(response.choices[0].message.content)["AssignedThemes"]
        
        # Update progress bar
        progress = int((row_index + 1) / total_rows * 100)
        progress_bar.progress(progress, text=f"Processing row {row_index + 1} of {total_rows}")
        
        return Assignedthemes

    df["Theme_Tags"] = [assign_theme(text, idx) for idx, text in enumerate(df[text_column])]
    return df

st.title("Survey Text Analyzer")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Uploaded file is not a CSV or Excel file.")

        st.write("Top 5 Rows of the Uploaded Data:")
        st.dataframe(df.head())

        column = st.selectbox("Choose a column to extract themes from:", df.columns)
        num_samples = st.number_input("Number of samples to analyze:", min_value=1, max_value=100, value=10)
        focus_themes = st.text_area("Specify key themes the model should consider (comma-separated)")

        if st.button("Extract Themes"):
            progress_text = "Processing rows..."
            my_bar = st.progress(0, text=progress_text)

            if openai_api_key:
                focus_themes_list = [theme.strip() for theme in focus_themes.split(",") if theme.strip()]
                themes = extract_themes_from_column(df[column], num_samples, focus_themes_list, openai_api_key)
                st.write("Extracted Themes:")
                st.write(themes)
                
                df_tagged = tag_themes_to_rows(df, column, themes, openai_api_key, my_bar)
                theme_counts = df_tagged["Theme_Tags"].explode().value_counts()
                st.write("Theme Counts:")
                st.write(theme_counts)

                csv = df_tagged.to_csv(index=False).encode('utf-8')
                st.download_button("Download Labeled Data", data=csv, file_name="labeled_data.csv", mime="text/csv")
            else:
                st.error("API key not found. Please ensure the OPENAI_API_KEY environment variable is set.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
