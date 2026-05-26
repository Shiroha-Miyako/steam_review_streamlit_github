# Steam Review Intelligence Assistant

A Streamlit app for ISOM5240 Deep Learning Business Applications with Python.

## What it does

This app analyzes Steam player reviews using two fine-tuned Hugging Face models:

1. Steam review sentiment classification
2. Steam issue category classification

The app outputs:

- sentiment
- issue category
- confidence scores
- priority level
- suggested developer action
- executive summary
- visual dashboard
- downloadable analyzed results

## Input modes

1. Single review input
2. CSV upload with a `review_text` column
3. Fetch recent reviews by Steam App ID

## Where to find Steam App ID

Steam App ID appears in the store URL.

Example:

`https://store.steampowered.com/app/1086940/...`

The App ID is:

`1086940`

