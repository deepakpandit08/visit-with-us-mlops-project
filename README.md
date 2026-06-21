
# Visit with Us - Wellness Tourism Package MLOps Project

This project builds an end-to-end MLOps pipeline for predicting whether a customer is likely to purchase the Wellness Tourism Package.

## Project Objective

The objective is to automate customer purchase prediction using machine learning and MLOps practices.

## Main Components

- Data registration on Hugging Face Dataset Hub
- Data cleaning and train/test split
- Model training and hyperparameter tuning
- Experiment tracking through saved result files
- Best model registration on Hugging Face Model Hub
- Streamlit app deployment on Hugging Face Spaces using Docker
- GitHub Actions pipeline for automation

## Model Used

The final selected model is Random Forest.

## Hugging Face Links

Dataset: https://huggingface.co/datasets/deepakpandit08/visit-with-us-tourism-data

Model: https://huggingface.co/deepakpandit08/visit-with-us-wellness-model

Space: https://huggingface.co/spaces/deepakpandit08/visit-with-us-wellness-app

## GitHub Actions

The workflow file is available at:

.github/workflows/pipeline.yml

The pipeline runs automatically when changes are pushed to the main branch.
