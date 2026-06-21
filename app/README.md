---
title: Visit With Us Wellness App
emoji: 🧳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Visit with Us - Wellness Tourism Package Predictor

This is a Streamlit app for predicting whether a customer is likely to purchase the Wellness Tourism Package.

The app loads the trained Random Forest model from the Hugging Face Model Hub and uses customer details to generate a purchase prediction.

## Files

- app.py: Streamlit frontend application
- requirements.txt: Python dependencies
- Dockerfile: Docker configuration for deployment
- upload_to_space.py: Script to upload deployment files to Hugging Face Space
