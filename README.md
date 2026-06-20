# Bengaluru Event-Driven Congestion Intelligence

AI-powered system to forecast traffic impact of planned events and recommend police resource deployment.

## Problem
Political rallies, festivals, and construction activities cause unpredictable traffic congestion in Bengaluru. Resource deployment today is experience-driven, not data-driven.

## Solution
- Spatial-temporal correlation engine linking planned events to historical incident impact (5km radius, 6-hour window)
- XGBoost classifier predicting impact severity (None/Low/Medium/High)
- Resource recommendation engine outputting officer count, barricading needs, and diversion routes

## Results
- Test Accuracy: 42%
- High-Impact Recall: 55%
- Trained on 8,173 real Bengaluru traffic incidents (Nov 2023–Apr 2024)

## Live Demo
[Hugging Face Space link here]

## Files
- `flipkart_2.0.ipynb` — full notebook with EDA, model training, recommendation engine
- `requirements.txt` — dependencies

## How to Run
See Instructions to Run in HackerEarth submission, or run notebook cells sequentially in Jupyter.
