# Career Recommendation Python Script (Flask + Cosine Similarity)

This script is Flask-based REST API that recommends suitable career paths for users based on their personality (OCEAN model) and aptitude test scores gotten from the career system using **cosine similarity**.

It needs to be integrated with the system/frontend for it to work properly.
It works through a simple `POST` request

- Accepts 10 user test scores
- Compares user vector with dataset (`Data_final.csv`)
- Returns top 3 most similar career paths
- Generates detailed explanations based on user strengths
