from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

df = pd.read_csv("my work/Data_final.csv")

FEATURES = [
    "O_score", "C_score", "E_score", "A_score", "N_score",
    "Numerical Aptitude", "Spatial Aptitude", "Perceptual Aptitude",
    "Abstract Reasoning", "Verbal Reasoning"
]

trait_labels = {
    "O_score": "openness to experience",
    "C_score": "conscientiousness",
    "E_score": "extraversion",
    "A_score": "agreeableness",
    "N_score": "emotional stability",
    "Numerical Aptitude": "numerical aptitude",
    "Spatial Aptitude": "spatial aptitude",
    "Perceptual Aptitude": "perceptual accuracy",
    "Abstract Reasoning": "abstract reasoning",
    "Verbal Reasoning": "verbal communication"
}

trait_explanations = {
    "O_score": "creative thinking and innovation",
    "C_score": "organization, responsibility, and reliability",
    "E_score": "confidence in social and team settings",
    "A_score": "cooperation and empathy",
    "N_score": "resilience and calm under pressure",
    "Numerical Aptitude": "quantitative analysis and logic",
    "Spatial Aptitude": "working with diagrams, space, or 3D models",
    "Perceptual Aptitude": "attention to visual details and patterns",
    "Abstract Reasoning": "solving unfamiliar problems and spotting patterns",
    "Verbal Reasoning": "communicating ideas clearly and persuasively"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not all(f in data for f in FEATURES):
            return jsonify({"error": "Missing required features"}), 400

        
        user_vector = np.array([[float(data[f]) for f in FEATURES]])
        career_vectors = df[FEATURES].values

        similarities = cosine_similarity(user_vector, career_vectors)[0]
        df["similarity"] = similarities

        top_matches = df.sort_values(by="similarity", ascending=False).head(3).reset_index(drop=True)

    
        sorted_traits = sorted(zip(FEATURES, user_vector[0]), key=lambda x: x[1], reverse=True)
        top_traits = sorted_traits[:3]

        results = []
        for i, row in top_matches.iterrows():
            career = row["Career"]
            score = round(row["similarity"] * 100, 2)

        
            if i == 0:
                reason = (
                    f"Your strong {trait_labels[top_traits[0][0]]} and high ability in {trait_labels[top_traits[1][0]]} "
                    f"make you ideal for this career. It demands excellence in {trait_explanations[top_traits[0][0]]}, "
                    f"alongside strength in {trait_explanations[top_traits[1][0]]} and {trait_explanations[top_traits[2][0]]}."
                )
            elif i == 1:
                reason = (
                    f"This field aligns well with your high {trait_labels[top_traits[0][0]]} and {trait_labels[top_traits[1][0]]}. "
                    f"Professionals here often thrive with strong {trait_explanations[top_traits[0][0]]} "
                    f"and a solid foundation in {trait_explanations[top_traits[2][0]]}."
                )
            else:
                reason = (
                    f"Your top strengths in {trait_labels[top_traits[0][0]]}, {trait_labels[top_traits[1][0]]}, and "
                    f"{trait_labels[top_traits[2][0]]} support success in this area. "
                    f"It benefits from sharp {trait_explanations[top_traits[1][0]]} and dependable {trait_explanations[top_traits[2][0]]}."
                )

            results.append({
                "career": career,
                "confidence": score,
                "reason": reason
            })

        print("✅ Cosine Matches:", results)
        return jsonify({"matches": results})

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)