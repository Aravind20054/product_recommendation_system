from flask import Flask, request, jsonify, render_template
from surprise.dump import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# File paths
model_filename = 'best_model.pkl'
data_path = 'dataset.csv'
# Globals
best_model = None
df = None
tfidf_vectorizer = None
tfidf_matrix_content = None

# === Load CF model ===
def load_model(filename):
    global best_model
    try:
        _, best_model = load(filename)
        print(" Model loaded successfully")

    except Exception as e:
        print("Error loading model:", e)
        best_model = None

# === Load data and initialize TF-IDF ===
def initialize_data_and_tfidf(data_path):
    global df, tfidf_vectorizer, tfidf_matrix_content
    try:
        df = pd.read_csv(data_path)
        print(f" Data loaded: {df.shape}")

        if 'combined_text' not in df.columns:
            df['combined_text'] = df['product_category_name_english'].fillna('') + \
                                  (' ' + df['Brand'].fillna('')) * 3 + \
                                  (' ' + df['Name'].fillna('')) * 3 + \
                                  ' ' + df['Description'].fillna('') + \
                                  ' ' + df['Tags'].fillna('')

        tfidf_vectorizer = TfidfVectorizer(
                                        stop_words='english',
                                        strip_accents='unicode',
                                    )
        tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['combined_text'])
        print(f" TF-IDF initialized: {tfidf_matrix_content.shape}")
    except Exception as e:
        print(f" Error initializing TF-IDF: {e}")

# === Content-based recommendation ===
def Content_Base_Recomendation(dataframe, search_term, top_n=10):
    try:
        search_vector = tfidf_vectorizer.transform([search_term])
        cos_sim = cosine_similarity(search_vector, tfidf_matrix_content)
        similar_items = list(enumerate(cos_sim[0]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
        Top_similar_items = [item for item in similar_items if item[1] > 0.0][:top_n]
        recommended_indexes = [x[0] for x in Top_similar_items]

        valid_recomended_indexes = [idx for idx in recommended_indexes if idx < len(dataframe)]
        if not valid_recomended_indexes:
            return pd.DataFrame()

        recommended_items = dataframe.iloc[valid_recomended_indexes][['product_id', 'Name', 'Brand', 'ReviewCount', 'review_score']]        
        return recommended_items
    except Exception as e:
        print(f" Content-based error: {e}")
        return pd.DataFrame()

# === Hybrid recommendation combining CF + CB ===
def hybrid_recommendations(user_id, search_term=None, cf_model=None, cb_function=None, dataframe=None, top_n=10):
    try:
        if cf_model is not None and tfidf_matrix_content is not None and search_term is not None:
            cb_recs = cb_function(dataframe, search_term, top_n=top_n * 2)
            if not cb_recs.empty:
                cb_item_ids = []
                for _, row in cb_recs.iterrows():
                    match = dataframe[(dataframe['Name'] == row['Name']) & (dataframe['Brand'] == row['Brand'])]
                    if not match.empty:
                        cb_item_ids.append(match.iloc[0]['product_id'])

                cf_predictions = [cf_model.predict(user_id, iid) for iid in cb_item_ids if not pd.isna(iid)]
                recommended_items = []
                for pred in cf_predictions:
                    if pred.iid in dataframe['product_id'].values:
                        details = dataframe[dataframe['product_id'] == pred.iid].iloc[0]
                        recommended_items.append({
                            'Name': details['Name'],
                            'Brand': details['Brand'],
                            'ReviewCount': details['ReviewCount'],
                            'Rating': pred.est
                        })
                return pd.DataFrame(recommended_items).sort_values(by='Rating', ascending=False).head(top_n)
            else:
                print(" No CB recommendations to refine.")
                return pd.DataFrame()
        elif search_term is not None:
            print(" Cold-start user. Using CB only.")
            return cb_function(dataframe, search_term, top_n)
        else:
            print(" Insufficient input for recommendation.")
            return pd.DataFrame()
    except Exception as e:
        print(f" Hybrid recommendation error: {e}")
        return pd.DataFrame()

# === Initialize Flask ===
app = Flask(__name__)
load_model(model_filename)
initialize_data_and_tfidf(data_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if best_model is None or df is None or tfidf_vectorizer is None or tfidf_matrix_content is None:
        return jsonify({"error": "System not fully initialized."}), 500

    if request.method == 'POST':
        payload = request.get_json()
        user_id = float(payload.get('user_id')) if 'user_id' in payload else None
        search_term = payload.get('search_term', None)
        top_n = int(payload.get('top_n', 10))
    else:
        user_id_str = request.args.get('user_id')
        user_id = float(user_id_str) if user_id_str else None
        search_term = request.args.get('search_term', type=str)
        top_n = request.args.get('top_n', default=10, type=int)

    if user_id is None and search_term is None:
        return jsonify({"error": "Please provide either 'user_id' or 'search_term'."}), 400

    recommendations = hybrid_recommendations(
        user_id=user_id,
        search_term=search_term,
        cf_model=best_model,
        cb_function=Content_Base_Recomendation,
        dataframe=df,
        top_n=top_n
    )

    if recommendations.empty:
        return jsonify({"message": "No recommendations found."}), 404
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))