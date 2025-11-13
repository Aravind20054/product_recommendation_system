import streamlit as st
import pandas as pd
import requests
from surprise.dump import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# === Load model from GitHub ===
model_url = "https://raw.githubusercontent.com/hoshikrana/Product-Recommendation-System-Project/main/best_model.pkl"
model_path = "best_model.pkl"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        f.write(requests.get(model_url).content)

_, best_model = load(model_path)

# === Load dataset from GitHub ===
dataset_url = "https://raw.githubusercontent.com/hoshikrana/Product-Recommendation-System-Project/main/dataset.csv"
df = pd.read_csv(dataset_url)

# === Build TF-IDF matrix ===
if "combined_text" not in df.columns:
    df['combined_text'] = df['product_category_name_english'].fillna('') + \
                          (' ' + df['Brand'].fillna('')) * 3 + \
                          (' ' + df['Name'].fillna('')) * 3 + \
                          ' ' + df['Description'].fillna('') + \
                          ' ' + df['Tags'].fillna('')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['combined_text'])

# === Recommendation functions ===
def content_based_recommendation(search_term, top_n=10):
    search_vector = tfidf_vectorizer.transform([search_term])
    cos_sim = cosine_similarity(search_vector, tfidf_matrix_content)
    similar_items = sorted(list(enumerate(cos_sim[0])), key=lambda x: x[1], reverse=True)
    recommended_indexes = [x[0] for x in similar_items if x[1] > 0.0][:top_n]
    return df.iloc[recommended_indexes][['product_id', 'Name', 'Brand', 'ReviewCount', 'review_score']]

def hybrid_recommendations(user_id, search_term, top_n=10):
    cb_recs = content_based_recommendation(search_term, top_n * 2)
    if cb_recs.empty:
        return pd.DataFrame()
    
    cb_item_ids = []
    for _, row in cb_recs.iterrows():
        match = df[
            (df['Name'].str.strip().str.lower() == str(row['Name']).strip().lower()) &
            (df['Brand'].str.strip().str.lower() == str(row['Brand']).strip().lower())
        ]
        if not match.empty:
            cb_item_ids.append(match.iloc[0]['product_id'])

    cf_predictions = [best_model.predict(user_id, iid) for iid in cb_item_ids if not pd.isna(iid)]
    recommended_items = []
    for pred in cf_predictions:
        if pred.iid in df['product_id'].values:
            details = df[df['product_id'] == pred.iid].iloc[0]
            recommended_items.append({
                'Name': details['Name'],
                'Brand': details['Brand'],
                'ReviewCount': details['ReviewCount'],
                'Rating': round(pred.est, 2)
            })
    return pd.DataFrame(recommended_items).sort_values(by='Rating', ascending=False).head(top_n)

# === Streamlit UI ===
st.title("🛍️ Hybrid Product Recommendation System")
st.markdown("Enter a **User ID** and a **Search Term** to get personalized product recommendations.")

user_id = st.number_input("User ID", min_value=0.0, format="%.0f", value=0.0)
search_term = st.text_input("Search Term", value="headphones")  # Pre-filled example
top_n = st.slider("Number of Recommendations", 5, 20, 10)
if st.button("Recommend"):
    with st.spinner("🔄 Generating recommendations..."):
        search_term = search_term.strip()

        if search_term != "":
            st.write(f"🔍 Inputs → User ID: `{int(user_id)}`, Search Term: `{search_term}`")
            result = hybrid_recommendations(user_id, search_term, top_n)

            if not result.empty:
                st.subheader("🔗 Hybrid Recommendations")
                st.success(f"✅ Found {len(result)} recommendations")
                st.dataframe(result)
            else:
                st.warning("⚠️ Collaborative filtering returned no results. Showing content-based recommendations instead.")
                fallback = content_based_recommendation(search_term, top_n)
                if not fallback.empty:
                    st.subheader("🎯 Content-Based Recommendations")
                    st.dataframe(fallback)
                else:
                    st.info("No recommendations found for your input.")
        else:
            st.warning("Please enter a valid search term.")