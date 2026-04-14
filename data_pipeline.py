import yfinance as yf
import pandas as pd
import requests
import sqlite3
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# --- CONFIGURATION ---
API_KEY = "5b7edee298df4374b80fa69f3c032e18"
DB_NAME = "agri_data.db"

# --- 1. COLLECTE DES PRIX (yfinance) ---
print("📥 Téléchargement des données boursières...")
tickers = {
    "Yara (Norvège)": "YAR.OL",
    "OCI (Pays-Bas)": "OCI.AS",
    "ICL (Israël)": "ICL",
    "CF Industries (USA/UK)": "CF",
    "Grupa Azoty (Pologne)": "ATT.WA",
    "K+S (Allemagne)": "SDF.DE"
}

df_prices = yf.download(list(tickers.values()), start="2022-01-01", interval="1d")['Close']
inv_tickers = {v: k for k, v in tickers.items()}
df_prices.rename(columns=inv_tickers, inplace=True)
print(f"✅ Prix récupérés ! ({len(df_prices)} jours de cotation)")

# --- 2. COLLECTE DES ACTUALITÉS (NewsAPI) ---
print("📰 Récupération des articles de presse...")
query = '("fertilizer" OR "agriculture") AND ("Europe" OR "gas" OR "Middle East" OR "Iran")'
url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

if data.get('status') == 'ok':
    articles = data['articles']
    df_news = pd.DataFrame(articles)[['publishedAt', 'title', 'description', 'source']]
    df_news['source'] = df_news['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
    print(f"✅ Actualités récupérées ! ({len(df_news)} articles)")
else:
    print("❌ Erreur NewsAPI :", data.get('message'))
    df_news = pd.DataFrame() # DataFrame vide pour éviter les bugs si ça plante

# --- 3. ANALYSE DE SENTIMENT (NLP) ---
if not df_news.empty:
    print("🧠 Analyse du sentiment des titres...")
    nltk.download('vader_lexicon', quiet=True) # quiet=True pour ne pas polluer l'écran
    sia = SentimentIntensityAnalyzer()

    def calculer_sentiment(texte):
        if pd.isna(texte):
            return 0
        return sia.polarity_scores(str(texte))['compound']

    df_news['score_sentiment'] = df_news['title'].apply(calculer_sentiment)
    print("✅ Analyse de sentiment terminée !")

# --- 4. SAUVEGARDE DANS LA BASE DE DONNÉES ---
print("💾 Sauvegarde dans SQLite...")
conn = sqlite3.connect(DB_NAME)

# On sauvegarde les prix
df_prices.to_sql('stock_prices', conn, if_exists='replace')

# On sauvegarde les news (si on en a)
if not df_news.empty:
    df_news.to_sql('news_sentiment', conn, if_exists='replace', index=False)

conn.close()
print("🎉 Terminé ! Base de données mise à jour avec succès.")