import yfinance as yf
import pandas as pd
import requests
import sqlite3
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# --- CONFIGURATION ---
API_KEY = os.environ.get("NEWS_API_KEY")
DB_NAME = "agri_data.db"

if not API_KEY:
    raise ValueError("🚨 AUCUNE CLÉ API TROUVÉE ! Vérifie tes secrets GitHub ou ton environnement local.")

# --- 1. COLLECTE DES PRIX DES ACTIONS (yfinance) ---
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
print(f"✅ Prix des actions récupérés ! ({len(df_prices)} jours de cotation)")

# --- 2. COLLECTE DES MATIÈRES PREMIÈRES (Gaz et Blé) ---
print("📥 Téléchargement des données des matières premières...")
commodities_tickers = {
    "Gaz_Nat_EU": "TTF=F",      # Gaz naturel néerlandais (référence européenne)
    "Ble_Chicago": "ZW=F"      # Blé (Chicago SRW Wheat Futures)
}

df_commodities = yf.download(list(commodities_tickers.values()), start="2022-01-01", interval="1d")['Close']
df_commodities.rename(columns={v: k for k, v in commodities_tickers.items()}, inplace=True)
print(f"✅ Matières premières récupérées ! ({len(df_commodities)} jours de cotation)")

# --- 3. COLLECTE DES ACTUALITÉS (NewsAPI) ---
print("📰 Récupération des articles de presse...")
query = '("fertilizer" OR "agriculture") AND ("Europe" OR "gas" OR "Middle East" OR "Iran")'
url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

if data.get('status') == 'ok':
    articles = data['articles']
    df_news = pd.DataFrame(articles)[['publishedAt', 'title', 'description', 'source']]
    df_news['source'] = df_news['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
    print(f" Actualités récupérées ! ({len(df_news)} articles)")
else:
    print("❌ Erreur NewsAPI :", data.get('message'))
    df_news = pd.DataFrame()

# --- 4. ANALYSE DE SENTIMENT (NLP) ---
if not df_news.empty:
    print("🧠 Analyse du sentiment des titres...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

    def calculer_sentiment(texte):
        if pd.isna(texte):
            return 0
        return sia.polarity_scores(str(texte))['compound']

    df_news['score_sentiment'] = df_news['title'].apply(calculer_sentiment)
    print(" Analyse de sentiment terminée !")

# --- 5. SAUVEGARDE DANS LA BASE DE DONNÉES ---
print("💾 Sauvegarde dans SQLite...")
conn = sqlite3.connect(DB_NAME)

# Sauvegarde des prix des actions
df_prices.to_sql('stock_prices', conn, if_exists='replace')
print("   - Table 'stock_prices' sauvegardée.")

# Sauvegarde des matières premières
df_commodities.to_sql('commodity_prices', conn, if_exists='replace')
print("   - Table 'commodity_prices' sauvegardée.")

# Sauvegarde des news (si on en a)
if not df_news.empty:
    df_news.to_sql('news_sentiment', conn, if_exists='replace', index=False)
    print("   - Table 'news_sentiment' sauvegardée.")
else:
    print("   - Aucune actualité à sauvegarder.")

conn.close()
print(" Terminé ! Base de données mise à jour avec succès.")
