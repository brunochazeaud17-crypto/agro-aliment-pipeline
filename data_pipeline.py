import yfinance as yf
import pandas as pd
import requests
import sqlite3
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from openbb import obb

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


import pandas as pd
import sqlite3

# URL du fichier CSV public de la FAO
FAO_FPI_URL = "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data.csv"

def fetch_fao_fpi():
    """Télécharge l'Indice FAO des prix alimentaires."""
    print("📥 Téléchargement de l'Indice FAO des prix alimentaires...")
    try:
        df_fao = pd.read_csv(FAO_FPI_URL, encoding='ISO-8859-1')
        # La colonne date est généralement 'Date'
        df_fao['Date'] = pd.to_datetime(df_fao['Date'])
        df_fao.set_index('Date', inplace=True)
        # On sélectionne la colonne 'Food Price Index'
        df_fao_fpi = df_fao[['Food Price Index']].dropna()
        print(f"✅ Indice FAO récupéré ! ({len(df_fao_fpi)} mois de données)")
        return df_fao_fpi
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de l'indice FAO : {e}")
        return pd.DataFrame()


# Ajoute cette importation en haut du fichier
from openbb import obb

def fetch_usda_stocks():
    """Récupère les stocks de maïs et de blé depuis l'USDA PSD."""
    print("📦 Récupération des stocks céréaliers (USDA)...")
    try:
        # Exemple pour le maïs (corn)
        df_corn = obb.commodity.psd_data(commodity='corn', country='world', attribute='ending_stocks', start_year=2020)
        df_corn['Date'] = pd.to_datetime(df_corn['date'])
        df_corn = df_corn[['Date', 'value']].rename(columns={'value': 'corn_stocks'})
        df_corn.set_index('Date', inplace=True)
        
        # Exemple pour le blé (wheat)
        df_wheat = obb.commodity.psd_data(commodity='wheat', country='world', attribute='ending_stocks', start_year=2020)
        df_wheat['Date'] = pd.to_datetime(df_wheat['date'])
        df_wheat = df_wheat[['Date', 'value']].rename(columns={'value': 'wheat_stocks'})
        df_wheat.set_index('Date', inplace=True)
        
        # Fusion des données
        df_stocks = pd.merge(df_corn, df_wheat, left_index=True, right_index=True, how='outer')
        print(f"✅ Stocks USDA récupérés ! ({len(df_stocks)} années de données)")
        return df_stocks
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des stocks USDA : {e}")
        return pd.DataFrame()


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

# Sauvegarde indice FAO
df_fao_fpi = fetch_fao_fpi()
if not df_fao_fpi.empty:
    df_fao_fpi.to_sql('fao_fpi', conn, if_exists='replace')
    print("   - Table 'fao_fpi' sauvegardée.")

# Sauvegarde stock céréale
df_stocks = fetch_usda_stocks()
if not df_stocks.empty:
    df_stocks.to_sql('usda_stocks', conn, if_exists='replace')
    print("   - Table 'usda_stocks' sauvegardée.")

conn.close()
print(" Terminé ! Base de données mise à jour avec succès.")
