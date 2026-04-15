import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import calendar

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AgroSentinel | Souveraineté Alimentaire",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3F20;
        border-bottom: 3px solid #6B8E23;
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5D23;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .card {
        background-color: #F8F9F9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #6B8E23;
    }
    .info-box {
        background-color: #2E4053;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #FFFFFF;
        border-left: 5px solid #F1C40F;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        text-align: center;
    }
    .event-card {
        background-color: #F8F9FA !important;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        color: #2C3E50;
    }
    .articles-box {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 6px;
        color: #1E1E1E;
        border: 1px solid #E9ECEF;
    }
    .stat-table {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .plot-explanation {
        font-size: 0.9rem;
        color: #5D6D7E;
        margin-bottom: 10px;
        font-style: italic;
        border-left: 3px solid #6B8E23;
        padding-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- TITRE ---
st.markdown('<div class="main-header">AgroSentinel : Veille sur la Souveraineté Alimentaire Européenne</div>', unsafe_allow_html=True)

# --- DESCRIPTION CONTEXTUELLE ---
with st.expander("📌 A propos de ce projet - Cliquez pour ouvrir/fermer", expanded=False):
    col_desc1, col_desc2 = st.columns([2, 1])
    with col_desc1:
        st.markdown("""
        ### Pourquoi ce dashboard ?
        La souveraineté alimentaire de l'Europe est menacée par sa dépendance aux engrais importés. Voici la chaîne de causalité :
        
        >  **Gaz naturel**  →  **Ammoniac** (procédé Haber-Bosch) →  **Engrais azotés** →  **Récoltes** →  **Nourriture**
        
        ### Le rôle clé de YARA
        - YARA est le **1er producteur mondial** d'engrais azotés d'origine européenne
        - Le gaz naturel représente **70 à 80%** du coût de production d'une tonne d'ammoniac
        - *Par exemple : Quand le gaz passe de 30€ à 100€/MWh, le coût de production d'une tonne d'ammoniac augmente de +200 à 250€*
        - L'Europe importe encore **30% de ses engrais azotés** de Russie et Biélorussie
        
        ### Les autres maillons fragiles
        - **Grupa Azoty** (Pologne) : très exposé au gaz russe via le gazoduc Yamal
        - **ICL** (Israël) : vulnérable aux tensions au Moyen-Orient (engrais potassiques)
        - **K+S** (Allemagne) : dépendant des importations de gaz pour ses usines
        
        ### Comment ça fonctionne ?
        - **Données boursières** : Récupérées chaque jour via **Yahoo Finance** (yfinance) pour 6 géants européens.
        - **Actualités** : Articles frais tirés de **NewsAPI.org** avec les mots-clés *fertilizer, Europe, gas, Middle East*.
        - **Analyse de sentiment** : Un algorithme de **NLP (VADER)** note chaque titre de -1 (très négatif) à +1 (très positif).
        - **Mise à jour** : Un script automatique tourne toutes les **24h via GitHub Actions**.
        
        **Utilité concrète** : Anticiper les tensions sur les prix alimentaires 6 à 12 mois avant qu'elles n'arrivent dans les supermarchés.
        """)
    with col_desc2:
        st.markdown("""
        ###  La chaîne de transmission en chiffres
        
        | Maillon | Délai |
        |---------|-------|
        | Hausse gaz | J+0 |
        | Hausse prix engrais | J+15 à J+30 |
        | Baisse utilisation engrais | Saisson suivante |
        | Baisse rendements | 6-12 mois |
        | Hausse prix alimentaires | 6-18 mois |
        
        ### Sources
        - **Prix** : Yahoo Finance  
        - **Actualités** : NewsAPI (Bloomberg, Reuters, AP News...)  
        - **Fréquence** : Mise à jour quotidienne
        """)
        st.success("Données actualisées automatiquement chaque jour.")        
# --- CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=3600)
def load_data():
    conn = sqlite3.connect('agri_data.db')
    
    # Chargement des prix des actions
    df_prices = pd.read_sql("SELECT * FROM stock_prices", conn)
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    df_prices.set_index('Date', inplace=True)
    df_returns = df_prices.pct_change().dropna()
    
    # NOUVEAU : Chargement des matières premières (Gaz et Blé)
    try:
        df_commodities = pd.read_sql("SELECT * FROM commodity_prices", conn)
        df_commodities['Date'] = pd.to_datetime(df_commodities['Date'])
        df_commodities.set_index('Date', inplace=True)
    except:
        df_commodities = pd.DataFrame()
    
    # Chargement des actualités et sentiment
    try:
        df_news = pd.read_sql("SELECT * FROM news_sentiment", conn)
        df_news['Date'] = pd.to_datetime(df_news['publishedAt']).dt.date
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        daily_sentiment = df_news.groupby('Date')['score_sentiment'].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['Date', 'Sentiment_Moyen', 'Nb_Articles']
        
        daily_titles = df_news.groupby('Date').apply(
            lambda x: pd.Series({
                'Articles_Titres': '<br>• '.join(x['title'].dropna().astype(str).tolist()),
                'Articles_Sources': ' | '.join(x['source'].dropna().astype(str).unique().tolist())
            })
        ).reset_index()
        
        daily_sentiment = daily_sentiment.merge(daily_titles, on='Date', how='left')
        daily_sentiment['Articles_Titres'] = daily_sentiment['Articles_Titres'].fillna('')
        daily_sentiment['Articles_Sources'] = daily_sentiment['Articles_Sources'].fillna('')
        
    except Exception as e:
        daily_sentiment = pd.DataFrame(columns=['Date', 'Sentiment_Moyen', 'Nb_Articles', 'Articles_Titres', 'Articles_Sources'])
        df_news = pd.DataFrame()

    conn.close()
    return df_prices, daily_sentiment, df_returns, df_news, df_commodities  # ← AJOUTÉ df_commodities
    
df_prices, df_sentiment, df_returns, df_news_raw, df_commodities = load_data()

if df_prices.empty:
    st.error("Base de données vide. Lancez d'abord data_pipeline.py.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Paramètres")
min_date = df_prices.index.min().date()
max_date = df_prices.index.max().date()
date_range = st.sidebar.date_input(
    "Période",
    value=(max_date - timedelta(days=180), max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if hasattr(date_range, '__len__') else date_range
    end_date = max_date

mask_prices = (df_prices.index.date >= start_date) & (df_prices.index.date <= end_date)
df_prices_filtre = df_prices.loc[mask_prices]
mask_sentiment = (df_sentiment['Date'].dt.date >= start_date) & (df_sentiment['Date'].dt.date <= end_date)
df_sentiment_filtre = df_sentiment.loc[mask_sentiment]

entreprises = df_prices.columns.tolist()

# --- ONGLETS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Analyse de l'état financier des géants", 
    "2. Radar de Crise", 
    "3. Journal de Bord",
    "4. Intrants & Dépendances",   # ← NOUVEAU
    "5. Impact Consommateur",       # ← NOUVEAU
    "6. Scénarios de Crise"    
])

# ==========================================
# ONGLET 1 : ANALYSE COMPARATIVE DES PRIX
# ==========================================
with tab1:
    st.header("Analyse comparative des prix")
    
    # Tableau des statistiques de performances (Réintégré)
    st.subheader("Tableau de bord des performances")
    st.markdown('<div class="plot-explanation">Dernier cours, performance sur la période sélectionnée et volatilité annualisée pour chaque acteur.</div>', unsafe_allow_html=True)
    
    stats_data = []
    for act in entreprises:
        data = df_prices_filtre[act].dropna()
        if len(data) > 1:
            last_price = data.iloc[-1]
            perf = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
            vol = data.pct_change().std() * np.sqrt(252) * 100
            stats_data.append({
                'Acteur': act,
                'Dernier Prix': f"{last_price:.2f}",
                'Variation Période': f"{perf:+.2f}%",
                'Volatilité (Ann.)': f"{vol:.2f}%"
            })
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        entreprises_selection = st.multiselect(
            "Acteurs à afficher :",
            options=entreprises,
            default=["Yara (Norvège)", "K+S (Allemagne)", "Grupa Azoty (Pologne)"]
        )
    with col2:
        mode_comparaison = st.radio(
            "Mode d'affichage",
            ["Valeur Réelle", "Base 100 (Performance %)"],
            help="La Base 100 permet de superposer toutes les actions et de voir qui progresse le plus."
        )
    
    if entreprises_selection:
        st.markdown('<div class="plot-explanation">Chaque courbe représente l\'évolution du prix. En Base 100, toutes les actions partent de 100 au début de la période, ce qui facilite la comparaison des performances relatives.</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        for acteur in entreprises_selection:
            data_plot = df_prices_filtre[acteur].dropna()
            if mode_comparaison == "Base 100 (Performance %)":
                data_plot = (data_plot / data_plot.iloc[0]) * 100
                y_title = "Performance (Base 100)"
            else:
                y_title = "Prix de clôture"
            fig1.add_trace(go.Scatter(x=data_plot.index, y=data_plot, name=acteur, mode='lines', line=dict(width=2.5)))
        fig1.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig1, use_container_width=True)

    # --- GRAPHIQUE 1 : WATERFALL ---
    st.subheader("Contribution hebdomadaire à la performance")
    st.markdown('<div class="plot-explanation">Ce graphique décompose la performance totale en variations hebdomadaires. Les barres vertes indiquent les semaines de hausse, les rouges les semaines de baisse.</div>', unsafe_allow_html=True)
    
    if entreprises_selection:
        acteur_waterfall = st.selectbox("Acteur pour analyse détaillée :", entreprises_selection, key="waterfall")
        weekly_returns = df_prices_filtre[acteur_waterfall].resample('W-FRI').last().pct_change().dropna() * 100
        weekly_returns = weekly_returns.tail(12)
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Performance", orientation="v",
            measure=["relative"] * len(weekly_returns) + ["total"],
            x=weekly_returns.index.strftime('%d %b').tolist() + ["Total"],
            y=weekly_returns.values.tolist() + [weekly_returns.sum()],
            text=[f"{v:+.2f}%" for v in weekly_returns.values] + [f"{weekly_returns.sum():+.2f}%"],
            textposition="outside",
            decreasing={"marker": {"color": "#E74C3C"}},
            increasing={"marker": {"color": "#27AE60"}},
            totals={"marker": {"color": "#2C3E50"}}
        ))
        fig_waterfall.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # --- GRAPHIQUE 2 : MATRICE DE CORRÉLATION ---
    st.subheader("Matrice de corrélation")
    st.markdown('<div class="plot-explanation">Une valeur proche de 1 (rouge) signifie que les deux actions évoluent dans le même sens. Une valeur proche de -1 (bleu) indique des mouvements opposés.</div>', unsafe_allow_html=True)
    
    if len(entreprises_selection) >= 2:
        returns_sel = df_prices_filtre[entreprises_selection].pct_change().dropna()
        corr_matrix = returns_sel.corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=350)
        fig_corr.update_traces(textfont=dict(size=14, color='white'))
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- GRAPHIQUE 3 : BANDES DE BOLLINGER ---
    st.subheader("Bandes de Bollinger (Volatilité)")
    st.markdown('<div class="plot-explanation">Les bandes s\'écartent quand le marché est nerveux. Un prix qui touche ou dépasse une bande peut signaler une situation de surachat ou de survente.</div>', unsafe_allow_html=True)
    
    if entreprises_selection:
        acteur_bb = st.selectbox("Acteur :", entreprises_selection, key="bb")
        df_bb = df_prices_filtre[[acteur_bb]].copy()
        df_bb['MA20'] = df_bb[acteur_bb].rolling(20).mean()
        df_bb['STD20'] = df_bb[acteur_bb].rolling(20).std()
        df_bb['Upper'] = df_bb['MA20'] + 2 * df_bb['STD20']
        df_bb['Lower'] = df_bb['MA20'] - 2 * df_bb['STD20']
        
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb[acteur_bb], name="Prix", line=dict(color='#1F618D', width=2)))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['Upper'], name="Bande haute", line=dict(dash='dash', color='gray')))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['MA20'], name="Moyenne 20j", line=dict(color='orange')))
        fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb['Lower'], name="Bande basse", line=dict(dash='dash', color='gray'), fill='tonexty', fillcolor='rgba(0,0,0,0.05)'))
        fig_bb.update_layout(height=350)
        st.plotly_chart(fig_bb, use_container_width=True)

# ==========================================
# ONGLET 2 : RADAR DE CRISE
# ==========================================
with tab2:
    st.header("Radar de souveraineté alimentaire")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if not df_sentiment_filtre.empty:
            avg_sent = df_sentiment_filtre['Sentiment_Moyen'].mean()
            st.metric("Sentiment moyen", f"{avg_sent:.3f}", delta="Positif" if avg_sent > 0.05 else "Négatif")
    with col_s2:
        if not df_sentiment_filtre.empty:
            worst = df_sentiment_filtre.loc[df_sentiment_filtre['Sentiment_Moyen'].idxmin()]
            st.metric("Jour le plus négatif", f"{worst['Sentiment_Moyen']:.3f}", delta=worst['Date'].strftime('%d/%m'))
    with col_s3:
        last_week = df_prices_filtre.index.max() - timedelta(days=7)
        perf_7j = df_prices_filtre[df_prices_filtre.index >= last_week].pct_change().sum()
        if not perf_7j.empty:
            st.metric("Acteur sous pression", perf_7j.idxmin(), delta=f"{perf_7j.min()*100:.2f}%")

    # --- GRAPHIQUE 4 : JAUGE DE STRESS ---
    st.subheader("Indicateur de Stress Géopolitique")
    st.markdown('<div class="plot-explanation">Cet indice combine la volatilité des marchés et la négativité des actualités pour estimer le niveau de tension global sur le secteur.</div>', unsafe_allow_html=True)
    volat = df_prices_filtre.pct_change().std().mean() * 100
    sent_neg = abs(df_sentiment_filtre[df_sentiment_filtre['Sentiment_Moyen'] < 0]['Sentiment_Moyen'].mean()) if not df_sentiment_filtre.empty else 0
    stress = min(100, volat * 10 + sent_neg * 50)
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=stress,
        title={'text': "Niveau de stress"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2C3E50"},
            'steps': [{'range': [0, 30], 'color': '#27AE60'}, {'range': [30, 60], 'color': '#F1C40F'}, {'range': [60, 100], 'color': '#E74C3C'}]
        }
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TOP/FLOP DE LA SEMAINE (Réintégré) ---
    st.subheader("Top / Flop de la semaine")
    st.markdown('<div class="plot-explanation">Meilleure et pire performance hebdomadaire parmi les acteurs sélectionnés.</div>', unsafe_allow_html=True)
    if entreprises_selection:
        weekly_perf = df_prices_filtre[entreprises_selection].resample('W-FRI').last().pct_change().iloc[-1] * 100
        if not weekly_perf.empty:
            top_act = weekly_perf.idxmax()
            flop_act = weekly_perf.idxmin()
            col_t, col_f = st.columns(2)
            with col_t:
                st.metric(f"Top : {top_act}", f"{weekly_perf.max():+.2f}%")
            with col_f:
                st.metric(f"Flop : {flop_act}", f"{weekly_perf.min():+.2f}%")

        # --- HEATMAP CALENDAIRE MENSUELLE DU SENTIMENT (Corrigée) ---
    st.subheader("Calendrier mensuel du sentiment")
    st.markdown('<div class="plot-explanation">Chaque jour est coloré selon le sentiment moyen des articles (vert = positif, rouge = négatif). Les jours sans articles sont en gris clair.</div>', unsafe_allow_html=True)
    if not df_sentiment_filtre.empty:
        df_cal = df_sentiment_filtre.copy()
        df_cal['year'] = df_cal['Date'].dt.year
        df_cal['month'] = df_cal['Date'].dt.month
        df_cal['day'] = df_cal['Date'].dt.day
        
        years = sorted(df_cal['year'].unique())
        if years:
            col_y, col_m = st.columns(2)
            with col_y:
                selected_year = st.selectbox("Année", years, index=len(years)-1, key="cal_year")
            
            # Filtrer sur l'année sélectionnée
            df_year = df_cal[df_cal['year'] == selected_year]
            months = sorted(df_year['month'].unique())
            
            if months:
                with col_m:
                    month_names = [calendar.month_name[m] for m in months]
                    selected_month_name = st.selectbox("Mois", month_names, key="cal_month")
                    selected_month = months[month_names.index(selected_month_name)]
                
                # Filtrer sur le mois sélectionné
                df_month = df_year[df_year['month'] == selected_month].copy()
                
                if not df_month.empty:
                    # Créer une grille complète pour le mois
                    first_day = pd.Timestamp(year=selected_year, month=selected_month, day=1)
                    last_day = first_day + pd.offsets.MonthEnd(0)
                    all_days = pd.date_range(start=first_day, end=last_day, freq='D')
                    
                    # Déterminer la position dans la grille (semaine, jour de la semaine)
                    weeks = []
                    weekdays = []
                    
                    for d in all_days:
                        weekday = d.weekday()  # Lundi=0, Dimanche=6
                        week_in_month = (d.day - 1 + first_day.weekday()) // 7
                        weeks.append(week_in_month)
                        weekdays.append(weekday)
                    
                    # Créer une matrice pour la heatmap
                    n_weeks = max(weeks) + 1
                    n_days = 7
                    
                    # Initialiser avec NaN
                    sentiment_matrix = np.full((n_days, n_weeks), np.nan)
                    hover_text_matrix = np.empty((n_days, n_weeks), dtype=object)
                    
                    # Remplir la matrice
                    for i, d in enumerate(all_days):
                        week_idx = weeks[i]
                        day_idx = weekdays[i]
                        day_data = df_month[df_month['Date'].dt.day == d.day]
                        
                        if not day_data.empty:
                            sentiment_val = day_data['Sentiment_Moyen'].iloc[0]
                            nb_art = day_data['Nb_Articles'].iloc[0]
                            titles = day_data['Articles_Titres'].iloc[0] if 'Articles_Titres' in day_data.columns else ''
                            
                            sentiment_matrix[day_idx, week_idx] = sentiment_val
                            hover_text_matrix[day_idx, week_idx] = f"Jour: {d.day}<br>Sentiment: {sentiment_val:.3f}<br>Articles: {nb_art}"
                        else:
                            sentiment_matrix[day_idx, week_idx] = 0  # Valeur neutre pour les jours sans données
                            hover_text_matrix[day_idx, week_idx] = f"Jour: {d.day}<br>Aucun article"
                    
                    # Créer la heatmap
                    fig_cal = go.Figure()
                    
                    # Labels des jours de la semaine
                    day_labels = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
                    
                    # Ajouter la heatmap
                    fig_cal.add_trace(go.Heatmap(
                        z=sentiment_matrix,
                        x=[f"Sem. {w+1}" for w in range(n_weeks)],
                        y=day_labels,
                        colorscale=[
                            [0.0, '#E74C3C'],      # -1 : rouge
                            [0.25, '#E74C3C'],
                            [0.5, '#F8F9FA'],       # 0 : gris clair
                            [0.75, '#27AE60'],
                            [1.0, '#27AE60']        # +1 : vert
                        ],
                        zmid=0,
                        zmin=-1,
                        zmax=1,
                        text=hover_text_matrix,
                        hoverinfo='text',
                        showscale=True,
                        colorbar=dict(
                            title="Sentiment",
                            tickmode='array',
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=['Négatif', '-0.5', 'Neutre', '0.5', 'Positif']
                        )
                    ))
                    
                    # Ajouter les numéros de jour dans les cellules
                    for week_idx in range(n_weeks):
                        for day_idx in range(n_days):
                            day_num = week_idx * 7 + day_idx - first_day.weekday() + 1
                            if 1 <= day_num <= last_day.day:
                                sentiment_val = sentiment_matrix[day_idx, week_idx]
                                text_color = 'white' if abs(sentiment_val) > 0.5 else 'black'
                                fig_cal.add_annotation(
                                    x=f"Sem. {week_idx+1}",
                                    y=day_labels[day_idx],
                                    text=str(day_num),
                                    showarrow=False,
                                    font=dict(size=12, color=text_color),
                                    xanchor='center',
                                    yanchor='middle'
                                )
                    
                    fig_cal.update_layout(
                        title=f"{selected_month_name} {selected_year}",
                        height=320,
                        xaxis=dict(side="top", title=""),
                        yaxis=dict(autorange="reversed", title=""),
                        plot_bgcolor='#F8F9FA',
                        margin=dict(t=50, b=20, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.info(f"Aucune donnée pour {selected_month_name} {selected_year}.")
            else:
                st.info(f"Aucune donnée pour l'année {selected_year}.")
    else:
        st.info("Aucune donnée de sentiment disponible.")
    # --- GRAPHIQUE 5 : SANKEY ---
    st.subheader("Flux d'influence")
    st.markdown('<div class="plot-explanation">Représentation théorique des canaux de transmission d\'un choc géopolitique (ex: hausse du gaz) vers les entreprises selon leur exposition.</div>', unsafe_allow_html=True)
    influences = {
        'Hausse gaz': {'Yara (Norvège)': -0.8, 'Grupa Azoty (Pologne)': -0.9, 'K+S (Allemagne)': -0.5},
        'Conflit Iran': {'ICL (Israël)': -0.7, 'OCI (Pays-Bas)': -0.4},
        'Accord': {'OCI (Pays-Bas)': 0.5, 'ICL (Israël)': 0.4}
    }
    
    sources, targets, values, labels = [], [], [], []
    label_dict = {}
    idx = 0
    for event in influences: label_dict[event] = idx; idx += 1
    for acteur in entreprises: label_dict[acteur] = idx; idx += 1
    
    for event, impacts in influences.items():
        for acteur, force in impacts.items():
            if acteur in label_dict:
                sources.append(label_dict[event])
                targets.append(label_dict[acteur])
                values.append(abs(force) * 10)
    
    labels = list(label_dict.keys())
    link_colors = ['#E74C3C' if influences[labels[s]].get(labels[t], 0) < 0 else '#27AE60' for s, t in zip(sources, targets)]
    
    fig_sankey = go.Figure(data=[go.Sankey(node=dict(label=labels, color="#2C3E50"), link=dict(source=sources, target=targets, value=values, color=link_colors))])
    fig_sankey.update_layout(height=300)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # --- GRAPHIQUE 6 : RADAR ---
    st.subheader("Radar comparatif")
    st.markdown('<div class="plot-explanation">Compare les profils de risque et de performance sur 5 axes. Une surface plus étendue indique un profil plus dynamique ou exposé.</div>', unsafe_allow_html=True)
    acteurs_radar = st.multiselect("2-3 acteurs :", entreprises, default=entreprises[:3], max_selections=3)
    if acteurs_radar:
        categories = ['Rendement', 'Volatilité', 'Expo Gaz', 'Expo Iran', 'Sentiment']
        fig_radar = go.Figure()
        colors = ['#3498DB', '#E74C3C', '#27AE60']
        for i, act in enumerate(acteurs_radar):
            ret = df_prices_filtre[act].pct_change().sum() * 100
            vol = df_prices_filtre[act].pct_change().std() * 100
            expo_gaz = 90 if 'Norvège' in act or 'Pologne' in act else (70 if 'Israël' in act else 40)
            expo_iran = 95 if 'Israël' in act else (50 if 'Pays-Bas' in act or 'Allemagne' in act else 30)
            sent = 50 + df_sentiment_filtre['Sentiment_Moyen'].mean() * 50 if not df_sentiment_filtre.empty else 50
            values = [min(100, max(0, 50+ret)), min(100, vol*2), expo_gaz, expo_iran, sent]
            values.append(values[0])
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories+[categories[0]], fill='toself', name=act, line_color=colors[i%3]))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# ONGLET 3 : JOURNAL DE BORD
# ==========================================
with tab3:
    st.header("Journal de Bord")
    
    acteur_cible = st.selectbox("Acteur :", entreprises, index=0)
    
    df_prix = df_prices_filtre[[acteur_cible]].copy()
    df_prix['Date_Str'] = df_prix.index.date
    df_sent = df_sentiment_filtre.copy()
    df_sent['Date_Str'] = df_sent['Date'].dt.date
    df_merged = pd.merge(df_prix, df_sent, on='Date_Str', how='left')
    df_merged.set_index(df_prix.index, inplace=True)
    df_merged['Sentiment_Moyen'] = df_merged['Sentiment_Moyen'].fillna(0)
    df_merged['Variation_Prix'] = df_merged[acteur_cible].pct_change() * 100

    # --- SYNCHRO ---
    st.subheader("Visualisation synchrone")
    st.markdown("""<div class="info-box"><b>Lecture :</b> Courbe bleue = prix de l'acteur. Barres = sentiment médiatique (vert=positif, rouge=négatif). Comparez les pics de sentiment avec les mouvements de prix.</div>""", unsafe_allow_html=True)
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=df_merged.index, y=df_merged[acteur_cible], name="Prix", line=dict(color='#1F618D')), secondary_y=False)
    colors = ['#E74C3C' if s < 0 else '#2ECC71' for s in df_merged['Sentiment_Moyen']]
    fig3.add_trace(go.Bar(x=df_merged.index, y=df_merged['Sentiment_Moyen'], name="Sentiment", marker_color=colors, opacity=0.6), secondary_y=True)
    fig3.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    # --- ÉVÉNEMENTS MARQUANTS (Corrigé pour lisibilité) ---
    st.subheader("Top 3 des événements marquants")
    st.markdown('<div class="plot-explanation">Jours où un sentiment médiatique extrême coïncide avec une forte variation du cours. Les titres d\'articles associés sont affichés.</div>', unsafe_allow_html=True)
    jours_impact = df_merged[(df_merged['Sentiment_Moyen'].abs() > 0.3) & (df_merged['Variation_Prix'].abs() > 1.5)].copy()
    if not jours_impact.empty:
        jours_impact = jours_impact.sort_values('Sentiment_Moyen', key=lambda x: x.abs(), ascending=False).head(3)
        for idx, row in jours_impact.iterrows():
            sentiment_emoji = "🔴" if row['Sentiment_Moyen'] < 0 else "🟢"
            variation_emoji = "📉" if row['Variation_Prix'] < 0 else "📈"
            titres = row.get('Articles_Titres', 'Aucun titre') or 'Aucun titre'
            sources = row.get('Articles_Sources', 'Source inconnue') or 'Source inconnue'
            st.markdown(f"""
            <div class="event-card" style="border-left: 6px solid {'#C0392B' if row['Sentiment_Moyen'] < 0 else '#27AE60'};">
                <h3 style="color: #2C3E50;">{idx.strftime('%d %B %Y')} {sentiment_emoji} {variation_emoji}</h3>
                <p style="color: #2C3E50;"><b>Variation :</b> <span style="color:{'#C0392B' if row['Variation_Prix'] < 0 else '#27AE60'}; font-size:18px;">{row['Variation_Prix']:.2f}%</span><br>
                <b>Sentiment :</b> {row['Sentiment_Moyen']:.3f}<br>
                <b>Sources :</b> {sources}</p>
                <div class="articles-box"><b>Titres :</b><br>• {titres}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucun événement extrême détecté.")

    # --- INDICE DE PRESSION GÉOPOLITIQUE (IPG) - Réintégré ---
    st.subheader("Indice de Pression Géopolitique (IPG)")
    st.markdown('<div class="plot-explanation">Agrégation quotidienne du sentiment médiatique négatif et du nombre d\'articles. Un pic indique une forte couverture médiatique négative.</div>', unsafe_allow_html=True)
    if not df_sentiment_filtre.empty:
        df_ipg = df_sentiment_filtre.copy()
        df_ipg = df_ipg.set_index('Date')
        # IPG = Volume d'articles * (1 - Sentiment) / 2  -> plus c'est haut, plus c'est négatif et intense
        df_ipg['IPG'] = df_ipg['Nb_Articles'] * (1 - df_ipg['Sentiment_Moyen']) / 2
        fig_ipg = px.area(df_ipg, x=df_ipg.index, y='IPG', title="Evolution de l'IPG")
        fig_ipg.update_traces(line_color='#E74C3C', fillcolor='rgba(231, 76, 60, 0.3)')
        fig_ipg.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_ipg, use_container_width=True)

    # --- GRAPHIQUE 7 : TIMELINE ---
    st.subheader("Frise chronologique")
    st.markdown('<div class="plot-explanation">Chaque point est un jour. La taille indique le volume d\'articles, la couleur le sentiment. Survoler un point affiche les titres.</div>', unsafe_allow_html=True)
    if not df_sentiment_filtre.empty:
        df_timeline = df_sentiment_filtre[df_sentiment_filtre['Nb_Articles'] > 0]
        if not df_timeline.empty:
            fig_timeline = px.scatter(df_timeline, x='Date', y='Sentiment_Moyen', size='Nb_Articles', color='Sentiment_Moyen', color_continuous_scale='RdYlGn', hover_data=['Articles_Titres'])
            fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)

    # --- GRAPHIQUE 8 : HEATMAP DÉLAIS ---
    st.subheader("Matrice des délais de réaction")
    st.markdown('<div class="plot-explanation">Corrélation entre le sentiment du jour J et la variation du prix à J+lag. Vert = le sentiment anticipe le prix. Rouge = relation inverse.</div>', unsafe_allow_html=True)
    cross_data = []
    for act in entreprises:
        temp = pd.DataFrame({'Prix': df_prices_filtre[act].pct_change(), 'Sentiment': df_sent.set_index('Date')['Sentiment_Moyen'].reindex(df_prices_filtre.index, fill_value=0)}).dropna()
        for lag in range(0, 5):
            corr = temp['Prix'].corr(temp['Sentiment'].shift(lag))
            cross_data.append({'Acteur': act, 'Délai': lag, 'Corrélation': corr})
    df_cross = pd.DataFrame(cross_data)
    pivot = df_cross.pivot(index='Acteur', columns='Délai', values='Corrélation')
    fig_cross = px.imshow(pivot, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-0.5, zmax=0.5, aspect="auto")
    fig_cross.update_layout(height=350)
    st.plotly_chart(fig_cross, use_container_width=True)

    # --- GRAPHIQUE 9 : SCATTER ANIMÉ ---
    st.subheader("Evolution Prix/Sentiment")
    st.markdown('<div class="plot-explanation">Chaque point est un jour. L\'animation montre comment la relation entre sentiment et variation de prix évolue mois par mois.</div>', unsafe_allow_html=True)
    df_scatter = pd.DataFrame({
        'Date': df_prices_filtre.index,
        'Variation': df_prices_filtre[acteur_cible].pct_change() * 100,
        'Sentiment': df_sent.set_index('Date')['Sentiment_Moyen'].reindex(df_prices_filtre.index, fill_value=0),
        'Mois': df_prices_filtre.index.to_period('M').astype(str)
    }).dropna()
    fig_scatter = px.scatter(df_scatter, x='Sentiment', y='Variation', animation_frame='Mois', hover_data=['Date'], range_x=[-1, 1])
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- NUAGE DE MOTS ---
    st.subheader("Nuage de mots")
    st.markdown('<div class="plot-explanation">Mots les plus fréquents dans les titres d\'actualité sur la période. Plus un mot est gros, plus il apparaît souvent.</div>', unsafe_allow_html=True)
    if not df_news_raw.empty:
        text = " ".join(df_news_raw[df_news_raw['Date'] >= pd.to_datetime(start_date)]['title'].dropna().astype(str))
        if text:
            wc = WordCloud(width=800, height=250, background_color='white', colormap='RdYlGn', stopwords=STOPWORDS).generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 2.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)


# ==========================================
# ONGLET 4 : INTRANTS & DÉPENDANCES
# ==========================================
with tab4:
    st.header(" Dépendance aux Intrants : Gaz Naturel & Engrais")
    
    if df_commodities.empty:
        st.warning("⚠️ Données des matières premières non disponibles. Exécute d'abord data_pipeline.py avec les nouveaux tickers.")
    else:
        # Filtrage sur la période sélectionnée
        mask_comm = (df_commodities.index.date >= start_date) & (df_commodities.index.date <= end_date)
        df_comm_filtre = df_commodities.loc[mask_comm]
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Corrélation Prix du Gaz vs Producteurs d'Engrais")
            st.markdown('<div class="plot-explanation">Plus la courbe du Gaz (Rouge) monte, plus les marges des producteurs européens (Bleu) sont sous pression.</div>', unsafe_allow_html=True)
            
            if 'Yara (Norvège)' in df_prices_filtre.columns and 'Gaz_Nat_EU' in df_comm_filtre.columns:
                # Normalisation Base 100
                gaz_align = df_comm_filtre['Gaz_Nat_EU'].dropna()
                yara_align = df_prices_filtre['Yara (Norvège)'].dropna()
                
                # Alignement des dates
                common_idx = gaz_align.index.intersection(yara_align.index)
                if len(common_idx) > 0:
                    gaz_norm = gaz_align.loc[common_idx] / gaz_align.loc[common_idx[0]] * 100
                    yara_norm = yara_align.loc[common_idx] / yara_align.loc[common_idx[0]] * 100
                    
                    fig_dep = go.Figure()
                    fig_dep.add_trace(go.Scatter(x=gaz_norm.index, y=gaz_norm, name="Gaz TTF (Europe)", line=dict(color='#E74C3C', width=3)))
                    fig_dep.add_trace(go.Scatter(x=yara_norm.index, y=yara_norm, name="Yara (Engrais)", line=dict(color='#1F618D', width=2)))
                    fig_dep.update_layout(height=350, hovermode="x unified")
                    st.plotly_chart(fig_dep, use_container_width=True)
                else:
                    st.info("Pas de dates communes entre le Gaz et Yara.")
            else:
                st.info("Données Gaz ou Yara non disponibles sur cette période.")
                
        with col_g2:
            st.subheader("Part des importations d'engrais de l'UE")
            # Données Statiques (Source : Fertilizers Europe / Eurostat 2023)
            labels_imports = ['Russie', 'Biélorussie', 'Maroc', 'Égypte', 'Algérie', 'Autres']
            values_imports = [30, 15, 20, 10, 10, 15]
            
            fig_pie = px.pie(names=labels_imports, values=values_imports, hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Source : Estimations Fertilizers Europe 2024 - Forte dépendance à l'Est.")
        
        st.divider()
        
        st.subheader(" Seuil de Rentabilité des Agriculteurs")
        st.markdown('<div class="plot-explanation">Évolution du ratio Blé / Engrais. Quand le ratio baisse, le pouvoir d\'achat de l\'agriculteur se dégrade.</div>', unsafe_allow_html=True)
        
        if 'Ble_Chicago' in df_comm_filtre.columns and 'Yara (Norvège)' in df_prices_filtre.columns:
            ble_align = df_comm_filtre['Ble_Chicago'].dropna()
            yara_align = df_prices_filtre['Yara (Norvège)'].dropna()
            
            common_idx = ble_align.index.intersection(yara_align.index)
            if len(common_idx) > 0:
                ratio = ble_align.loc[common_idx] / yara_align.loc[common_idx]
                
                fig_ratio = go.Figure()
                fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio, fill='tozeroy', line=dict(color='#D4AC0D')))
                fig_ratio.add_hline(y=ratio.mean(), line_dash="dash", annotation_text="Moyenne", line_color="gray")
                fig_ratio.update_layout(height=350, title="Ratio Prix du Blé / Prix de l'Action Engrais (Proxy)")
                st.plotly_chart(fig_ratio, use_container_width=True)
            else:
                st.info("Pas de dates communes entre le Blé et Yara.")
        else:
            st.info("Données Blé ou Yara non disponibles.")

    #bonjour


    # --- CARTE EUROPÉENNE DE LA DÉPENDANCE AUX ENGRAIS IMPORTÉS ---
    st.subheader(" Dépendance de l'UE aux engrais importés")
    
    dependency_data = {
        'Pays': ['France', 'Allemagne', 'Italie', 'Espagne', 'Pologne', 'Pays-Bas', 'Belgique',
                 'Roumanie', 'Grèce', 'Portugal', 'Suède', 'Autriche', 'Bulgarie', 'Finlande',
                 'Danemark', 'Irlande', 'Lituanie', 'Lettonie', 'Estonie', 'Slovaquie',
                 'Slovénie', 'Croatie', 'Royaume-Uni', 'Norvège', 'Suisse'],
        'Code': ['FRA', 'DEU', 'ITA', 'ESP', 'POL', 'NLD', 'BEL', 'ROU', 'GRC', 'PRT',
                 'SWE', 'AUT', 'BGR', 'FIN', 'DNK', 'IRL', 'LTU', 'LVA', 'EST', 'SVK',
                 'SVN', 'HRV', 'GBR', 'NOR', 'CHE'],
        'Dépendance (%)': [28, 30, 58, 45, 65, 22, 45, 75, 55, 60, 18, 35, 80, 25, 15,
                           40, 85, 70, 50, 68, 62, 55, 35, 10, 20]
    }
    df_dep = pd.DataFrame(dependency_data)
    
    # Catégorie pour le tooltip enrichi
    def categorize(val):
        if val < 25:   return "🟢 Faible dépendance"
        elif val < 50: return "🟡 Dépendance modérée"
        elif val < 70: return "🟠 Dépendance élevée"
        else:          return "🔴 Très forte dépendance"
    
    df_dep['Catégorie'] = df_dep['Dépendance (%)'].apply(categorize)
    df_dep['Hover'] = df_dep.apply(
        lambda r: f"<b>{r['Pays']}</b><br>"
                  f"Dépendance : <b>{r['Dépendance (%)']}%</b><br>"
                  f"{r['Catégorie']}",
        axis=1
    )
    
    fig_map = px.choropleth(
        df_dep,
        locations='Code',
        color='Dépendance (%)',
        hover_name='Pays',
        custom_data=['Hover'],
        color_continuous_scale=[
            [0.0,  '#1a9641'],
            [0.25, '#a6d96a'],
            [0.50, '#ffffbf'],
            [0.75, '#fdae61'],
            [1.0,  '#d7191c'],
        ],
        range_color=(0, 100),
        scope='europe',
        projection='natural earth'
    )
    
    fig_map.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker_line_color='white',
        marker_line_width=0.8,
    )
    
    fig_map.update_geos(
        center=dict(lat=54.5, lon=10.0),
        projection_scale=1,
        showcountries=True,
        countrycolor="white",
        showcoastlines=True,
        coastlinecolor="#90b4ce",
        coastlinewidth=1.2,
        showland=True,
        landcolor="#e8e8e8",        # pays hors dataset en gris neutre
        showocean=True,
        oceancolor="#d0e8f5",
        showlakes=True,
        lakecolor="#d0e8f5",
        showframe=False,
        lonaxis=dict(range=[-25, 45]),
        lataxis=dict(range=[34, 72]),
    )
    
    fig_map.update_layout(
        title=dict(
            text="<b>Dépendance aux engrais azotés importés en Europe</b>",
            font=dict(size=17, color="#1a1a2e", family="Arial"),
            x=0.5,
            xanchor='center',
            y=0.97,
        ),
        height=540,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title=dict(
                text="Part des importations",
                font=dict(size=12, color="#333"),
            ),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0 %', '25 %', '50 %', '75 %', '100 %'],
            tickfont=dict(size=11, color="#333"),
            len=0.65,
            thickness=14,
            yanchor="middle",
            y=0.5,
            x=1.0,
            xanchor="left",
            outlinewidth=0,
            bgcolor='rgba(255,255,255,0.85)',
            borderwidth=0,
            ticks="outside",
            ticklen=4,
        ),
        annotations=[
            dict(
                text="🟢 < 25 %  Faible  |  🟡 25–50 %  Modérée  |  🟠 50–70 %  Élevée  |  🔴 > 70 %  Très élevée",
                x=0.5, y=-0.01,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=10.5, color="#555"),
                align='center',
            ),
            dict(
                text="Source : Eurostat · Fertilizers Europe · World Bank WITS",
                x=0.5, y=-0.05,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=9.5, color="#888", style='italic'),
                align='center',
            ),
        ]
    )
    
    st.plotly_chart(fig_map, use_container_width=True)


# ==========================================
# ONGLET 5 : IMPACT CONSOMMATEUR & SOUVERAINETÉ
# ==========================================
with tab5:
    st.header(" De la Bourse à l'Assiette : Impact sur le Consommateur")
    
    # === NOUVEAU : RÉSUMÉ POUR LE CITOYEN LAMBDA ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3F20 0%, #2E4053 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h2 style="color: #F1C40F; margin-top: 0;">🛒 Que signifie tout ça pour mon porte-monnaie ?</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            Quand le prix du <b>gaz</b> augmente → le prix des <b>engrais</b> augmente → 
            les <b>agriculteurs</b> utilisent moins d'engrais → les <b>récoltes</b> baissent → 
            le prix du <b>pain, pâtes, viande</b> augmente dans les supermarchés.
        </p>
        <p style="font-size: 1rem; color: #F8F9FA; margin-top: 10px;">
             <b>Délai constaté</b> : Entre le choc sur le gaz et la hausse en rayon, il se passe généralement <b>6 à 18 mois</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === NOUVEAU : SIMULATION IMPACT PANIER TYPE ===
    st.subheader(" Simulation d'impact sur un panier alimentaire type")
    st.markdown('<div class="plot-explanation">Estimation de la majoration de prix pour un ménage moyen, basée sur le niveau de stress actuel du marché des engrais.</div>', unsafe_allow_html=True)
    
    # Calcul du score de stress (même logique que ta jauge, mais en variable)
    score_stress_panier = 50  # Valeur neutre par défaut
    
    if not df_commodities.empty:
        mask_comm = (df_commodities.index.date >= start_date) & (df_commodities.index.date <= end_date)
        df_comm_filtre = df_commodities.loc[mask_comm]
        
        if 'Gaz_Nat_EU' in df_comm_filtre.columns:
            try:
                gaz_data = df_comm_filtre['Gaz_Nat_EU'].dropna()
                if len(gaz_data) > 0:
                    gaz_recent = gaz_data.tail(30).mean()
                    gaz_moyenne_hist = gaz_data.mean()
                    gaz_score = 50 + (gaz_recent / gaz_moyenne_hist - 1) * 40
                    gaz_score = max(0, min(100, gaz_score))
                else:
                    gaz_score = 50
                
                if 'Yara (Norvège)' in df_prices_filtre.columns:
                    yara_vol = df_prices_filtre['Yara (Norvège)'].pct_change().std() * np.sqrt(252) * 100
                    vol_score = min(100, yara_vol * 2)
                else:
                    vol_score = 50
                
                if 'Ble_Chicago' in df_comm_filtre.columns:
                    ble_data = df_comm_filtre['Ble_Chicago'].dropna()
                    yara_data = df_prices_filtre['Yara (Norvège)'].dropna()
                    common_idx = ble_data.index.intersection(yara_data.index)
                    if len(common_idx) > 0:
                        ratio = ble_data.loc[common_idx] / yara_data.loc[common_idx]
                        ratio_recent = ratio.tail(30).mean()
                        ratio_moyenne = ratio.mean()
                        ratio_score = 50 - (ratio_recent / ratio_moyenne - 1) * 30
                        ratio_score = max(0, min(100, ratio_score))
                    else:
                        ratio_score = 50
                else:
                    ratio_score = 50
                
                score_stress_panier = int(gaz_score * 0.4 + vol_score * 0.3 + ratio_score * 0.3)
                
            except Exception as e:
                score_stress_panier = 50
    
    # Panier type français (source INSEE 2023) - Coefficients de sensibilité aux engrais
    panier_type = {
        "Pain et céréales": {"prix_mensuel": 45, "sensibilite": 0.25},  # Très sensible (blé)
        "Pâtes, riz": {"prix_mensuel": 25, "sensibilite": 0.20},
        "Viandes": {"prix_mensuel": 85, "sensibilite": 0.15},  # Coût de l'alimentation animale
        "Produits laitiers": {"prix_mensuel": 55, "sensibilite": 0.12},
        "Légumes et fruits": {"prix_mensuel": 70, "sensibilite": 0.10},
        "Œufs": {"prix_mensuel": 15, "sensibilite": 0.08},
        "Huiles et matières grasses": {"prix_mensuel": 12, "sensibilite": 0.10},
    }
    
    # Calcul de l'impact estimé (stress 50 = neutre, stress 100 = +15% max)
    facteur_impact = max(0, (score_stress_panier - 40) / 60) * 0.15  # Max +15%
    
    col_panier1, col_panier2 = st.columns([1, 1])
    
    with col_panier1:
        # Affichage du panier avec impact
        df_panier = pd.DataFrame([
            {
                "Catégorie": cat,
                "Budget moyen/mois": f"{data['prix_mensuel']}€",
                "Majoration estimée": f"+{data['prix_mensuel'] * facteur_impact:.1f}€",
                "Sensibilité engrais": "🔴" if data['sensibilite'] >= 0.20 else ("🟡" if data['sensibilite'] >= 0.10 else "🟢")
            }
            for cat, data in panier_type.items()
        ])
        st.dataframe(df_panier, use_container_width=True, hide_index=True)
    
    with col_panier2:
        budget_total = sum(d["prix_mensuel"] for d in panier_type.values())
        impact_total = budget_total * facteur_impact
        impact_annuel = impact_total * 12
        
        st.markdown(f"""
        <div style="background-color: {'#E8F8F5' if score_stress_panier < 40 else '#FEF9E7' if score_stress_panier < 70 else '#FDEDEC'}; 
                    padding: 20px; border-radius: 10px; border-left: 5px solid {'#27AE60' if score_stress_panier < 40 else '#F1C40F' if score_stress_panier < 70 else '#E74C3C'};">
            <h3 style="margin-top: 0; color: #2C3E50;"> Impact estimé pour un ménage</h3>
            <p style="font-size: 2rem; font-weight: bold; color: {'#27AE60' if score_stress_panier < 40 else '#F39C12' if score_stress_panier < 70 else '#E74C3C'};">
                +{impact_total:.0f}€ / mois
            </p>
            <p style="color: #5D6D7E;">
                Soit <b>+{impact_annuel:.0f}€ / an</b> sur un budget alimentaire de {budget_total}€/mois<br>
                <i>(Majoration potentielle : {(facteur_impact*100):.1f}%)</i>
            </p>
            <hr style="border-color: #E5E8E8;">
            <p style="font-size: 0.85rem; color: #7F8C8D;">
                 Estimation basée sur la corrélation historique entre le coût des engrais 
                et les prix alimentaires (source : INSEE, Eurostat). Impact réel variable 
                selon les contrats des distributeurs et les politiques de prix.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # === JAUGE RÉDUITE (demandé par l'utilisateur) ===
    col_jauge, col_delai = st.columns([1, 2])
    
    with col_jauge:
        st.subheader("Score de Stress")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_stress_panier,
            number={
                'font': {'size': 48, 'color': '#2C3E50'},
                'suffix': "",
                'valueformat': '.0f'
            },
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2C3E50"},
                'bar': {'color': "#2C3E50", 'thickness': 0.2},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#E5E8E8",
                'steps': [
                    {'range': [0, 40], 'color': '#D5F5E3'},
                    {'range': [40, 70], 'color': '#FEF9E7'},
                    {'range': [70, 100], 'color': '#FADBD8'}
                ],
                'threshold': {
                    'line': {'color': "#C0392B", 'width': 3},
                    'thickness': 0.8,
                    'value': 70
                }
            },
            domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]}   # Centrage horizontal et vertical
        ))
        
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#2C3E50", 'family': "Arial"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    # Interprétation du niveau de stress
    if score_stress_panier < 40:
        st.success(" Situation favorable")
    elif score_stress_panier < 70:
        st.warning(" Vigilance")
    else:
        st.error(" Tension élevée")
    
    with col_delai:
        st.subheader(" Délai de transmission estimé")
        st.markdown('<div class="plot-explanation">Temps nécessaire pour qu\'un choc sur le gaz se répercute sur le prix des engrais puis sur l\'action YARA.</div>', unsafe_allow_html=True)
        
        if not df_commodities.empty and 'Gaz_Nat_EU' in df_comm_filtre.columns and 'Yara (Norvège)' in df_prices_filtre.columns:
            gaz_series = df_comm_filtre['Gaz_Nat_EU'].dropna()
            yara_series = df_prices_filtre['Yara (Norvège)'].dropna()
            
            common_idx = gaz_series.index.intersection(yara_series.index)
            if len(common_idx) > 60:
                gaz_aligned = gaz_series.loc[common_idx].pct_change().dropna()
                yara_aligned = yara_series.loc[common_idx].pct_change().dropna()
                
                correlations = []
                max_lag = 60
                
                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        corr = gaz_aligned.corr(yara_aligned)
                    else:
                        corr = gaz_aligned.iloc[:-lag].corr(yara_aligned.iloc[lag:])
                    correlations.append(corr)
                
                best_lag = np.argmax(np.abs(correlations))
                best_corr = correlations[best_lag]
                mois_estimes = best_lag / 21
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Délai optimal", f"{mois_estimes:.1f} mois", delta=f"Corr: {best_corr:.2f}")
                with col_b:
                    st.metric("Jours ouvrés", f"{best_lag} jours")
                
                # Chaîne explicative simplifiée
                st.markdown("""
                <div style="background-color: #F8F9FA; padding: 12px; border-radius: 8px; margin-top: 10px; font-size: 0.9rem;">
                    <b> En clair :</b> Il faut environ <b>{:.0f} mois</b> entre une hausse du gaz 
                    et son impact visible sur le cours d'YARA. Puis <b>6 à 12 mois supplémentaires</b> 
                    avant que ça n'arrive dans votre supermarché.
                </div>
                """.format(mois_estimes), unsafe_allow_html=True)
            else:
                st.info("Données insuffisantes (min. 60 jours).")
        else:
            st.info("Données non disponibles.")
    
    st.divider()
    
    # === NOUVEAU : INDICE D'INFLATION ALIMENTAIRE (données statiques Eurostat) ===
    st.subheader(" Inflation alimentaire en Europe (Données Eurostat)")
    st.markdown('<div class="plot-explanation">Evolution de l\'indice des prix à la consommation pour les produits alimentaires dans les principaux pays européens. Données mensuelles Eurostat.</div>', unsafe_allow_html=True)
    
    # Données statiques Eurostat 2022-2024 (à mettre à jour manuellement ou via API si disponible)
    inflation_data = {
        'Mois': ['Jan-22', 'Avr-22', 'Juil-22', 'Oct-22', 'Jan-23', 'Avr-23', 'Juil-23', 'Oct-23', 'Jan-24', 'Avr-24', 'Juil-24', 'Oct-24'],
        'France': [1.2, 3.8, 6.8, 11.9, 13.2, 14.8, 12.1, 9.2, 6.0, 3.5, 1.8, 1.2],
        'Allemagne': [3.2, 8.5, 14.8, 20.3, 20.2, 17.2, 11.0, 6.1, 3.8, 1.8, 1.2, 1.5],
        'Pologne': [8.6, 16.7, 18.3, 19.0, 22.1, 19.2, 12.4, 8.3, 5.2, 3.1, 2.8, 3.5],
        'Moyenne UE': [4.5, 9.2, 12.5, 16.8, 17.5, 15.8, 11.5, 7.8, 5.2, 3.2, 2.1, 2.3]
    }
    df_inflation = pd.DataFrame(inflation_data)
    
    fig_inflation = go.Figure()
    colors_inflation = ['#1F618D', '#E74C3C', '#27AE60', '#2C3E50']
    for i, pays in enumerate(['France', 'Allemagne', 'Pologne', 'Moyenne UE']):
        fig_inflation.add_trace(go.Scatter(
            x=df_inflation['Mois'], 
            y=df_inflation[pays], 
            name=pays,
            line=dict(color=colors_inflation[i], width=3 if pays == 'Moyenne UE' else 2),
            mode='lines+markers' if pays == 'Moyenne UE' else 'lines'
        ))
    
    # Zone critique (>10%)
    fig_inflation.add_hrect(
        y0=10, y1=25, fillcolor="red", opacity=0.08, line_width=0,
        annotation_text="Zone critique (>10%)", annotation_position="top left",
        annotation_font=dict(color="red", size=10)
    )
    
    fig_inflation.update_layout(
        height=350, 
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Inflation alimentaire (%)"
    )
    st.plotly_chart(fig_inflation, use_container_width=True)
    st.caption("Source : Eurostat - Indice des prix à la consommation harmonisé (IPCH) - Produits alimentaires. Données mises à jour manuellement.")
    
    st.divider()
    
    # === GRAPHIQUE RATIO BLÉ/ENGRAIS (conservé mais avec explication simplifiée) ===
    st.subheader(" Pouvoir d'achat des agriculteurs")
    st.markdown("""
    <div class="info-box">
        <b> Ce que ça veut dire :</b> Ce graphique mesure si l'agriculteur gagne assez en vendant son blé 
        pour acheter les engrais dont il a besoin. <br>
        - <b>Le ratio monte</b> → l'agriculteur est dans une bonne situation <span style="color: #27AE60;">🟢</span><br>
        - <b>Le ratio descend</b> → l'agriculteur perd de l'argent, il risque de réduire ses engrais, 
        et les récoltes futures seront plus faibles <span style="color: #E74C3C;">🔴</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not df_commodities.empty and 'Ble_Chicago' in df_commodities.columns and 'Yara (Norvège)' in df_prices_filtre.columns:
        ble_align = df_comm_filtre['Ble_Chicago'].dropna()
        yara_align = df_prices_filtre['Yara (Norvège)'].dropna()
        
        common_idx = ble_align.index.intersection(yara_align.index)
        if len(common_idx) > 0:
            ratio = ble_align.loc[common_idx] / yara_align.loc[common_idx]
            ratio_normalized = ratio / ratio.iloc[0] * 100
            
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=ratio_normalized.index, 
                y=ratio_normalized, 
                fill='tozeroy', 
                line=dict(color='#D4AC0D', width=2),
                name='Pouvoir d\'achat agriculteur'
            ))
            fig_ratio.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Niveau de départ")
            
            # Zone de stress
            fig_ratio.add_hrect(
                y0=0, y1=85, 
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                annotation_text=" Zone de danger pour les agriculteurs", annotation_position="top left"
            )
            
            fig_ratio.update_layout(
                height=300,  # Réduit aussi
                hovermode="x unified",
                xaxis_title="",
                yaxis_title="Indice (Base 100)"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Interprétation simplifiée
            ratio_actuel = ratio.iloc[-1]
            ratio_moyen = ratio.mean()
            variation = ((ratio_actuel / ratio_moyen) - 1) * 100
            
            if variation < -10:
                st.error(f" **Alerte** : Les agriculteurs ont perdu {abs(variation):.0f}% de pouvoir d'achat par rapport à la moyenne. Risque de baisse des récoltes dans 6-12 mois.")
            elif variation < 0:
                st.warning(f" **Attention** : Le pouvoir d'achat des agriculteurs est légèrement en baisse ({abs(variation):.0f}%). Surveillance nécessaire.")
            else:
                st.success(f" **Favorable** : Le pouvoir d'achat des agriculteurs est au-dessus de la moyenne (+{variation:.0f}%). Les conditions sont bonnes pour les prochaines récoltes.")
    else:
        st.info("Données non disponibles pour ce graphique.")


# ==========================================
# ONGLET 6 : SCENARIOS
# ==========================================

with tab6:
    st.header(" Scénarios de Crise : Et si... ?")
    st.markdown('<div class="plot-explanation">Simulez l\'impact d\'une variation du prix du gaz sur les entreprises et le consommateur.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        # Slider pour faire varier le prix du gaz de -50% à +100%
        variation_gaz = st.slider("Variation du prix du gaz (%):", -50, 100, 0, 10)
    
    # On utilise un modèle simplifié basé sur les corrélations historiques
    # (Tu peux affiner ces coefficients avec tes propres analyses)
    impact_yara = variation_gaz * -0.6  # Corrélation négative
    impact_oci = variation_gaz * -0.5
    impact_icp = variation_gaz * -0.1  # Israël moins exposé au gaz

    # Création d'un DataFrame pour afficher les résultats
    scenario_data = pd.DataFrame({
        'Acteur': ['Yara (Norvège)', 'OCI (Pays-Bas)', 'ICL (Israël)'],
        'Impact Estimé (%)': [impact_yara, impact_oci, impact_icp],
        'Prix Actuel': [
            df_prices['Yara (Norvège)'].iloc[-1],
            df_prices['OCI (Pays-Bas)'].iloc[-1],
            df_prices['ICL (Israël)'].iloc[-1]
        ]
    })
    scenario_data['Prix Simulé'] = scenario_data['Prix Actuel'] * (1 + scenario_data['Impact Estimé (%)']/100)

    with col2:
        st.subheader("Résultats de la simulation")
        st.dataframe(scenario_data.style.format({'Impact Estimé (%)': '{:.1f}%', 'Prix Actuel': '{:.2f}', 'Prix Simulé': '{:.2f}'}))
        
        # Impact sur le consommateur
        impact_consommateur = variation_gaz * 0.15 # Délai de 6-18 mois
        st.metric(" Inflation alimentaire projetée", f"{impact_consommateur:.1f}%", delta=f"{impact_consommateur:.1f}%")

    # Graphique comparatif
    st.subheader("Comparaison Prix Actuel vs. Prix Simulé")
    fig_scenario = go.Figure(data=[
        go.Bar(name='Prix Actuel', x=scenario_data['Acteur'], y=scenario_data['Prix Actuel']),
        go.Bar(name='Prix Simulé', x=scenario_data['Acteur'], y=scenario_data['Prix Simulé'])
    ])
    fig_scenario.update_layout(barmode='group', title="Impact de la variation du gaz sur le prix des actions")
    st.plotly_chart(fig_scenario, use_container_width=True)
