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
        La souveraineté alimentaire de l'Europe dépend à **70% des engrais importés**, dont une large part du gaz naturel (Russie, Moyen-Orient).  
        Toute tension géopolitique faisant flamber le prix du gaz... impact donc le prix des engrais, menaçant les récoltes Européennes.
        
        ### Comment ça fonctionne ?
        - **Données boursières** : Récupérées chaque jour via Yahoo Finance pour 6 géants européens des fertilisants.
        - **Matières premières** : Prix du Gaz naturel européen (TTF) et du Blé (Chicago) également via Yahoo Finance.
        - **Actualités** : Articles frais tirés de NewsAPI.org avec les mots-clés *fertilizer, Europe, gas, Middle East*.
        - **Analyse de sentiment** : Un algorithme de NLP (VADER) note chaque titre de -1 (très négatif) à +1 (très positif).
        - **Mise à jour** : Un script automatique tourne toutes les 24h via GitHub Actions.
        
        **Utilité concrète** : Anticiper les tensions sur les prix alimentaires, comprendre les marchés, suivre l'impact des crises sur la souveraineté alimentaire européenne.
        """)
    with col_desc2:
        st.markdown("""
        ### Sources
        - **Prix des actions** : Yahoo Finance  
        - **Gaz naturel (TTF)** : Yahoo Finance
        - **Blé (Chicago SRW)** : Yahoo Finance
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Analyse de l'état financier des géants", 
    "2. Radar de Crise", 
    "3. Journal de Bord",
    "4. Intrants & Dépendances",   # ← NOUVEAU
    "5. Impact Consommateur"       # ← NOUVEAU
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


# ==========================================
# ONGLET 5 : IMPACT CONSOMMATEUR & SOUVERAINETÉ
# ==========================================
with tab5:
    st.header(" De la Bourse à l'Assiette : Indicateurs de Souveraineté")
    
    st.markdown("""
    <div class="info-box">
        <b>📊 Mécanisme de transmission :</b> Une hausse du prix du gaz → hausse du coût des engrais azotés → 
        baisse des marges des agriculteurs → réduction des surfaces cultivées ou des rendements → 
        hausse des prix alimentaires (délai estimé : 6-12 mois).
    </div>
    """, unsafe_allow_html=True)

    if df_commodities.empty:
        st.warning(" Données des matières premières non disponibles.")
    else:
        # Filtrage sur la période sélectionnée
        mask_comm = (df_commodities.index.date >= start_date) & (df_commodities.index.date <= end_date)
        df_comm_filtre = df_commodities.loc[mask_comm]
        
        # Utilisation de colonnes avec proportions égales
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            # Conteneur pour aligner les hauteurs
            with st.container():
                st.subheader(" Score de Stress Alimentaire")
                st.markdown('<div class="plot-explanation">Indicateur composite calculé à partir des données réelles : Prix du Gaz (40%), Volatilité de Yara (30%), Ratio Blé/Engrais (30%).</div>', unsafe_allow_html=True)
                
                # Calcul du score composite basé sur les données RÉELLES
                score_stress = 50  # Valeur neutre par défaut
                
                if 'Gaz_Nat_EU' in df_comm_filtre.columns and 'Ble_Chicago' in df_comm_filtre.columns:
                    try:
                        # 1. Composante Gaz : écart à la moyenne (normalisé 0-100)
                        gaz_data = df_comm_filtre['Gaz_Nat_EU'].dropna()
                        if len(gaz_data) > 0:
                            gaz_recent = gaz_data.tail(30).mean()
                            gaz_moyenne_hist = gaz_data.mean()
                            if gaz_moyenne_hist > 0:
                                gaz_score = 50 + (gaz_recent / gaz_moyenne_hist - 1) * 40
                                gaz_score = max(0, min(100, gaz_score))
                            else:
                                gaz_score = 50
                        else:
                            gaz_score = 50
                        
                        # 2. Composante Ratio Blé/Engrais (proxy de rentabilité agricole)
                        if 'Yara (Norvège)' in df_prices_filtre.columns:
                            ble_data = df_comm_filtre['Ble_Chicago'].dropna()
                            yara_data = df_prices_filtre['Yara (Norvège)'].dropna()
                            common_idx = ble_data.index.intersection(yara_data.index)
                            
                            if len(common_idx) > 0:
                                ratio = ble_data.loc[common_idx] / yara_data.loc[common_idx]
                                ratio_recent = ratio.tail(30).mean()
                                ratio_moyenne = ratio.mean()
                                if ratio_moyenne > 0:
                                    ratio_score = 50 - (ratio_recent / ratio_moyenne - 1) * 30
                                    ratio_score = max(0, min(100, ratio_score))
                                else:
                                    ratio_score = 50
                            else:
                                ratio_score = 50
                        else:
                            ratio_score = 50
                        
                        # 3. Composante Volatilité de Yara
                        if 'Yara (Norvège)' in df_prices_filtre.columns:
                            yara_vol = df_prices_filtre['Yara (Norvège)'].pct_change().std() * np.sqrt(252) * 100
                            vol_score = min(100, yara_vol * 2) if not pd.isna(yara_vol) else 50
                        else:
                            vol_score = 50
                        
                        # Score composite pondéré
                        score_stress = int(gaz_score * 0.4 + vol_score * 0.3 + ratio_score * 0.3)
                        
                    except Exception as e:
                        score_stress = 50
                
                # Affichage de la jauge avec le score calculé
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_stress,
                    title={'text': "Score de Stress Alimentaire", 'font': {'size': 16}},
                    number={'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#2C3E50"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#27AE60'},
                            {'range': [40, 70], 'color': '#F1C40F'},
                            {'range': [70, 100], 'color': '#E74C3C'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=250,  # Hauteur réduite pour meilleur alignement
                    margin=dict(t=30, b=10, l=20, r=20),
                    font={'size': 12}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Interprétation dans un conteneur stylisé
                if score_stress < 40:
                    st.success(" **Stress faible** : Les conditions de marché sont favorables aux agriculteurs européens.")
                elif score_stress < 70:
                    st.warning(" **Stress modéré** : Des tensions apparaissent, surveillance recommandée.")
                else:
                    st.error(" **Stress élevé** : Risque important sur la souveraineté alimentaire européenne.")
        
        with col2:
            # Conteneur pour aligner les hauteurs
            with st.container():
                st.subheader("⏱️ Délai de Transmission estimé")
                st.markdown('<div class="plot-explanation">Corrélation croisée entre le prix du Gaz et l\'action Yara (proxy des engrais), décalée dans le temps.</div>', unsafe_allow_html=True)
                
                # Calcul du délai optimal via corrélation croisée
                if 'Gaz_Nat_EU' in df_comm_filtre.columns and 'Yara (Norvège)' in df_prices_filtre.columns:
                    gaz_series = df_comm_filtre['Gaz_Nat_EU'].dropna()
                    yara_series = df_prices_filtre['Yara (Norvège)'].dropna()
                    
                    common_idx = gaz_series.index.intersection(yara_series.index)
                    if len(common_idx) > 60:
                        # Calcul des rendements journaliers (variations en %)
                        gaz_returns = gaz_series.loc[common_idx].pct_change().dropna()
                        yara_returns = yara_series.loc[common_idx].pct_change().dropna()
                        
                        # Réalignement après dropna
                        common_returns_idx = gaz_returns.index.intersection(yara_returns.index)
                        gaz_ret_aligned = gaz_returns.loc[common_returns_idx]
                        yara_ret_aligned = yara_returns.loc[common_returns_idx]
                        
                        # Initialisation des listes pour stocker les corrélations
                        correlations = []
                        lag_values = []
                        
                        # Calcul des corrélations pour différents décalages
                        max_lag = min(90, len(gaz_ret_aligned) // 3)
                        
                        for lag in range(-max_lag, max_lag + 1):
                            if lag < 0:
                                abs_lag = abs(lag)
                                corr = gaz_ret_aligned.iloc[:-abs_lag].corr(yara_ret_aligned.iloc[abs_lag:])
                            elif lag > 0:
                                corr = gaz_ret_aligned.iloc[lag:].corr(yara_ret_aligned.iloc[:-lag])
                            else:
                                corr = gaz_ret_aligned.corr(yara_ret_aligned)
                            
                            correlations.append(corr if not pd.isna(corr) else 0)
                            lag_values.append(lag)
                        
                        # Trouver le lag avec la corrélation maximale (en valeur absolue)
                        correlations_abs = [abs(c) for c in correlations]
                        best_idx = np.argmax(correlations_abs)
                        best_lag = lag_values[best_idx]
                        best_corr = correlations[best_idx]
                        
                        # Conversion en mois ouvrés (21 jours = 1 mois)
                        mois_estimes = abs(best_lag) / 21
                        
                        # Métriques dans un conteneur pour équilibrer avec la jauge
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                label="Délai de transmission optimal",
                                value=f"{mois_estimes:.1f} mois",
                                delta=f"Corrélation : {best_corr:.2f}" if not pd.isna(best_corr) else None
                            )
                        with col_b:
                            direction_lag = "Gaz précède Yara" if best_lag > 0 else ("Yara précède Gaz" if best_lag < 0 else "Instantané")
                            st.metric(
                                label="Sens du décalage",
                                value=direction_lag,
                                delta=f"Lag = {best_lag} jours"
                            )
                        
                        # Espacement pour alignement
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Interprétation dans un conteneur avec bordure
                        with st.container():
                            st.markdown("---")
                            if pd.isna(best_corr) or abs(best_corr) < 0.1:
                                st.markdown(" **Corrélation faible ou inexistante**")
                                st.markdown("""
                                La relation entre le gaz et Yara n'est pas linéaire sur cette période.
                                Cela peut indiquer que d'autres facteurs (géopolitiques, saisonniers) 
                                dominent actuellement le marché.
                                """)
                            elif best_lag == 0:
                                st.markdown(" **Interprétation**")
                                st.markdown(f"""
                                La corrélation maximale ({best_corr:.2f}) est observée **sans décalage**.  
                                Le marché des engrais (Yara) réagit quasi-instantanément aux variations du prix du gaz.  
                                Cela peut indiquer une forte efficience du marché ou une période d'observation trop courte.
                                """)
                            elif best_lag > 0:
                                direction = "positive" if best_corr > 0 else "négative"
                                st.markdown(" **Interprétation**")
                                st.markdown(f"""
                                Une variation du prix du gaz met environ **{mois_estimes:.1f} mois** pour se répercuter significativement sur l'action Yara.  
                                Le Gaz **précède** Yara de **{best_lag} jours ouvrés**.  
                                La corrélation est **{direction} ({best_corr:.2f})**.  
                                
                                Ce délai reflète le temps de transmission des coûts de production aux marchés financiers.
                                """)
                            else:  # best_lag < 0
                                direction = "positive" if best_corr > 0 else "négative"
                                st.markdown(" **Interprétation**")
                                st.markdown(f"""
                                Contre-intuitivement, c'est Yara qui **précède** le Gaz de **{abs(best_lag)} jours ouvrés**.  
                                La corrélation est **{direction} ({best_corr:.2f})**.  
                                
                                Cela peut indiquer que le marché anticipe les tensions sur le gaz ou que d'autres facteurs 
                                (spéculation, saisonnalité) dominent la relation.
                                """)
                            
                            st.markdown("*Ce délai est recalculé automatiquement à chaque mise à jour des données.*")
                        
                        # Optionnel : Afficher le graphique des corrélations par lag
                        with st.expander(" Voir le détail des corrélations par décalage"):
                            fig_lags = go.Figure()
                            
                            colors = ['#E74C3C' if c < 0 else '#27AE60' for c in correlations]
                            fig_lags.add_trace(go.Bar(
                                x=lag_values,
                                y=correlations,
                                marker_color=colors,
                                name='Corrélation',
                                text=[f"{c:.3f}" for c in correlations],
                                textposition='outside'
                            ))
                            
                            fig_lags.add_vline(
                                x=best_lag, 
                                line_dash="dash", 
                                line_color="blue", 
                                annotation_text=f"Optimal: {best_lag} jours",
                                annotation_position="top"
                            )
                            
                            fig_lags.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
                            
                            fig_lags.update_layout(
                                title="Corrélation Gaz vs Yara par décalage (jours ouvrés)",
                                xaxis_title="Décalage en jours (négatif = Yara précède Gaz, positif = Gaz précède Yara)",
                                yaxis_title="Corrélation",
                                height=300,
                                showlegend=False,
                                bargap=0.1
                            )
                            st.plotly_chart(fig_lags, use_container_width=True)
                            
                            # Tableau récapitulatif
                            st.caption("📋 **Top 5 des meilleurs décalages :**")
                            correlations_with_lag = list(zip(lag_values, correlations, correlations_abs))
                            correlations_with_lag.sort(key=lambda x: x[2], reverse=True)
                            
                            top5_data = []
                            for lag, corr, abs_corr in correlations_with_lag[:5]:
                                sens = "Gaz → Yara" if lag > 0 else ("Yara → Gaz" if lag < 0 else "Instantané")
                                top5_data.append({
                                    "Décalage (jours)": lag,
                                    "Sens": sens,
                                    "Corrélation": f"{corr:.3f}",
                                    "|Corr|": f"{abs_corr:.3f}"
                                })
                            
                            df_top5 = pd.DataFrame(top5_data)
                            st.dataframe(df_top5, use_container_width=True, hide_index=True)
                            
                    else:
                        st.info("Données insuffisantes pour calculer le délai de transmission (minimum 60 jours requis).")
                else:
                    st.info("Données Gaz ou Yara non disponibles.")
    
    st.divider()
    
    st.subheader(" Évolution du Ratio Blé / Engrais")
    st.markdown('<div class="plot-explanation">Ce ratio mesure le pouvoir d\'achat des agriculteurs : quantité de blé nécessaire pour "acheter" une unité d\'engrais (proxy via l\'action Yara). Quand le ratio baisse, la rentabilité agricole se dégrade.</div>', unsafe_allow_html=True)
    
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
                name='Ratio Blé/Engrais (Base 100)'
            ))
            fig_ratio.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Début de période")
            fig_ratio.add_hline(y=ratio_normalized.mean(), line_dash="dot", line_color="blue", annotation_text="Moyenne période")
            
            fig_ratio.add_hrect(
                y0=0, y1=85, 
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Zone de stress", annotation_position="top left"
            )
            
            fig_ratio.update_layout(
                height=350, 
                hovermode="x unified",
                title="Ratio Blé / Engrais (Base 100 au début de période)"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            ratio_actuel = ratio.iloc[-1]
            ratio_moyen = ratio.mean()
            variation = ((ratio_actuel / ratio_moyen) - 1) * 100 if ratio_moyen > 0 else 0
            
            if variation < -10:
                st.error(f" Le ratio est **{abs(variation):.1f}% inférieur** à sa moyenne historique. Situation critique pour les agriculteurs.")
            elif variation < 0:
                st.warning(f" Le ratio est **{abs(variation):.1f}% inférieur** à sa moyenne historique.")
            else:
                st.success(f" Le ratio est **{variation:.1f}% supérieur** à sa moyenne historique.")
        else:
            st.info("Pas de dates communes entre le Blé et Yara.")
    else:
        st.info("Données Blé ou Yara non disponibles.")
    
    st.divider()
    st.caption("""
    **Méthodologie :** Tous les indicateurs de cet onglet sont calculés à partir des données réelles 
    mises à jour quotidiennement via Yahoo Finance (Gaz TTF, Blé Chicago, actions). 
    Le Score de Stress est un indicateur composite pondéré. 
    Le délai de transmission est calculé par corrélation croisée dynamique avec des décalages positifs et négatifs.
    """)
