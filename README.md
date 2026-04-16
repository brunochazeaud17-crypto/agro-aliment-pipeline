# AgroSentinel

Veille automatisée sur la souveraineté alimentaire européenne. Ce projet suit quotidiennement les valorisations boursières des principaux producteurs d'engrais, les prix des matières premières (gaz naturel, blé) et analyse le sentiment des actualités géopolitiques pour anticiper les tensions sur les prix alimentaires.

## Fonctionnalités

- Collecte automatique des cours de 6 géants européens des engrais (Yara, OCI, ICL, CF Industries, Grupa Azoty, K+S)
- Suivi du prix du gaz TTF et du blé (Chicago SRW)
- Récupération des actualités sectorielles via NewsAPI et analyse de sentiment (NLP / VADER)
- Intégration des données macroéconomiques : indice FAO des prix alimentaires, stocks céréaliers USDA
- Dashboard interactif avec Streamlit et Plotly (6 onglets d'analyse)
- Mise à jour quotidienne automatisée via GitHub Actions

## Accès rapide

- Application en ligne : [https://agro-sentinel.streamlit.app](https://agro-sentinel.streamlit.app)
- Code source : ce dépôt

## Installation locale

```bash
git clone https://github.com/brunochazeaud17-crypto/agro-aliment-pipeline.git
cd agro-aliment-pipeline
pip install -r requirements.txt