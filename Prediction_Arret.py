import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, mean_squared_error
from collections import Counter
import altair as alt

# 1. Définition du problème
st.title("🔧 Analyse Prédictive des Dysfonctionnements - UMS PASTA")
st.markdown("""
Ce tableau de bord permet de **prédire les probabilités de dysfonctionnement** des équipements 
à partir des données historiques des temps d'arrêt.
""")

# 2. Collecte des données
st.sidebar.markdown("## 📊 Données à analyser")
uploaded_file = st.sidebar.file_uploader(
    label="📁**Sélectionnez un fichier CSV :**", 
    type=["csv"], 
    help="Format accepté : .csv uniquement"
)
st.sidebar.info("📎 Le fichier doit contenir les colonnes : Date, Equipements, Temps d'immobilisation, etc.")

# === ÉTAPE 1 : Chargement du fichier ===
if uploaded_file:
    if st.button("✅ Étape 1 : Charger et afficher les premières lignes"):
        df = pd.read_csv(uploaded_file)
        st.success("Fichier chargé avec succès !")
        st.write(df.head())

        # Nettoyage des équipements
        def nettoyer_equipements(df):
            df['Equipements'] = df['Equipements'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
            return df

        df = nettoyer_equipements(df)

        # Conversion date en mois
        if not np.issubdtype(df['Date'].dtype, np.number):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.month

        st.session_state.df = df  # Sauvegarde pour étapes suivantes
        
# === ÉTAPE 2 : Prétraitement et agrégation ===
if "df" in st.session_state:
    if st.button("✅ Étape 2 : Prétraitement et agrégation"):
        df = st.session_state.df
        premiers = df[df['Date'] <= 6]
        derniers = df[df['Date'] > 6]

        def calculer_aggregat(df_):
            return df_.groupby('Equipements').agg(
                Récurrence=('Equipements', 'size'),
                Temps_total=("Temps D'immobilisation", 'sum')
            ).reset_index()

        agg_premiers = calculer_aggregat(premiers)
        agg_derniers = calculer_aggregat(derniers)

        st.session_state.agg_premiers = agg_premiers
        st.session_state.agg_derniers = agg_derniers

        st.subheader("📊 Données Agrégées - 6 Premiers Mois")
        st.dataframe(agg_premiers)
        
# === ÉTAPE 3 : Feature Engineering ===
if "agg_premiers" in st.session_state:
    st.sidebar.subheader("🔧 Étape 3 : Feature Engineering")
    threshold = st.sidebar.slider("📏 Seuil de récurrence", 1, 20, 3)

    if st.button("✅ Appliquer le seuil et générer les cibles"):
        def preprocess(df, t):
            df = df.rename(columns={"Récurrence": "Recurrence", "Temps_total": "Total_Time"})
            df["Dysfunction"] = (df["Recurrence"] > t).astype(int)
            return df

        agg_premiers = preprocess(st.session_state.agg_premiers.copy(), threshold)
        agg_derniers = preprocess(st.session_state.agg_derniers.copy(), threshold)

        X_premiers = agg_premiers[["Recurrence", "Total_Time"]]
        y_premiers = agg_premiers["Dysfunction"]
        X_derniers = agg_derniers[["Recurrence", "Total_Time"]]
        y_derniers = agg_derniers["Dysfunction"]

        scaler = StandardScaler()
        X_premiers_scaled = scaler.fit_transform(X_premiers)
        X_derniers_scaled = scaler.transform(X_derniers)

        st.session_state.X_premiers_scaled = X_premiers_scaled
        st.session_state.X_derniers_scaled = X_derniers_scaled
        st.session_state.y_premiers = y_premiers
        st.session_state.agg_premiers_feat = agg_premiers
        st.session_state.agg_derniers_feat = agg_derniers

        st.success("Seuil appliqué et variables prêtes pour l'entraînement.")

# === ÉTAPE 4 : Entraînement des modèles ===
if "X_premiers_scaled" in st.session_state:
    if st.button("✅ Étape 4 : Entraîner les modèles"):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(st.session_state.X_premiers_scaled, st.session_state.y_premiers)
            y_proba = model.predict_proba(st.session_state.X_premiers_scaled)[:, 1]
            results[name] = y_proba

        probs_df = pd.DataFrame(results, index=st.session_state.agg_premiers_feat["Equipements"])
        st.session_state.probs_df = probs_df

        st.success("Modèles entraînés avec succès.")
        st.subheader("📈 Probabilités prédites")
        st.dataframe(probs_df)
        
# === ÉTAPE 5 : Évaluation des prédictions ===
if "probs_df" in st.session_state:
    if st.button("✅ Étape 5 : Comparer avec les données passées"):
        agg_derniers = st.session_state.agg_derniers_feat.copy()
        agg_derniers["Recurrence_Bin"] = pd.cut(agg_derniers["Recurrence"], bins=5, labels=False)
        agg_derniers["Total_Time_Bin"] = pd.cut(agg_derniers["Total_Time"], bins=5, labels=False)

        observed_probs = agg_derniers.groupby(["Recurrence_Bin", "Total_Time_Bin"])["Dysfunction"].mean().reset_index()
        observed_probs = observed_probs.rename(columns={"Dysfunction": "Observed_Probability"})

        agg_derniers = agg_derniers.merge(observed_probs, on=["Recurrence_Bin", "Total_Time_Bin"], how="left")
        probs_past = agg_derniers[["Equipements", "Observed_Probability"]].set_index("Equipements")

        common_index = st.session_state.probs_df.index.intersection(probs_past.index)
        diff_df = pd.DataFrame(index=common_index)

        for model_name in st.session_state.probs_df.columns:
            diff_df[f"{model_name}"] = (
                probs_past.loc[common_index]["Observed_Probability"] -
                st.session_state.probs_df.loc[common_index][model_name]
            )
        
        # 🔄 Sauvegarde pour étape 6
        st.session_state.diff_df = diff_df

        st.subheader("📉 Écarts entre prédictions et observations")
        st.dataframe(diff_df)
        
        
        # Préparation des données au format long pour Altair
        diff_df_abs = diff_df.abs().reset_index().rename(columns={"index": "Equipements"})
        diff_long_bar = diff_df_abs.melt(id_vars="Equipements", var_name="Modèle", value_name="Écart")

        # Création du graphique barres Altair avec padding personnalisé
        bar_chart = alt.Chart(diff_long_bar).mark_bar().encode(
            x=alt.X('Equipements:N', title='Équipements', sort=None, axis=alt.Axis(labelAngle=90,labelFontSize=11)),
            y=alt.Y('Écart:Q', title='Écart Absolu'),
            color='Modèle:N',
            tooltip=['Equipements', 'Modèle', 'Écart']
        ).properties(
            width=1100,
            height=600,
            title="Écarts Absolus entre Prédictions et Observations",
            padding={'left': 30, 'top': 10, 'right': 5, 'bottom': 45},
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            titlePadding=10
        ).configure_legend(
            orient='bottom',
            title=None,
            labelFontSize=15
        ).interactive()

        st.altair_chart(bar_chart, use_container_width=True)
        
        # Line chart interactif avec points foncés (Altair)
        st.write("📈 Évolution interactive des écarts par modèle")

        # Mise en forme des données pour Altair
        diff_df_reset = diff_df.abs().reset_index().rename(columns={"index": "Equipements"})
        diff_long = diff_df_reset.melt(id_vars="Equipements", var_name="Modèle", value_name="Écart")

        line_chart = alt.Chart(diff_long).mark_line(point=alt.OverlayMarkDef(color='black')).encode(
            x=alt.X('Equipements:N', sort=None, title='Équipements',axis=alt.Axis(labelAngle=90,labelFontSize=11)),
            y=alt.Y('Écart:Q', title='Écart Absolu'),
            color='Modèle:N',
            tooltip=['Equipements', 'Modèle', 'Écart']
        ).properties(
            width=1100,
            height=600,
            title="",
            padding={'left': 30, 'top': 10, 'right': 5, 'bottom': 45}  # 🛠️ Ajoute de l'espace à gauche
        ).configure_legend(
            orient='bottom',
            title=None,
            labelFontSize=15,
            labelLimit=200).interactive()

        st.altair_chart(line_chart, use_container_width=True)


        # Évaluation finale
        avg_diff = {col: diff_df[col].abs().mean() for col in diff_df.columns}
        st.subheader("📊 Résumé de l'écart moyen par modèle")
        st.write(pd.DataFrame.from_dict(avg_diff, orient='index', columns=["Écart Moyen Probabiliste"]))
        
# === ÉTAPE 6 : Affichage du tableau de bord global structuré ===
if (
    "probs_df" in st.session_state and 
    "agg_derniers_feat" in st.session_state and 
    "diff_df" in st.session_state
):
    # Gestion de l'état persistant du bouton pour afficher le dashboard
    if st.button("📊 Accéder au Tableau de Bord Complet"):
        st.session_state.show_dashboard = True

    if st.session_state.get("show_dashboard", False):
        st.markdown("## 🧭 Tableau de Bord Prédictif Complet")

        # Onglets pour organiser les vues
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Probabilités Prédites",
            "📉 Écarts Absolus (Barres)",
            "📉 Écarts Absolus (Lignes)",
            "📊 Résumé & Export"
        ])

        with st.sidebar:
            st.markdown("## 🔍 Filtres Avancés")

            # Filtrer par modèle
            modeles = st.session_state.probs_df.columns.tolist()
            selected_modeles = st.multiselect("Modèles à afficher :", modeles, default=modeles)

            # Filtrer par intervalles de probabilité
            prob_min, prob_max = st.slider(
                "🎯 Filtrer par Probabilité Moyenne :", 0.0, 1.0, (0.0, 1.0), 0.01
            )

            # Trier
            #tri_option = st.selectbox(
                #"🔽 Trier les équipements par :", 
                #["Aucun", "Probabilité Croissante", "Probabilité Décroissante"]
            #)

        # Construction de la table de probabilités moyennes
        probs_df = st.session_state.probs_df[selected_modeles].copy()
        probs_df["Moyenne"] = probs_df.mean(axis=1)

        # Filtrage par intervalle de probabilité
        filtered_probs_df = probs_df[
            (probs_df["Moyenne"] >= prob_min) & (probs_df["Moyenne"] <= prob_max)
        ].copy()

        # Tri des équipements
        #if tri_option == "Probabilité Croissante":
            #filtered_probs_df = filtered_probs_df.sort_values("Moyenne", ascending=True)
        #elif tri_option == "Probabilité Décroissante":
            #filtered_probs_df = filtered_probs_df.sort_values("Moyenne", ascending=False)

        # Sélection sécurisée des équipements présents dans les deux datasets
        available_equipements = st.session_state.diff_df.index.intersection(filtered_probs_df.index)
        filtered_probs_df = filtered_probs_df.loc[available_equipements]
        filtered_diff_df = st.session_state.diff_df.loc[available_equipements][selected_modeles]
        selected_equipements = available_equipements.tolist()

        # Mise au format long pour les graphiques Altair
        filtered_probs_long = filtered_probs_df[selected_modeles].reset_index().melt(
            id_vars="Equipements", var_name="Modèle", value_name="Probabilité"
        )
        filtered_diff_long = filtered_diff_df.abs().reset_index().rename(
            columns={"index": "Equipements"}
        ).melt(id_vars="Equipements", var_name="Modèle", value_name="Écart")

        # --- Onglet 1 : Probabilités
        with tab1:
            st.subheader("📈 Probabilités Prédites")
            st.dataframe(filtered_probs_df[selected_modeles])

            prob_chart = alt.Chart(filtered_probs_long).mark_bar().encode(
                x=alt.X('Equipements:N', sort=None, axis=alt.Axis(labelAngle=90, labelFontSize=11)),
                y=alt.Y('Probabilité:Q'),
                color='Modèle:N',
                tooltip=['Equipements', 'Modèle', 'Probabilité']
            ).properties(width=1100, height=500)

            st.altair_chart(prob_chart, use_container_width=True)

        # --- Onglet 2 : Écarts absolus - Barres
        with tab2:
            st.subheader("📉 Écarts Absolus - Barres")
            bar_chart = alt.Chart(filtered_diff_long).mark_bar().encode(
                x=alt.X('Equipements:N', sort=None, axis=alt.Axis(labelAngle=90, labelFontSize=11)),
                y='Écart:Q',
                color='Modèle:N',
                tooltip=['Equipements', 'Modèle', 'Écart']
            ).properties(width=1100, height=500)

            st.altair_chart(bar_chart, use_container_width=True)

        # --- Onglet 3 : Écarts absolus - Lignes
        with tab3:
            st.subheader("📉 Écarts Absolus - Lignes")
            line_chart = alt.Chart(filtered_diff_long).mark_line(
                point=alt.OverlayMarkDef(color='black')
            ).encode(
                x=alt.X('Equipements:N', sort=None, axis=alt.Axis(labelAngle=90, labelFontSize=11)),
                y='Écart:Q',
                color='Modèle:N',
                tooltip=['Equipements', 'Modèle', 'Écart']
            ).properties(width=1100, height=500)

            st.altair_chart(line_chart, use_container_width=True)

        # --- Onglet 4 : Résumé & Export
        with tab4:
            st.subheader("📊 Écart Moyen par Modèle (Équipements filtrés)")
            avg_filtered = {
                col: filtered_diff_df[col].abs().mean() for col in filtered_diff_df.columns
            }
            st.write(pd.DataFrame.from_dict(avg_filtered, orient='index', columns=["Écart Moyen Probabiliste"]))

            st.markdown("### 📥 Télécharger les données filtrées")
            col1, col2 = st.columns(2)
            with col1:
                csv_probs = filtered_probs_df[selected_modeles].to_csv(index=True).encode('utf-8')
                st.download_button("⬇️ Exporter Probabilités", csv_probs, file_name="probabilites_filtrees.csv", mime="text/csv")
            with col2:
                csv_diff = filtered_diff_df.to_csv(index=True).encode('utf-8')
                st.download_button("⬇️ Exporter Écarts", csv_diff, file_name="ecarts_filtrees.csv", mime="text/csv")

else:
    st.warning("Veuillez charger un fichier CSV pour commencer.")