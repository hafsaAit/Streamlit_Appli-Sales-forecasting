import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("Application de Prévision des Ventes")

uploaded_file = st.file_uploader("Choisissez un fichier Excel", type="xlsx")

if uploaded_file is not None:
    # Charger les données et les préparer
    df = pd.read_excel(uploaded_file)
    df['Jour Fiscal'] = pd.to_datetime(df['Jour Fiscal'])
    df.set_index('Jour Fiscal', inplace=True)
    df = df.asfreq('D')
    df['Ventes Totales TTC'] = df['Ventes Totales TTC'].replace(0, float('nan')).fillna(method='ffill')

    # Diviser les données en données d'entraînement et de test
    date_fin_train = '2022-10-01'
    df_train = df[df.index < date_fin_train]
    df_test = df[df.index >= date_fin_train]

    Jour_fiscal_formatted = df_test.index.strftime('%Y-%m-%d')

    st.markdown("**Table des ventes :**")
    table_data = pd.DataFrame({'Jour Fiscal': Jour_fiscal_formatted, 'Ventes Totales TTC': df_test['Ventes Totales TTC']}).reset_index(drop=True)

    st.dataframe(table_data, height=500, width=800)

    #  afficher le modèle de prévision
    st.markdown("**Prévision des ventes :**")
    forecast_period = st.slider("Période de prévision (en jours)", min_value=1, max_value=365, value=90)

    # Modèle de prévision
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=268
    )

    forecaster.fit(y=df_train['Ventes Totales TTC'])

    steps = len(df_test)
    rf_recursive_predictions = forecaster.predict(steps=steps)

    end_date = rf_recursive_predictions.index[-1]

    # Forecast
    steps_forecast = forecast_period
    forecast = forecaster.predict(
        steps=steps_forecast, last_window=df_test.loc[:'2023-07-01', "Ventes Totales TTC"])

    # Créer un index pour les prévisions
    forecast_index = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=steps_forecast, freq='D')

    # Créer une série pandas pour les prévisions avec l'index approprié
    forecast_series = pd.Series(forecast, index=forecast_index)

    # Afficher le graphe de prévision
    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(forecast_index, forecast_series, color='green', label='Prévision')
    plt.xlabel('Jour Fiscal')
    plt.ylabel('Ventes Totales (MAD) TTC')
    plt.title('Prévision des ventes')
    plt.legend()
    st.pyplot(fig1)

    # Calculer les intervalles de fluctuation
    n = len(forecast_series)
    t_value = 1.96
    std_error = np.std(forecast_series) / np.sqrt(n)
    margin_of_error = t_value * std_error

    forecast_lower = forecast_series - margin_of_error
    forecast_upper = forecast_series + margin_of_error

    # Afficher le graphe avec les intervalles de fluctuation
    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(forecast_series.index, forecast_series.values, color='green', label='Prévision')
    plt.fill_between(forecast_series.index, forecast_lower.values, forecast_upper.values, color='grey', alpha=0.3)

    # Tracer les données de test et les prédictions
    df_test['Ventes Totales TTC'].plot(label='Test')
    rf_recursive_predictions.plot(label='Prédictions')

    # Afficher les informations du graphe
    plt.xlabel('Jour Fiscal')
    plt.ylabel('Ventes Totales (MAD) TTC')
    plt.title('Prédictions et prévision des ventes')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig2)
