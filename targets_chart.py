import pandas as pd
import matplotlib.pyplot as plt

# Zakładam, że Twoja klasa znajduje się w pliku portfolio_symulator.py
from portfolio_symulator import DCAPortfolioSimulator


def main():
    print("Wczytywanie i przygotowywanie danych...")

    # 1. Wczytanie danych surowych (ceny)
    raw_df = pd.read_csv('btc_raw_data.csv')
    raw_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])

    # 2. Wczytanie cech ML (sygnały)
    ml_df = pd.read_csv('btc_ml_features_final.csv')
    ml_df['Date'] = pd.to_datetime(ml_df['Date'])

    # 3. Synchronizacja danych (Inner Join na Dacie)
    df = pd.merge(
        raw_df[['Date', 'BTC_Close', 'BTC_Open']],
        ml_df[['Date', 'Target']],
        on='Date',
        how='inner'
    )

    # --- NOWY KOD: PRZYCIĘCIE OD 2022 ROKU ---
    # Filtrujemy ramkę danych, zostawiając tylko wiersze od 1 stycznia 2022 włącznie
    # df = df[df['Date'] >= '2020-01-01'].reset_index(drop=True)

    print(f"Dane wyrównane i przycięte. Liczba dni do symulacji: {len(df)}")

    # 4. Inicjalizacja i wykonanie symulacji
    simulator = DCAPortfolioSimulator(initial_capital=1000, dca_amount=1000)

    # Uruchomienie symulacji od 2022 roku
    results = simulator.simulate(
        dates=df['Date'],
        prices=df['BTC_Close'],
        signals=df['Target'],
        execution_prices=df['BTC_Open']
    )

    # 5. Wyświetlenie wykresu
    print("Generowanie wykresu...")
    simulator.plot_simulation_results(results, execution_prices=df['BTC_Open'])


if __name__ == "__main__":
    main()