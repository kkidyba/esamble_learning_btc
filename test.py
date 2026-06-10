import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

# Import Twojego symulatora
from portfolio_symulator import DCAPortfolioSimulator

# Ukrycie ostrzeżeń dotyczących optymalizacji i deprecjacji
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TimeSeriesDataLoader:
    """
    Klasa odpowiedzialna za bezpieczne ładowanie, czyszczenie i wyrównywanie
    szeregów czasowych z plików bazowych.
    """

    def __init__(self, raw_data_path: str, features_path: str):
        self.raw_data_path = raw_data_path
        self.features_path = features_path

    def load_and_align(self, target_column: str) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_path) or not os.path.exists(self.features_path):
            raise FileNotFoundError("Upewnij się, że pliki danych znajdują się w podanej ścieżce.")

        df_raw = pd.read_csv(self.raw_data_path)
        if 'Unnamed: 0' in df_raw.columns:
            df_raw.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw = df_raw[['Date', 'BTC_Close']].sort_values('Date').set_index('Date')

        df_feat = pd.read_csv(self.features_path)
        if 'Unnamed: 0' in df_feat.columns:
            df_feat.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        df_feat['Date'] = pd.to_datetime(df_feat['Date'])
        df_feat = df_feat.sort_values('Date').set_index('Date')

        if target_column not in df_feat.columns:
            raise KeyError(f"Kolumna targetu '{target_column}' nie została znaleziona w pliku cech.")

        df_combined = df_feat.join(df_raw[['BTC_Close']], how='inner')
        # df_combined.dropna(subset=[target_column], inplace=True)
        print(df_combined)
        print(f"Pomyślnie dopasowano dane. Zakres: {df_combined.index.min().strftime('%Y-%m-%d')} "
              f"do {df_combined.index.max().strftime('%Y-%m-%d')}. Liczba obserwacji: {len(df_combined)}")

        return df_combined


import pandas as pd


class WalkForwardSplitter:
    """
    Zapewnia rygorystyczny, chronologiczny podział danych bez mieszania.
    Obsługuje ułamki lat (np. 0.5 to 6 miesięcy) wykorzystując DateOffset.
    Posiada opcję 'expanding_window', która pozwala na "zakotwiczenie"
    początku zbioru treningowego na pierwszej dostępnej dacie.
    """

    def __init__(self, train_years: float = 3.0, val_years: float = 1.0, test_years: float = 1.0,
                 expanding_window: bool = False):
        # Konwersja ułamków lat na równe miesiące
        self.train_months = int(train_years * 12)
        self.val_months = int(val_years * 12)
        self.test_months = int(test_years * 12)
        self.expanding_window = expanding_window

        if self.test_months == 0:
            raise ValueError("Wartość test_years musi wynosić przynajmniej (np. 1 miesiąc = 0.083).")

    def generate_windows(self, df: pd.DataFrame):
        # Punkt startowy: 1 stycznia pierwszego dostępnego roku w danych
        start_year = df.index.min().year
        initial_start = pd.to_datetime(f"{start_year}-01-01")

        # current_start będzie służyć do wyznaczania końca okna treningowego (oraz val/test)
        current_start = initial_start

        # Limit górny, po którym pętla się zatrzyma
        end_date_limit = df.index.max()

        while True:
            # --- WYZNACZANIE OKIEN CZASOWYCH ---

            # Train
            # Jeśli expanding_window = True, zawsze zaczynamy od initial_start.
            # W przeciwnym razie okno się przesuwa (zaczyna od current_start).
            train_start = initial_start if self.expanding_window else current_start
            train_end = current_start + pd.DateOffset(months=self.train_months) - pd.Timedelta(days=1)

            # Val
            val_start = current_start + pd.DateOffset(months=self.train_months)
            val_end = val_start + pd.DateOffset(months=self.val_months) - pd.Timedelta(days=1)

            # Test
            test_start = val_start + pd.DateOffset(months=self.val_months)
            test_end = test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1)

            # Warunek stopu: wyjście poza dostępne w pliku ramy czasowe
            if test_start > end_date_limit:
                break

            # Wycięcie danych
            df_train = df.loc[train_start:train_end]
            df_val = df.loc[val_start:val_end]
            df_test = df.loc[test_start:test_end]

            if df_test.empty:
                break

            yield {
                'metadata': {
                    'train_range': (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                    'val_range': (val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d')),
                    'test_range': (test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))
                },
                'train': df_train,
                'val': df_val,
                'test': df_test
            }

            # --- PRZESUNIĘCIE OKNA ---
            # Przesuwamy wskaźnik końca treningu o długość trwania testu.
            current_start += pd.DateOffset(months=self.test_months)


class XGBoostOptunaOptimizer:
    """
    Optymalizuje model XGBoost pod kątem maksymalizacji wskaźnika Sortino,
    wykorzystując zewnętrzny symulator portfela na zbiorze walidacyjnym.
    """

    def __init__(self, simulator_params: dict, n_trials: int = 30):
        self.simulator_params = simulator_params
        self.n_trials = n_trials

    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 val_dates: np.ndarray, val_prices: np.ndarray) -> dict:

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 2, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,

                # --- NOWE PARAMETRY GPU ---
                'tree_method': 'hist',  # Algorytm optymalny dla GPU
                'device': 'cuda',  # Wskazanie na użycie karty graficznej NVIDIA
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds_class = model.predict(X_val)

            simulator = DCAPortfolioSimulator(**self.simulator_params)

            try:
                results = simulator.simulate(
                    dates=val_dates,
                    prices=val_prices,
                    signals=preds_class,
                    execution_prices=val_prices
                )
                sortino = results['Sortino']
                if pd.isna(sortino) or np.isinf(sortino):
                    return 0.0
                return sortino
            except Exception as e:
                return -99.0

        study = optuna.create_study(direction='maximize')
        # Uruchomienie np. 4 procesów roboczych na raz
        study.optimize(objective, n_trials=self.n_trials, n_jobs=8)

        return study.best_params



class WalkForwardOrchestrator:
    def __init__(self, data_loader, splitter, optimizer, target_column):
        self.data_loader = data_loader
        self.splitter = splitter
        self.optimizer = optimizer
        self.target_column = target_column

    def execute(self) -> pd.DataFrame:
        print("Trwa ładowanie i wyrównywanie szeregów czasowych...")
        df_master = self.data_loader.load_and_align(self.target_column)

        all_out_of_sample_predictions = []
        cols_to_drop = [self.target_column, 'BTC_Close']

        for step, window in enumerate(self.splitter.generate_windows(df_master), 1):
            meta = window['metadata']
            print(f"\n--- KROK {step} | Okres Testowy: {meta['test_range'][0]} do {meta['test_range'][1]} ---")
            print(f"Train: {meta['train_range'][0]} do {meta['train_range'][1]} | "
                  f"Val: {meta['val_range'][0]} do {meta['val_range'][1]}")

            train_df = window['train']
            val_df = window['val']
            test_df = window['test']

            X_train = train_df.drop(columns=cols_to_drop)
            y_train = train_df[self.target_column]

            X_val = val_df.drop(columns=cols_to_drop)
            y_val = val_df[self.target_column]

            X_test = test_df.drop(columns=cols_to_drop)
            y_test = test_df[self.target_column]

            val_dates = val_df.index.values
            val_prices = val_df['BTC_Close'].values

            print("Szukanie hiperparametrów pod kątem maksymalizacji Sortino...")
            best_params = self.optimizer.optimize(
                X_train, y_train, X_val, y_val, val_dates, val_prices
            )
            print(f"Najlepsze parametry (Sortino): {best_params}")

            # --- DODAJ TE LINIE PRZED TRENOWANIEM FINALNEGO MODELU ---
            best_params['tree_method'] = 'hist'
            best_params['device'] = 'cuda'
            best_params['objective'] = 'binary:logistic'
            best_params['random_state'] = 42
            # -----------------------------------------------------------

            X_train_val = pd.concat([X_train, X_val])
            y_train_val = pd.concat([y_train, y_val])

            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(X_train_val, y_train_val)

            preds_proba = final_model.predict_proba(X_test)[:, 1]
            preds_class = final_model.predict(X_test)

            results_df = pd.DataFrame({
                'Date': test_df.index,
                'BTC_Close': test_df['BTC_Close'].values,
                'True_Target': y_test.values,
                'XGB_Prob_Class1': preds_proba,
                'XGB_Signal': preds_class
            }).set_index('Date')

            all_out_of_sample_predictions.append(results_df)

        final_results = pd.concat(all_out_of_sample_predictions)
        return final_results


def evaluate_xgboost_strategy(predictions_file_path: str, sim_params: dict):
    """
    Funkcja, która wczytuje wygenerowane targety i uruchamia symulację portfela.
    """
    print(f"\nWczytywanie sygnałów z pliku: {predictions_file_path}")

    try:
        df_signals = pd.read_csv(predictions_file_path)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {predictions_file_path}.")
        return

    df_signals['Date'] = pd.to_datetime(df_signals['Date'])
    df_signals = df_signals.sort_values('Date')

    dates = df_signals['Date'].values
    prices = df_signals['BTC_Close'].values
    signals = df_signals['XGB_Signal'].values

    print("Inicjalizacja symulatora portfela DCA do wizualizacji...")
    simulator = DCAPortfolioSimulator(**sim_params)

    print("Trwa ostateczna symulacja na danych Testowych (Out-of-Sample)...")
    results = simulator.simulate(
        dates=dates,
        prices=prices,
        signals=signals,
        execution_prices=prices
    )

    print("\n--- PODSUMOWANIE WYNIKÓW SYMULACJI (Out-of-Sample) ---")
    print(f"Całkowity zainwestowany kapitał (Fiat): ${results['Total_Invested']:,.2f}")
    print(f"Wartość końcowa portfela (Strategia ML): ${results['Final_Equity']:,.2f}")
    print(f"ROI Strategii: {results['ROI'] * 100:.2f}%")
    print(f"Sortino Ratio Strategii: {results['Sortino']:.2f}")

    print("\nGenerowanie wykresu analizy portfela...")
    simulator.plot_simulation_results(results, execution_prices=prices)


# ==========================================
# GŁÓWNY BLOK EGZEKUCYJNY (Połączony)
# ==========================================
if __name__ == "__main__":
    # 1. KONFIGURACJA ŚCIEŻEK I ZMIENNYCH
    RAW_DATA_PATH = 'btc_raw_data.csv'
    FEATURES_PATH = 'btc_ml_features_final.csv'

    # UWAGA: Wpisz tutaj DOKŁADNĄ nazwę kolumny z targetem z pliku btc_ml_features_final.csv!
    TARGET_COLUMN_NAME = 'Target'

    PREDICTIONS_FILE = 'btc_xgboost_walk_forward_signals.csv'

    sim_params = {
        'initial_capital': 1000.0,
        'dca_amount': 1000.0,
        'fee_rate': 0.003,
        'mar': 0.0,
        'leverage': 0
    }

    # 2. INICJALIZACJA KOMPONENTÓW
    # Zdefiniowanie loadera (tego brakowało!)
    loader = TimeSeriesDataLoader(
        raw_data_path=RAW_DATA_PATH,
        features_path=FEATURES_PATH
    )

    train_years = 6
    val_years = 6

    while train_years > 3:
        test_years = 1
        while val_years > 0:
            if train_years >= val_years:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres


                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window = True
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=500
                )

                # Orkiestrator
                orchestrator = WalkForwardOrchestrator(
                    data_loader=loader,
                    splitter=splitter,
                    optimizer=optimizer,
                    target_column=TARGET_COLUMN_NAME
                )

                # 3. EGZEKUCJA - Faza uczenia i generowania sygnałów
                out_of_sample_df = orchestrator.execute()

                # Zapis finalnego zbioru sygnałów
                out_of_sample_df.to_csv(PREDICTIONS_FILE)

                print(f"\nSukces! Pełny wektor targetów od {out_of_sample_df.index.min().strftime('%Y-%m-%d')} "
                      f"do {out_of_sample_df.index.max().strftime('%Y-%m-%d')} został wygenerowany.")
                print(f"Dane zapisano do pliku: {PREDICTIONS_FILE}")
                print("Wyniki dla train: ", train_years, "Dla val: ", val_years, "Dla test: ", test_years)
                # 4. EGZEKUCJA - Faza testowania na wykresie
                evaluate_xgboost_strategy(PREDICTIONS_FILE, sim_params)

            val_years = val_years - 1




        test_years = 0.5

        while val_years > 0:
            if train_years >= val_years:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=500
                )

                # Orkiestrator
                orchestrator = WalkForwardOrchestrator(
                    data_loader=loader,
                    splitter=splitter,
                    optimizer=optimizer,
                    target_column=TARGET_COLUMN_NAME
                )

                # 3. EGZEKUCJA - Faza uczenia i generowania sygnałów
                out_of_sample_df = orchestrator.execute()

                # Zapis finalnego zbioru sygnałów
                out_of_sample_df.to_csv(PREDICTIONS_FILE)

                print(f"\nSukces! Pełny wektor targetów od {out_of_sample_df.index.min().strftime('%Y-%m-%d')} "
                      f"do {out_of_sample_df.index.max().strftime('%Y-%m-%d')} został wygenerowany.")
                print(f"Dane zapisano do pliku: {PREDICTIONS_FILE}")
                print("Wyniki dla train: ", train_years, "Dla val: ", val_years, "Dla test: ", test_years)
                # 4. EGZEKUCJA - Faza testowania na wykresie
                evaluate_xgboost_strategy(PREDICTIONS_FILE, sim_params)

            val_years = val_years - 1
        train_years= train_years - 1









