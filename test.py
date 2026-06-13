import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import gc

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
    początku zbioru treningowego na pierwszej dostępnej dacie lub wybranej start_date.
    """

    def __init__(self, train_years: float = 3.0, val_years: float = 1.0, test_years: float = 1.0,
                 expanding_window: bool = False, start_date: str = None):
        # Konwersja ułamków lat na równe miesiące
        self.train_months = int(train_years * 12)
        self.val_months = int(val_years * 12)
        self.test_months = int(test_years * 12)
        self.expanding_window = expanding_window
        self.start_date = start_date  # <-- ZAPISUJEMY PARAMETR

        if self.test_months == 0:
            raise ValueError("Wartość test_years musi wynosić przynajmniej (np. 1 miesiąc = 0.083).")

    def generate_windows(self, df: pd.DataFrame):
        # --- USTALANIE PUNKTU STARTOWEGO ---
        if self.start_date is not None:
            initial_start = pd.to_datetime(self.start_date)
        else:
            # Domyślne zachowanie, jeśli nie podano daty
            start_year = df.index.min().year
            initial_start = pd.to_datetime(f"{start_year}-01-01")

        # current_start będzie służyć do wyznaczania końca okna treningowego (oraz val/test)
        current_start = initial_start

        # Limit górny, po którym pętla się zatrzyma
        end_date_limit = df.index.max()

        while True:
            # --- WYZNACZANIE OKIEN CZASOWYCH ---

            # Train
            # Jeśli expanding_window = True, zawsze zaczynamy od initial_start (naszej start_date).
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
            current_start += pd.DateOffset(months=self.test_months)


class XGBoostOptunaOptimizer:
    """
    Optymalizuje model XGBoost pod kątem maksymalizacji wskaźnika Sortino,
    wykorzystując zewnętrzny symulator portfela na zbiorze walidacyjnym.
    Zabezpieczone przed wyciekami pamięci (VRAM/RAM).
    """

    def __init__(self, simulator_params: dict, n_trials: int = 30):
        self.simulator_params = simulator_params
        self.n_trials = n_trials

    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 val_dates: np.ndarray, val_prices: np.ndarray) -> dict:

        # 1. DYNAMICZNY BALANS KLAS
        num_zeros = (y_train == 0).sum()
        num_ones = (y_train == 1).sum()
        scale_pos_weight_val = num_zeros / num_ones if num_ones > 0 else 1.0

        def objective(trial):
            # 2. DYNAMICZNY ROZMIAR LIŚCIA
            min_child_fraction = trial.suggest_float('min_child_fraction', 0.01, 0.05)
            min_child_weight_val = max(1, int(min_child_fraction * len(X_train)))

            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'min_child_weight': min_child_weight_val,
                'gamma': trial.suggest_float('gamma', 0.1, 3.0),
                'subsample': trial.suggest_float('subsample', 0.4, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
                'alpha': trial.suggest_float('alpha', 1e-2, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-2, 10.0, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'n_estimators': 2500,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'device': 'cuda',
                'random_state': 42,
                'scale_pos_weight': scale_pos_weight_val,
                'early_stopping_rounds': 50
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            best_iter = getattr(model, 'best_iteration', 2500)
            trial.set_user_attr("best_iteration", best_iter)

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

                # --- CZYSZCZENIE PAMIĘCI (SUKCES) ---
                del model
                del simulator
                del preds_class
                gc.collect()

                if pd.isna(sortino) or np.isinf(sortino):
                    return 0.0
                return sortino

            except Exception as e:
                # --- CZYSZCZENIE PAMIĘCI (BŁĄD SYMULACJI) ---
                del model
                del simulator
                if 'preds_class' in locals():
                    del preds_class
                gc.collect()
                return -99.0

        study = optuna.create_study(direction='maximize')
        # n_jobs=1 zapobiega błędom "CUDA Out of Memory"
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)

        best_trial = study.best_trial
        best_params_raw = best_trial.params

        optimal_trees = best_trial.user_attrs.get("best_iteration", 500)
        final_n_estimators = int(optimal_trees * 1.1)

        final_train_len = len(X_train) + len(X_val)
        final_min_child_weight = max(1, int(best_params_raw['min_child_fraction'] * final_train_len))

        final_model_params = {
            'max_depth': best_params_raw['max_depth'],
            'min_child_weight': final_min_child_weight,
            'gamma': best_params_raw['gamma'],
            'subsample': best_params_raw['subsample'],
            'colsample_bytree': best_params_raw['colsample_bytree'],
            'alpha': best_params_raw['alpha'],
            'lambda': best_params_raw['lambda'],
            'learning_rate': best_params_raw['learning_rate'],
            'n_estimators': final_n_estimators,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight_val
        }

        return final_model_params


class WalkForwardOrchestrator:
    """
    Zarządza podziałem danych i procesem trenowania/testowania z brutalnym czyszczeniem pamięci po każdym cyklu.
    """

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

            final_model_params = self.optimizer.optimize(
                X_train, y_train, X_val, y_val, val_dates, val_prices
            )
            print(f"Najlepsze parametry dla finalnego trenowania: {final_model_params}")

            X_train_val = pd.concat([X_train, X_val])
            y_train_val = pd.concat([y_train, y_val])

            final_model = xgb.XGBClassifier(**final_model_params)
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

            # --- AGRESYWNE CZYSZCZENIE PO KAŻDYM OKNIE CZASOWYM ---
            del train_df, val_df, test_df
            del X_train, y_train, X_val, y_val, X_test, y_test
            del X_train_val, y_train_val
            del final_model
            del preds_proba, preds_class, results_df
            del val_dates, val_prices
            gc.collect()
            # ------------------------------------------------------

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
    TARGET_COLUMN_NAME = 'Target'
    PREDICTIONS_FILE = 'btc_xgboost_walk_forward_signals.csv'
    sim_params = {
        'initial_capital': 1000.0,
        'dca_amount': 1000.0,
        'fee_rate': 0.003,
        'mar': 0.0,
        'leverage': 0
    }

    n_trials = 20
    test_years = 1
    loader = TimeSeriesDataLoader(
        raw_data_path=RAW_DATA_PATH,
        features_path=FEATURES_PATH
    )
    START_DATE_LEARN = '2012-01-01'
    train_years = 2
    while train_years < 8:
        val_years = 6

        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 8:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date= START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1

    START_DATE_LEARN = '2013-01-01'
    train_years = 2
    while train_years < 8:

        val_years = 6

        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 7:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date = START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1

    START_DATE_LEARN = '2014-01-01'
    while train_years < 8:
        val_years = 6
        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 6:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date = START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1

    START_DATE_LEARN = '2015-01-01'
    train_years = 2
    while train_years < 8:
        val_years = 6
        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 5:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date = START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1

    START_DATE_LEARN = '2016-01-01'
    train_years = 2
    while train_years < 8:
        val_years = 6

        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 4:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date = START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1

    START_DATE_LEARN = '2017-01-01'
    train_years = 2
    while train_years < 8:
        val_years = 6

        while val_years > 0:
            if train_years >= val_years and train_years + val_years == 3:
                # Parametry symulatora, pod które optymalizowany będzie XGBoost oraz podpięty finalny wykres

                # Układ okien z Twojego zapytania (6 lat uczenia, 2 lata walidacji, 1 rok testu)
                splitter = WalkForwardSplitter(
                    train_years=train_years,
                    val_years=val_years,
                    test_years=test_years,
                    expanding_window=True,
                    start_date = START_DATE_LEARN
                )

                # Optymalizator
                optimizer = XGBoostOptunaOptimizer(
                    simulator_params=sim_params,
                    n_trials=n_trials
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
        train_years= train_years + 1