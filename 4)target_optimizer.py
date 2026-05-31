import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, matthews_corrcoef
import warnings

# --- IMPORT SYMULATORA ---
from portfolio_symulator import DCAPortfolioSimulator

# -------------------------

warnings.filterwarnings('ignore')


class BTCProxyOptimizer:
    def __init__(self, raw_data_path, features_path, horizon=28, hysteresis_k=7):
        print("Wczytywanie danych...")

        self.df_raw = pd.read_csv(raw_data_path)
        if 'Unnamed: 0' in self.df_raw.columns:
            self.df_raw = self.df_raw.rename(columns={'Unnamed: 0': 'Date'})
        self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'])
        self.df_raw = self.df_raw.sort_values('Date').reset_index(drop=True)

        self.df_features = pd.read_csv(features_path)
        if 'Unnamed: 0' in self.df_features.columns:
            self.df_features = self.df_features.rename(columns={'Unnamed: 0': 'Date'})
        self.df_features['Date'] = pd.to_datetime(self.df_features['Date'])

        self.horizon = horizon
        self.hysteresis_k = hysteresis_k

        self.simulator = DCAPortfolioSimulator(
            initial_capital=1000.0,
            dca_amount=1000.0,
            fee_rate=0.001,
            mar=0.0
        )

        # --- NOWY KOD: Dynamiczne wyliczanie daty odcięcia (60%) ---
        total_days = len(self.df_raw)
        train_idx = int(total_days * 0.60)
        self.cutoff_date = self.df_raw.iloc[train_idx]['Date']

        print("-" * 50)
        print(f"Całkowity zakres danych: {self.df_raw['Date'].min().date()} do {self.df_raw['Date'].max().date()}")
        print(f"Zablokowano optymalizator do daty (60%): {self.cutoff_date.date()}")
        print("-" * 50)

    def _apply_hysteresis(self, targets):
        smoothed = np.zeros_like(targets)
        valid_targets = targets[~np.isnan(targets)]
        if len(valid_targets) == 0: return targets

        current_state = valid_targets[0]
        count = 0

        for i in range(len(targets)):
            if np.isnan(targets[i]):
                smoothed[i] = np.nan
                continue

            if targets[i] == current_state:
                count = 0
            else:
                count += 1
                if count >= self.hysteresis_k:
                    current_state = targets[i]
                    count = 0
            smoothed[i] = current_state

        return smoothed

    def generate_targets(self, m_lower, m_upper):
        df = self.df_raw.copy()

        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['BTC_Close'].pct_change()
        if 'Vol_28d' not in df.columns:
            df['Vol_28d'] = df['Daily_Return'].rolling(28 * 3).std() * np.sqrt(28)

        df['Lower_Barrier'] = -(m_lower * df['Vol_28d'])
        df['Lower_Barrier'] = df['Lower_Barrier'].clip(lower=-1.0, upper=-0.01)

        df['Upper_Barrier'] = (m_upper * df['Vol_28d'])
        df['Upper_Barrier'] = df['Upper_Barrier'].clip(lower=0.02, upper=1.0)

        closes = df['BTC_Close'].values
        lows = df['BTC_Low'].values
        lower_barriers = df['Lower_Barrier'].values
        upper_barriers = df['Upper_Barrier'].values
        n = len(df)
        raw_targets = np.full(n, np.nan)

        # Pętla generuje targety tylko do n - horizon.
        # Ostatnie "horizon" dni zostaje jako np.nan
        for i in range(n - self.horizon):
            window_lows = lows[i + 1: i + 1 + self.horizon]
            window_closes = closes[i + 1: i + 1 + self.horizon]
            base_price = closes[i]

            low_returns = (window_lows / base_price) - 1
            close_returns = (window_closes / base_price) - 1

            hit_lower = low_returns < lower_barriers[i]
            hit_upper = close_returns > upper_barriers[i]

            idx_lower = np.argmax(hit_lower) if np.any(hit_lower) else self.horizon + 1
            idx_upper = np.argmax(hit_upper) if np.any(hit_upper) else self.horizon + 1

            if idx_upper < idx_lower and idx_upper != self.horizon + 1:
                raw_targets[i] = 1
            elif idx_lower < idx_upper and idx_lower != self.horizon + 1:
                raw_targets[i] = 0
            elif idx_lower == idx_upper and idx_lower != self.horizon + 1:
                raw_targets[i] = 0
            else:
                raw_targets[i] = 1 if close_returns[-1] > -0.02 else 0

        df['Target'] = self._apply_hysteresis(raw_targets)

        # --- ZMIANA: Nie usuwamy NaN, zachowujemy strukturę! ---
        df_targets = df[['Date', 'Target']]

        # Złączenie 'left' gwarantuje, że nie zniknie ani jeden dzień z bazy df_features
        df_final = pd.merge(self.df_features, df_targets, on='Date', how='left')
        # --------------------------------------------------------

        return df_final

    def evaluate_learnability(self, df_labeled):
        df_eval = df_labeled.dropna(subset=['Target']).copy()

        drop_cols = ['Date', 'Target']
        features = [c for c in df_eval.columns if c not in drop_cols]

        X = df_eval[features]
        y = df_eval['Target']

        dates_full = df_eval['Date']

        temp_df = pd.merge(dates_full, self.df_raw[['Date', 'BTC_Close']], on='Date', how='left')
        closes_full = temp_df['BTC_Close']

        tscv = TimeSeriesSplit(n_splits=5, gap=self.horizon)
        f1_macro_scores = []
        mcc_scores = []  # Nowa, bardzo restrykcyjna metryka
        strategy_sortinos = []

        clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1,
                                 verbose=-1)

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # --- POPRAWKI METRYK ---
            # Liczymy F1 Macro (średnia z F1 dla spadków i wzrostów)
            f1_macro_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            # Liczymy MCC
            mcc_scores.append(matthews_corrcoef(y_test, y_pred))

            signals_to_sim = y_pred
            dates_test = dates_full.iloc[test_index].values
            prices_test = closes_full.iloc[test_index].values

            sim_results = self.simulator.simulate(dates_test, prices_test, signals_to_sim)
            strategy_sortinos.append(sim_results['Sortino'])

        # Zwracamy MCC jako główną metrykę zdolności nauki
        return np.mean(f1_macro_scores), np.mean(mcc_scores), np.mean(strategy_sortinos), y.mean()

    def run_grid_search(self, m_lower_grid, m_upper_grid):
        results = []
        valid_combinations = [(ml, mu) for ml in m_lower_grid for mu in m_upper_grid if ml < mu]
        total = len(valid_combinations)
        i = 1

        print(f"\nRozpoczynam rygorystyczny DCA Grid Search dla {total} kombinacji...")

        for ml, mu in valid_combinations:
            # 1. Generujemy targety dla CAŁOŚCI (żeby utrzymać ciągłość wskaźników)
            df_labeled_full = self.generate_targets(ml, mu)

            # 2. KLUCZOWA ZMIANA: Odcinamy wszystko, co jest po dacie treningowej
            df_labeled_train = df_labeled_full[df_labeled_full['Date'] <= self.cutoff_date].copy()

            # Dodano .dropna() w len() bo bez sensu byłoby trenować, jeśli rynek nie ma historii
            if len(df_labeled_train.dropna(subset=['Target'])) < 1000:
                continue

            # 3. Ewaluacja modelu ML i DCA zachodzi WYŁĄCZNIE na 60% historii!
            f1_macro, mcc, sortino, class_balance = self.evaluate_learnability(df_labeled_train)

            results.append({
                'M_Lower': ml,
                'M_Upper': mu,
                'Rozkład_Jedynek (%)': round(class_balance * 100, 1),
                'F1_Macro': round(f1_macro, 3),
                'MCC': round(mcc, 3),  # Dodajemy MCC
                'Wynik_DCA_(Sortino)': round(sortino, 3)
            })
            print(
                f"[{i}/{total}] M_Low: {ml} | M_Up: {mu} -> Balans: {class_balance * 100:.1f}% | F1 Macro: {f1_macro:.3f} | MCC: {mcc:.3f} | Sortino: {sortino:.2f}")
            i += 1

        df_results = pd.DataFrame(results)

        if df_results.empty:
            return df_results

        print("\nPrzesiewam wyniki przez filtry bezpieczeństwa...")


        warunek_f1 = df_results['F1_Macro'] > 0.60
        warunek_mcc = df_results['MCC'] > 0.2
        warunek_balans_min = df_results['Rozkład_Jedynek (%)'] >= 35.0
        warunek_balans_max = df_results['Rozkład_Jedynek (%)'] <= 65.0
        warunek_sortino = df_results['Wynik_DCA_(Sortino)'] > 1

        # Aplikujemy wszystkie 5 warunków
        df_filtered = df_results[warunek_f1 & warunek_mcc & warunek_balans_min & warunek_balans_max & warunek_sortino]

        return df_filtered.sort_values('Wynik_DCA_(Sortino)', ascending=False)

    def plot_regimes(self, df_to_plot, m_lower, m_upper):
        df_plot = pd.merge(df_to_plot[['Date', 'Target']], self.df_raw[['Date', 'BTC_Close']], on='Date', how='left')

        # Wykres i tak zignoruje ostatnie dni, jeśli nie mają Targetu (to zamierzone dla wyświetlania)
        df_plot = df_plot.dropna(subset=['Target', 'BTC_Close']).copy()

        if not df_plot.empty:
            max_date = df_plot['Date'].max()
            five_years_ago = max_date - pd.DateOffset(years=5)
            df_plot = df_plot[df_plot['Date'] >= five_years_ago]

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(df_plot['Date'], df_plot['BTC_Close'], color='black', linewidth=1.5, label='Kurs BTC')
        ax.fill_between(df_plot['Date'], 0, 1, where=(df_plot['Target'] == 1),
                        color='green', alpha=0.25, transform=ax.get_xaxis_transform(), label='Target = 1 (Kup/Trzymaj)')
        ax.fill_between(df_plot['Date'], 0, 1, where=(df_plot['Target'] == 0),
                        color='red', alpha=0.25, transform=ax.get_xaxis_transform(), label='Target = 0 (Poza rynkiem)')

        ax.set_title(
            f'BTC Reżimy (Optymalne: M_Low={m_lower}, M_Up={m_upper}) | Horyzont={self.horizon}, Histereza={self.hysteresis_k} | Ost. 5 lat',
            fontsize=14, fontweight='bold')
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Cena Zamknięcia BTC', fontsize=12)
        ax.set_ylim(df_plot['BTC_Close'].min() * 0.95, df_plot['BTC_Close'].max() * 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()


# ==============================
# EGZEKUCJA
# ==============================
if __name__ == "__main__":
    optimizer = BTCProxyOptimizer(
        raw_data_path='btc_raw_data.csv',
        features_path='btc_ml_features_cleaned.csv',
        horizon=28,
        hysteresis_k=7
    )

    m_lower_test = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    m_upper_test = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    wyniki = optimizer.run_grid_search(m_lower_test, m_upper_test)

    if not wyniki.empty:
        print("\n--- ZWYCIĘSKIE PARAMETRY (DCA SYMULATOR) ---")
        print(wyniki)

        best_row = wyniki.iloc[0]
        best_ml = best_row['M_Lower']
        best_mu = best_row['M_Upper']
        best_sortino = best_row['Wynik_DCA_(Sortino)']

        df_final = optimizer.generate_targets(m_lower=best_ml, m_upper=best_mu)
        optimizer.plot_regimes(df_final, m_lower=best_ml, m_upper=best_mu)

        # ZAPIS DO PLIKU
        df_final.to_csv(f'btc_ml_features_final.csv', index=False)
        print(f"Zapisano ostateczny plik! Ostatnie 28 wierszy posiada poprawnie NaN w kolumnie Target.")
    else:
        print("\n[!] Żadna z testowanych kombinacji nie spełniła naraz wymogów F1 > 0.6 i Balansu.")