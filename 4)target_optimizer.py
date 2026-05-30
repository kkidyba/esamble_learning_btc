import pandas as pd
import numpy as np


class CryptoTargetOptimizer:
    # ==========================================
    # Inicjalizacja i Ładowanie Danych
    # ==========================================
    def __init__(self, features_path: str, raw_data_path: str):
        self.df_feat = pd.read_csv(features_path)
        if 'Unnamed: 0' in self.df_feat.columns:
            self.df_feat.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        self.df_feat['Date'] = pd.to_datetime(self.df_feat['Date'])
        self.df_feat.set_index('Date', inplace=True)
        self.df_feat.sort_index(inplace=True)

        self.df_raw = pd.read_csv(raw_data_path)
        if 'Unnamed: 0' in self.df_raw.columns:
            self.df_raw.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'])
        self.df_raw.set_index('Date', inplace=True)
        self.df_raw.sort_index(inplace=True)

        self.close_full = self.df_raw['BTC_Close']
        self.daily_returns_aligned = (self.close_full / self.close_full.shift(1) - 1.0).reindex(self.df_feat.index)

        print(f"Baza surowa: {len(self.df_raw)} dni. Docelowy zbiór ML: {len(self.df_feat)} wierszy.")

    # ==========================================
    # Kalkulacje Bazowe
    # ==========================================
    def _prepare_base_metrics(self):
        self.future_return_28d_full = (self.close_full.shift(-28) / self.close_full) - 1.0

        past_return_28d_full = (self.close_full / self.close_full.shift(28)) - 1.0
        self.rolling_std_28d_full = past_return_28d_full.rolling(window=365).std()

    # ==========================================
    # Generowanie Targetu
    # ==========================================
    def _generate_target_for_m(self, M: float) -> pd.Series:
        upper_threshold = self.rolling_std_28d_full * M
        lower_threshold = -self.rolling_std_28d_full * (M / 2.0)

        conditions = [
            self.future_return_28d_full.isna(),
            self.future_return_28d_full > upper_threshold,
            self.future_return_28d_full < lower_threshold
        ]
        choices = [np.nan, 1.0, -1.0]

        targets_full = np.select(conditions, choices, default=0.0)
        targets_series_full = pd.Series(targets_full, index=self.close_full.index)

        return targets_series_full.reindex(self.df_feat.index)

    # ==========================================
    # Moduł Backtestingu (Informacyjnie)
    # ==========================================
    def _backtest(self, targets: pd.Series, fee_pct: float = 0.0015) -> tuple:
        signal_map = targets.map({1.0: 1.0, -1.0: 0.0, 0.0: np.nan})
        positions = signal_map.ffill().fillna(0.0)
        pos_shifted = positions.shift(1).fillna(0.0)

        port_returns = pos_shifted * self.daily_returns_aligned

        trades = positions.diff().abs().fillna(0.0)
        net_returns = port_returns - (trades * fee_pct)

        total_days = len(net_returns.dropna())
        if total_days == 0:
            return 0.0, 0.0

        cumulative_return = (1 + net_returns).prod()

        return cumulative_return

    # ==========================================
    # Główny Moduł Optymalizacji (Balans Klas)
    # ==========================================
    def run_optimization(self, m_start=0.1, m_end=2.0, m_step=0.1, output_file="btc_ml_labeled_final.csv"):
        self._prepare_base_metrics()

        best_M = None
        best_imbalance = np.inf
        best_targets = None

        print("\nRozpoczynam Optymalizację Targetu pod kątem BALANSU KLAS...")
        print(
            f"{'M':<5} | {'KUP (1.0)':<10} | {'SPRZEDAJ (-1)':<13} | {'CZEKAJ (0.0)':<12} | {'Rozstrzał (Std)':<15}")
        print("-" * 80)

        m_space = np.arange(m_start, m_end + (m_step / 2), m_step)

        for m_val in m_space:
            targets = self._generate_target_for_m(m_val)

            counts = targets.value_counts(dropna=True)
            c_buy = counts.get(1.0, 0)
            c_sell = counts.get(-1.0, 0)
            c_wait = counts.get(0.0, 0)

            imbalance_score = np.std([c_buy, c_sell, c_wait])


            print(
                f"{m_val:<5.1f} | {c_buy:<10} | {c_sell:<13} | {c_wait:<12} | {imbalance_score:<15.1f}")

            if imbalance_score < best_imbalance:
                best_imbalance = imbalance_score
                best_M = m_val
                best_targets = targets

        print("-" * 80)
        print(f"WYGRYWA MNOŻNIK: {best_M:.1f}")
        print(f"Najlepsze wyrównanie klas (Odchylenie: {best_imbalance:.1f})")

        df_final = self.df_feat.copy()

        if 'Target_28d' in df_final.columns:
            df_final.drop(columns=['Target_28d'], inplace=True)

        df_final['Target_28d'] = best_targets

        df_final.to_csv(output_file)
        print(f"\nSukces! Zapisano gotowy do ML plik: {output_file}")

        counts = best_targets.value_counts(dropna=False)
        print(f"\nOstateczny rozkład docelowych klas dla M={best_M:.1f}:")
        print(f" 0.0 (CZEKAJ):   {counts.get(0.0, 0)}")
        print(f" 1.0 (KUP):      {counts.get(1.0, 0)}")
        print(f"-1.0 (SPRZEDAJ): {counts.get(-1.0, 0)}")
        print(f" NaN (Przyszłość):{counts.get(np.nan, 0)}")


if __name__ == "__main__":
    optimizer = CryptoTargetOptimizer("btc_ml_features_cleaned.csv", "btc_raw_data.csv")
    optimizer.run_optimization(m_start=0.1, m_end=1.0, m_step=0.05)