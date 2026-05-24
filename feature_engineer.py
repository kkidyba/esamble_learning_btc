import pandas as pd
import numpy as np


class CryptoFeatureGenerator:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)

        if 'Unnamed: 0' in self.df.columns:
            self.df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(inplace=True)

        print("Dane załadowane pomyślnie. Kształt wejściowy:", self.df.shape)

    # ==========================================
    # Metody Pomocnicze (Helpers)
    # ==========================================
    def _log_return(self, series: pd.Series, window: int) -> pd.Series:
        return np.log(series / series.shift(window))

    def _sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    def _ema(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    def _rsi(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).fillna(0).values
        down = (-1 * delta.clip(upper=0)).fillna(0).values

        roll_up = np.zeros_like(up)
        roll_down = np.zeros_like(down)

        roll_up[window] = up[1:window + 1].mean()
        roll_down[window] = down[1:window + 1].mean()

        for i in range(window + 1, len(up)):
            roll_up[i] = (roll_up[i - 1] * (window - 1) + up[i]) / window
            roll_down[i] = (roll_down[i - 1] * (window - 1) + down[i]) / window

        rsi = np.zeros_like(up)
        for i in range(window, len(up)):
            if roll_down[i] == 0:
                rsi[i] = 100.0
            else:
                rs = roll_up[i] / roll_down[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        rsi_series = pd.Series(rsi, index=series.index)
        rsi_series.iloc[:window] = np.nan
        return rsi_series

    def _ppo_hist(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = self._ema(series, fast)
        ema_slow = self._ema(series, slow)
        ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100.0
        ppo_signal = self._ema(ppo_line, signal)
        return ppo_line - ppo_signal

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        atr = np.zeros_like(tr)
        atr[window] = np.nanmean(tr[1:window + 1])

        for i in range(window + 1, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

        atr_series = pd.Series(atr, index=close.index)
        atr_series.iloc[:window] = np.nan
        return atr_series

    # ==========================================
    # Moduły Budujące Cechy (Kategoria A)
    # ==========================================
    def _build_cat_a_momentum(self):
        close = self.df['BTC_Close']
        self.df['LogRet_7d'] = self._log_return(close, 7)
        self.df['LogRet_14d'] = self._log_return(close, 14)
        self.df['LogRet_28d'] = self._log_return(close, 28)
        self.df['LogRet_56d'] = self._log_return(close, 56)
        self.df['LogRet_112d'] = self._log_return(close, 112)

    def _build_cat_a_mean_reversion(self):
        close = self.df['BTC_Close']
        sma_50 = self._sma(close, 50)
        sma_200 = self._sma(close, 200)
        ema_100 = self._ema(close, 100)

        self.df['Dist_SMA_50'] = (close / sma_50) - 1
        self.df['Dist_SMA_200'] = (close / sma_200) - 1
        self.df['Dist_EMA_100'] = (close / ema_100) - 1
        self.df['Cross_SMA50_SMA200'] = (sma_50 / sma_200) - 1

    def _build_cat_a_oscillators(self):
        close = self.df['BTC_Close']
        self.df['RSI_14'] = self._rsi(close, 14)
        self.df['RSI_56'] = self._rsi(close, 56)
        self.df['PPO_Hist_12_26'] = self._ppo_hist(close, fast=12, slow=26, signal=9)

        min_56d = close.rolling(window=56).min()
        max_56d = close.rolling(window=56).max()
        self.df['Position_in_Range_56d'] = (close - min_56d) / (max_56d - min_56d)

    def _build_cat_a_volatility(self):
        open_p = self.df['BTC_Open']
        close = self.df['BTC_Close']
        high = self.df['BTC_High']
        low = self.df['BTC_Low']

        log_ret_1d = self._log_return(close, window=1)
        self.df['Volatility_LogRet_28d'] = log_ret_1d.rolling(window=28).std()

        atr_14 = self._atr(high, low, close, window=14)
        self.df['NATR_14'] = (atr_14 / close) * 100.0

        sma_20 = self._sma(close, window=20)
        std_20 = close.rolling(window=20).std(ddof=0)
        self.df['BB_Width_20'] = (4.0 * std_20) / sma_20

        daily_spread = (high - low) / close
        self.df['Max_Spread_14d'] = daily_spread.rolling(window=14).max()

        epsilon = 1e-8
        body_abs = (close - open_p).abs()
        high_low_range = high - low
        self.df['Body_Size_Ratio'] = body_abs / (high_low_range + epsilon)

    def _build_cat_a_macro_structure(self):
        close = self.df['BTC_Close']
        high = self.df['BTC_High']

        max_high_365d = high.rolling(window=365).max()
        self.df['Historical_Drawdown_365d'] = (close / max_high_365d) - 1.0

        dynamic_ath = high.expanding().max()
        self.df['ATH_Distance_Log'] = np.log(close / dynamic_ath)

        self.df['Price_Percentile_90d'] = close.rolling(window=90).rank(pct=True)

    def _build_cat_a_volume(self):
        close = self.df['BTC_Close']
        high = self.df['BTC_High']
        low = self.df['BTC_Low']
        volume = self.df['Exchange_Trade_Volume_USD']

        epsilon = 1e-8
        mfm = ((close - low) - (high - close)) / ((high - low) + epsilon)
        mfv = mfm * volume
        self.df['CMF_28d'] = mfv.rolling(window=28).sum() / volume.rolling(window=28).sum()

        typical_price = (high + low + close) / 3.0
        typical_price_volume = typical_price * volume
        rolling_vwap_28d = typical_price_volume.rolling(window=28).sum() / volume.rolling(window=28).sum()
        self.df['VWAP_Distance_28d'] = (close / rolling_vwap_28d) - 1.0

        sma_7_vol = self._sma(volume, window=7)
        sma_28_vol = self._sma(volume, window=28)
        self.df['Volume_Oscillator_7_28'] = (sma_7_vol / sma_28_vol) - 1.0

        direction = np.sign(close.diff()).fillna(0)
        raw_obv = (direction * volume).cumsum()
        obv_mean_56d = raw_obv.rolling(window=56).mean()
        obv_std_56d = raw_obv.rolling(window=56).std(ddof=0)
        self.df['OBV_ZScore_56d'] = (raw_obv - obv_mean_56d) / obv_std_56d

    # ==========================================
    # Moduły Budujące Cechy (Kategoria B)
    # ==========================================
    def _build_cat_b_macro_flows(self):
        ndx = self.df['NASDAQ_100']
        dxy = self.df['DXY_Index']
        gold = self.df['Gold_Close']

        self.df['NDX_LogRet_28d'] = self._log_return(ndx, window=28)
        self.df['DXY_LogRet_28d'] = self._log_return(dxy, window=28)
        self.df['Gold_LogRet_56d'] = self._log_return(gold, window=56)

    def _build_cat_b_intermarket(self):
        close_btc = self.df['BTC_Close']
        ndx = self.df['NASDAQ_100']
        dxy = self.df['DXY_Index']

        log_ret_btc_1d = self._log_return(close_btc, window=1)
        log_ret_ndx_1d = self._log_return(ndx, window=1)
        log_ret_dxy_1d = self._log_return(dxy, window=1)

        self.df['Corr_BTC_NDX_56d'] = log_ret_btc_1d.rolling(window=56).corr(log_ret_ndx_1d)
        self.df['Corr_BTC_DXY_56d'] = log_ret_btc_1d.rolling(window=56).corr(log_ret_dxy_1d)

    def _build_cat_b_liquidity(self):
        tnx = self.df['TNX']
        dff = self.df['DFF_Rate']
        m2 = self.df['M2_Supply']

        self.df['Yield_Curve_Spread'] = tnx - dff
        self.df['TNX_Diff_28d'] = tnx - tnx.shift(28)
        self.df['M2_Growth_90d'] = (m2 / m2.shift(90)) - 1.0

    def _build_cat_b_inflation(self):
        """Kategoria B - Sub-domena 4: Szoki Inflacyjne i Zmienność Systemowa"""
        core_cpi = self.df['Core_CPI']
        tnx = self.df['TNX']
        vix = self.df['VIX_Index']

        # 33) CPI_Growth_YoY
        self.df['CPI_Growth_YoY'] = (core_cpi / core_cpi.shift(365)) - 1.0

        # 34) Real_Rates_Proxy
        self.df['Real_Rates_Proxy'] = tnx - (self.df['CPI_Growth_YoY'] * 100.0)

        # 35) VIX_Relative_28d - Detektor "strachu i kapitulacji"
        sma_28_vix = self._sma(vix, window=28)
        self.df['VIX_Relative_28d'] = (vix / sma_28_vix) - 1.0

    # ==========================================
    # Kompilator Głównego Zbioru
    # ==========================================
    def generate_dataset(self, output_filename: str = "btc_ml_features_step21.csv"):
        # Kategoria A
        self._build_cat_a_momentum()
        self._build_cat_a_mean_reversion()
        self._build_cat_a_oscillators()
        self._build_cat_a_volatility()
        self._build_cat_a_macro_structure()
        self._build_cat_a_volume()

        # Kategoria B
        self._build_cat_b_macro_flows()
        self._build_cat_b_intermarket()
        self._build_cat_b_liquidity()
        self._build_cat_b_inflation()

        features = [
            # ==== KATEGORIA A ====
            'LogRet_7d', 'LogRet_14d', 'LogRet_28d', 'LogRet_56d', 'LogRet_112d',
            'Dist_SMA_50', 'Dist_SMA_200', 'Dist_EMA_100', 'Cross_SMA50_SMA200',
            'RSI_14', 'RSI_56', 'PPO_Hist_12_26', 'Position_in_Range_56d',
            'Volatility_LogRet_28d', 'NATR_14', 'BB_Width_20', 'Max_Spread_14d', 'Body_Size_Ratio',
            'Historical_Drawdown_365d', 'ATH_Distance_Log', 'Price_Percentile_90d',
            'CMF_28d', 'VWAP_Distance_28d', 'Volume_Oscillator_7_28', 'OBV_ZScore_56d',

            # ==== KATEGORIA B ====
            'NDX_LogRet_28d', 'DXY_LogRet_28d', 'Gold_LogRet_56d',
            'Corr_BTC_NDX_56d', 'Corr_BTC_DXY_56d',
            'Yield_Curve_Spread', 'TNX_Diff_28d', 'M2_Growth_90d',
            'CPI_Growth_YoY', 'Real_Rates_Proxy', 'VIX_Relative_28d'
        ]

        ml_df = self.df[features].dropna()

        ml_df.to_csv(output_filename)
        print(f"Sukces! Zapisano plik {output_filename}.")
        print(f"Ilość wygenerowanych wierszy po odrzuceniu NaN: {len(ml_df)}")

        return ml_df


if __name__ == "__main__":
    generator = CryptoFeatureGenerator("btc_raw_data.csv")
    df_features = generator.generate_dataset("btc_features_step21.csv")

    print("\nPróbka wygenerowanych cech makroekonomicznych (ostatnie 5 wierszy - w tym VIX_Relative):")
    print(df_features[['Real_Rates_Proxy', 'VIX_Relative_28d']].tail())