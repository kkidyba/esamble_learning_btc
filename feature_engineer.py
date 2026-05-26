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

    def _rma(self, series: pd.Series, window: int) -> pd.Series:
        """Średnia krocząca Wildera (Running Moving Average) używana m.in. w ADX i klasycznym ATR."""
        return series.ewm(alpha=1 / window, adjust=False).mean()

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Kalkulacja wskaźnika ADX (Average Directional Index)"""
        up_move = high.diff()
        down_move = -low.diff()

        # Kierunkowe ruchy cenowe (Directional Movement)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wygładzone komponenty
        atr = self._rma(tr, window)
        plus_di = 100 * (self._rma(plus_dm, window) / atr)
        minus_di = 100 * (self._rma(minus_dm, window) / atr)

        # Directional Index (DX) i wygładzenie (ADX)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = self._rma(dx, window)

        return adx

    # ==========================================
    # Moduły Budujące Cechy (Kategoria A)
    # ==========================================
    def _build_cat_a_momentum(self):
        close = self.df['BTC_Close']
        high = self.df['BTC_High']  # Pobieramy High
        low = self.df['BTC_Low']  # Pobieramy Low

        self.df['LogRet_7d'] = self._log_return(close, 7)
        self.df['LogRet_14d'] = self._log_return(close, 14)
        self.df['LogRet_28d'] = self._log_return(close, 28)
        self.df['LogRet_56d'] = self._log_return(close, 56)
        self.df['LogRet_112d'] = self._log_return(close, 112)

        self.df['ADX_14'] = self._adx(high, low, close, window=14)
        self.df['ADX_28'] = self._adx(high, low, close, window=28)
        self.df['ADX_56'] = self._adx(high, low, close, window=56)
        self.df['ADX_112'] = self._adx(high, low, close, window=112)

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
        self.df['Volatility_LogRet_56d'] = log_ret_1d.rolling(window=56).std()

        atr_14 = self._atr(high, low, close, window=14)
        self.df['NATR_14'] = (atr_14 / close) * 100.0

        sma_20 = self._sma(close, window=20)
        std_20 = close.rolling(window=20).std(ddof=0)
        self.df['BB_Width_20'] = (4.0 * std_20) / sma_20

        sma_56 = self._sma(close, window=56)
        std_56 = close.rolling(window=56).std(ddof=0)
        self.df['BB_Width_56'] = (4.0 * std_56) / sma_56

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

        mfm = ((close - low) - (high - close)) / ((high - low) + epsilon)
        mfv = mfm * volume
        self.df['CMF_56d'] = mfv.rolling(window=56).sum() / volume.rolling(window=56).sum()

        typical_price = (high + low + close) / 3.0
        typical_price_volume = typical_price * volume
        rolling_vwap_28d = typical_price_volume.rolling(window=28).sum() / volume.rolling(window=28).sum()
        self.df['VWAP_Distance_28d'] = (close / rolling_vwap_28d) - 1.0

        sma_7_vol = self._sma(volume, window=7)
        sma_28_vol = self._sma(volume, window=28)
        self.df['Volume_Oscillator_7_28'] = (sma_7_vol / sma_28_vol) - 1.0

        sma_14_vol = self._sma(volume, window=14)
        sma_56_vol = self._sma(volume, window=56)
        self.df['Volume_Oscillator_14_56'] = (sma_14_vol / sma_56_vol) - 1.0

        direction = np.sign(close.diff()).fillna(0)
        raw_obv = (direction * volume).cumsum()
        obv_mean_56d = raw_obv.rolling(window=56).mean()
        obv_std_56d = raw_obv.rolling(window=56).std(ddof=0)
        self.df['OBV_ZScore_56d'] = (raw_obv - obv_mean_56d) / obv_std_56d

    def _build_cat_a_calendar(self):
        """Kategoria A - Sub-domena 6: Kalendarz i Sezonowość"""
        # dayofweek: 0 = Poniedziałek, 6 = Niedziela
        self.df['Day_of_Week'] = self.df.index.dayofweek

        self.df['Day_of_Month'] = self.df.index.day

        self.df['Month'] = self.df.index.month
        # quarter: 1 do 4
        self.df['Quarter'] = self.df.index.quarter

    # ==========================================
    # Moduły Budujące Cechy (Kategoria B)
    # ==========================================
    def _build_cat_b_macro_flows(self):
        ndx = self.df['NASDAQ_100']
        dxy = self.df['DXY_Index']
        gold = self.df['Gold_Close']

        self.df['NDX_LogRet_28d'] = self._log_return(ndx, window=28)
        self.df['NDX_LogRet_56d'] = self._log_return(ndx, window=56)
        self.df['DXY_LogRet_28d'] = self._log_return(dxy, window=28)
        self.df['DXY_LogRet_56d'] = self._log_return(dxy, window=56)
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
        core_cpi = self.df['Core_CPI']
        tnx = self.df['TNX']
        vix = self.df['VIX_Index']

        self.df['CPI_Growth_YoY'] = (core_cpi / core_cpi.shift(365)) - 1.0
        self.df['Real_Rates_Proxy'] = tnx - (self.df['CPI_Growth_YoY'] * 100.0)

        sma_28_vix = self._sma(vix, window=28)
        self.df['VIX_Relative_28d'] = (vix / sma_28_vix) - 1.0

    # ==========================================
    # Moduły Budujące Cechy (Kategoria C)
    # ==========================================
    def _build_cat_c_mining(self):
        miners_rev = self.df['Miners_Revenue_USD']
        hashrate = self.df['Hashrate']
        difficulty = self.df['Difficulty']

        self.df['Puell_Multiple_Proxy'] = miners_rev / self._sma(miners_rev, window=365)

        sma_30_hash = self._sma(hashrate, window=30)
        sma_60_hash = self._sma(hashrate, window=60)
        self.df['Hash_Ribbon_Momentum'] = (sma_30_hash / sma_60_hash) - 1.0

        self.df['Difficulty_Growth_28d'] = (difficulty / difficulty.shift(28)) - 1.0
        self.df['Difficulty_Growth_56d'] = (difficulty / difficulty.shift(56)) - 1.0

    def _build_cat_c_valuation(self):
        supply = self.df['Circulating_Supply']
        tx_vol_btc = self.df['Est_Tx_Volume_BTC']
        unique_addr = self.df['Unique_Addresses']
        tx_count = self.df['Tx_Count']

        epsilon = 1e-8
        raw_nvt = supply / (tx_vol_btc + epsilon)
        sma_90_nvt = self._sma(raw_nvt, window=90)
        self.df['NVT_Signal_90d'] = raw_nvt / sma_90_nvt

        sma_7_addr = self._sma(unique_addr, window=7)
        sma_28_addr = self._sma(unique_addr, window=28)
        self.df['Network_Adoption_Momentum'] = (sma_7_addr / sma_28_addr) - 1.0

        sma_14_addr = self._sma(unique_addr, window=14)
        sma_56_addr = self._sma(unique_addr, window=56)
        self.df['Network_Adoption_Momentum_14_56'] = (sma_14_addr / sma_56_addr) - 1.0

        self.df['Tx_Intensity_Ratio'] = tx_count / (unique_addr + epsilon)

    def _build_cat_c_congestion(self):
        """Kategoria C - Sub-domena 3: Zatłoczenie i FOMO Ulicy"""
        fees_btc = self.df['Total_Fees_BTC']
        mempool_size = self.df['Mempool_Size_Bytes']

        fees_usd = fees_btc * self.df['BTC_Close']
        miners_rev_usd = self.df['Miners_Revenue_USD']

        epsilon = 1e-8

        sma_56_fees = self._sma(fees_btc, window=56)
        self.df['Fee_Spike_Indicator'] = (fees_btc / (sma_56_fees + epsilon)) - 1.0

        mempool_mean_28d = mempool_size.rolling(window=28).mean()
        mempool_std_28d = mempool_size.rolling(window=28).std(ddof=0)

        # POPRAWKA 4: Zabezpieczenie odchylenia standardowego
        self.df['Mempool_Congestion_ZScore_28d'] = (mempool_size - mempool_mean_28d) / (mempool_std_28d + epsilon)

        self.df['Fee_to_Reward_Ratio'] = fees_usd / (miners_rev_usd + epsilon)

    def _build_cat_c_supply(self):
        total_blocks = self.df['Total_Blocks']
        utxo_count = self.df['UTXO_Count']

        self.df['Halving_Cycle_Position'] = (total_blocks % 210000) / 210000.0

        self.df['UTXO_Distribution_56d'] = (utxo_count / utxo_count.shift(56)) - 1.0

    def _build_cat_d_derivatives(self):
        """Kategoria D - Sub-domena 1: Instrumenty Pochodne i Dźwignia"""
        funding_rate = self.df['BitMEX_Funding_Rate_Max']
        open_interest = self.df['Bybit_Open_Interest']
        close_btc = self.df['BTC_Close']
        supply = self.df['Circulating_Supply']
        lsr = self.df['Bybit_Long_Short_Ratio']  # <- Pobranie danych LSR

        epsilon = 1e-8

        funding_mean_28d = funding_rate.rolling(window=28).mean()
        funding_std_28d = funding_rate.rolling(window=28).std(ddof=0)
        # POPRAWKA 1: Dodanie epsilon do mianownika w Z-Score
        self.df['Funding_Rate_ZScore_28d'] = (funding_rate - funding_mean_28d) / (funding_std_28d + epsilon)

        # POPRAWKA 2: Bezpieczne momentum za pomocą np.where.
        # Jeśli 14 dni temu wartość wynosiła 0, momentum wynosi 0 (zapobiega to wynikom NaN oraz inf)
        self.df['OI_Momentum_14d'] = np.where(
            open_interest.shift(14) == 0,
            0,
            (open_interest / open_interest.shift(14)) - 1.0
        )

        self.df['OI_Momentum_56d'] = np.where(
            open_interest.shift(56) == 0,
            0,
            (open_interest / open_interest.shift(56)) - 1.0
        )

        market_cap = close_btc * supply
        self.df['Open_Interest_Leverage_Ratio'] = open_interest / (market_cap + epsilon)
        self.df['Retail_Positioning_LSR'] = self._sma(lsr, window=7)

    def _build_cat_d_defi(self):
        """Kategoria D - Sub-domena 2: Przepływy Kapitałowe DeFi i Web3"""
        stablecoin_mcap = self.df['Stablecoin_Total_MCap']
        defi_tvl = self.df['DeFi_Global_TVL']
        close_btc = self.df['BTC_Close']
        supply = self.df['Circulating_Supply']
        hacks_loss = self.df['DeFi_Daily_Hacks_Loss_USD']
        defi_fees = self.df['DeFi_Global_Daily_Fees']

        # Nowe zmienne dla wskaźnika DEX/CEX
        dex_vol = self.df['DEX_Daily_Volume']
        cex_vol = self.df['Exchange_Trade_Volume_USD']

        epsilon = 1e-8

        self.df['Stablecoin_Growth_28d'] = np.where(
            stablecoin_mcap.shift(28) == 0,
            0,
            (stablecoin_mcap / stablecoin_mcap.shift(28)) - 1.0
        )

        self.df['Stablecoin_Growth_56d'] = np.where(
            stablecoin_mcap.shift(56) == 0,
            0,
            (stablecoin_mcap / stablecoin_mcap.shift(56)) - 1.0
        )


        # 50) DeFi_Risk_Appetite - Wskaźnik kapitału uciekającego w ryzykowne altcoiny
        market_cap = close_btc * supply
        self.df['DeFi_Risk_Appetite'] = defi_tvl / (market_cap + epsilon)

        # 51) DeFi_Fear_Index - Wygładzony stosunek strat z kradzieży do organicznych przychodów
        sma_14_hacks = self._sma(hacks_loss, window=14)
        sma_14_fees = self._sma(defi_fees, window=14)
        self.df['DeFi_Fear_Index'] = sma_14_hacks / (sma_14_fees + epsilon)

        # 52) DEX_to_EXCHANGE_Volume_Ratio - Gdzie toczy się prawdziwa spekulacja
        self.df['DEX_to_EXCHANGE_Volume_Ratio'] = dex_vol / (cex_vol + epsilon)

    def _build_cat_d_sentiment(self):
        """Kategoria D - Sub-domena 3: Nastroje Mas i Psychologia Ekstremów"""
        fear_greed = self.df['Fear_Greed_Index']
        google_trends = self.df['Google_Trends_BTC']

        # 53) Fear_Greed_Trend_14d - Odfiltrowana z szumu bazowa linia trendu emocji na rynku
        self.df['Fear_Greed_Trend_14d'] = self._sma(fear_greed, window=14)

        # 54) Extreme_Sentiment_Duration_28d - Skumulowana siła i czas trwania emocji
        conditions = [
            fear_greed > 80,
            fear_greed < 20
        ]
        choices = [1, -1]
        sentiment_score = pd.Series(np.select(conditions, choices, default=0), index=fear_greed.index)
        self.df['Extreme_Sentiment_Duration_28d'] = sentiment_score.rolling(window=28).sum()

        # 55) Retail_FOMO_Spike - Anomalia zapytań w wyszukiwarce Google (momentum)
        sma_56_trends = self._sma(google_trends, window=56)
        self.df['Retail_FOMO_Spike'] = google_trends - sma_56_trends

        # 56) Google_Trends_ZScore_90d - Kwartalna, statystyczna anomalia zainteresowania ulicy
        trends_mean_90d = google_trends.rolling(window=90).mean()
        trends_std_90d = google_trends.rolling(window=90).std(ddof=0)
        self.df['Google_Trends_ZScore_90d'] = (google_trends - trends_mean_90d) / trends_std_90d





    # ==========================================
    # Kompilator Głównego Zbioru
    # ==========================================
    def generate_dataset(self, output_filename: str = "btc_ml_features.csv"):
        self._build_cat_a_momentum()
        self._build_cat_a_mean_reversion()
        self._build_cat_a_oscillators()
        self._build_cat_a_volatility()
        self._build_cat_a_macro_structure()
        self._build_cat_a_volume()

        self._build_cat_b_macro_flows()
        self._build_cat_b_intermarket()
        self._build_cat_b_liquidity()
        self._build_cat_b_inflation()

        self._build_cat_c_mining()
        self._build_cat_c_valuation()
        self._build_cat_c_congestion()
        self._build_cat_c_supply()
        self._build_cat_d_derivatives()
        self._build_cat_d_defi()
        self._build_cat_d_sentiment()
        self._build_cat_a_calendar()


        features = [
            # ==== KATEGORIA A ====
            'Day_of_Week',
            'Day_of_Month',
            'Month',
            'Quarter',
            'LogRet_7d',
            'LogRet_14d',
            'LogRet_28d',
            'LogRet_56d',
            'LogRet_112d',
            'ADX_14',
            'ADX_28',
            'ADX_56',
            'ADX_112',
            'Dist_SMA_50',
            'Dist_SMA_200',
            'Dist_EMA_100',
            'Cross_SMA50_SMA200',
            'RSI_14',
            'RSI_56',
            'PPO_Hist_12_26',
            'Position_in_Range_56d',
            'Volatility_LogRet_28d',
            'Volatility_LogRet_56d',
            'NATR_14',
            'BB_Width_20',
            'BB_Width_56',
            'Max_Spread_14d',
            'Body_Size_Ratio',
            'Historical_Drawdown_365d',
            'ATH_Distance_Log',
            'Price_Percentile_90d',
            'CMF_28d',
            'CMF_56d',
            'VWAP_Distance_28d',
            'Volume_Oscillator_7_28',
            'Volume_Oscillator_14_56',
            'OBV_ZScore_56d',

            # ==== KATEGORIA B ====
            'NDX_LogRet_28d',
            'NDX_LogRet_56d',
            'DXY_LogRet_28d',
            'DXY_LogRet_56d',
            'Gold_LogRet_56d',
            'Corr_BTC_NDX_56d',
            'Corr_BTC_DXY_56d',
            'Yield_Curve_Spread',
            'TNX_Diff_28d',
            'M2_Growth_90d',
            'CPI_Growth_YoY',
            'Real_Rates_Proxy',
            'VIX_Relative_28d',

            # ==== KATEGORIA C ====
            'Puell_Multiple_Proxy',
            'Hash_Ribbon_Momentum',
            'Difficulty_Growth_28d',
            'Difficulty_Growth_56d',
            'NVT_Signal_90d',
            'Network_Adoption_Momentum',
            'Network_Adoption_Momentum_14_56',
            'Tx_Intensity_Ratio',
            'Fee_Spike_Indicator',
            'Mempool_Congestion_ZScore_28d',
            'Fee_to_Reward_Ratio',
            'Halving_Cycle_Position',
            'UTXO_Distribution_56d',

            # ==== KATEGORIA D ====
            'Funding_Rate_ZScore_28d',
            'OI_Momentum_14d',
            'OI_Momentum_56d',
            'Open_Interest_Leverage_Ratio',
            'Retail_Positioning_LSR',
            'Stablecoin_Growth_28d',
            'Stablecoin_Growth_56d',
            'DeFi_Risk_Appetite',
            'DeFi_Fear_Index',
            'DEX_to_EXCHANGE_Volume_Ratio',
            'Fear_Greed_Trend_14d',
            'Extreme_Sentiment_Duration_28d',
            'Retail_FOMO_Spike',
            'Google_Trends_ZScore_90d',


        ]

        ml_df = self.df[features].dropna(subset='CPI_Growth_YoY')

        ml_df.to_csv(output_filename)
        print(f"Sukces! Zapisano plik {output_filename}.")
        print(f"Ilość wygenerowanych wierszy po 365 dniusuwanie na: {len(ml_df)}")

        return ml_df



if __name__ == "__main__":
    generator = CryptoFeatureGenerator("btc_raw_data.csv")
    df_features = generator.generate_dataset("btc_ml_features.csv")

    print("\nPróbka wygenerowanych cech Makro-Struktury Podaży:")
    print(df_features.tail())