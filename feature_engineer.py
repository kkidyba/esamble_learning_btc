import pandas as pd
import numpy as np


class BitcoinFeatureEngineer:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

        self.df = pd.read_csv(self.input_filepath, index_col=0, parse_dates=True)
        self.features_df = pd.DataFrame(index=self.df.index)

    def _calculate_rsi(self, series, window):
        """Metoda pomocnicza do precyzyjnego wyliczania RSI (J. W. Wilder)"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def build_price_derived_features(self):
        """KONSOLIDACJA: Wylicza wszystkie cechy techniczne oparte na cenie BTC"""
        print("-> Budowanie bloku cech cenowych BTC (Momentum, Grawitacja, Oscylatory, Zmienność)...")
        if 'BTC_Price' not in self.df.columns:
            return self

        price = self.df['BTC_Price']

        # 1. Momentum
        for days in [7, 14, 28, 56]:
            self.features_df[f'BTC_Ret_{days}d'] = np.log(price / price.shift(days))

        # 2. Grawitacja
        sma_50 = price.rolling(window=50).mean()
        sma_100 = price.rolling(window=100).mean()
        sma_200 = price.rolling(window=200).mean()
        self.features_df['Dist_SMA_50'] = (price / sma_50) - 1
        self.features_df['Dist_SMA_100'] = (price / sma_100) - 1
        self.features_df['Dist_SMA_200'] = (price / sma_200) - 1
        self.features_df['Spread_50_200'] = (sma_50 / sma_200) - 1

        # 3. Oscylatory
        self.features_df['RSI_14'] = self._calculate_rsi(price, 14)
        self.features_df['RSI_28'] = self._calculate_rsi(price, 28)
        for days in [7, 14, 28, 56]:
            self.features_df[f'Dist_to_{days}d_High'] = (price / price.rolling(window=days).max()) - 1
            self.features_df[f'Dist_to_{days}d_Low'] = (price / price.rolling(window=days).min()) - 1

        # 4. Zmienność
        daily_ret = np.log(price / price.shift(1))
        for days in [14, 28, 56]:
            self.features_df[f'Volatility_{days}d'] = daily_ret.rolling(window=days).std() * np.sqrt(365)

        sma_28 = price.rolling(window=28).mean()
        std_28 = price.rolling(window=28).std()
        upper_band = sma_28 + (2 * std_28)
        lower_band = sma_28 - (2 * std_28)
        self.features_df['BB_Width_28d'] = (upper_band - lower_band) / sma_28
        bb_spread = upper_band - lower_band
        self.features_df['BB_Percent_28d'] = np.where(bb_spread == 0, 0, (price - lower_band) / bb_spread)

        return self

    def build_macro_features(self):
        """KONSOLIDACJA: Wylicza cechy makroekonomiczne (DXY, NASDAQ, M2)"""
        print("-> Budowanie bloku cech makroekonomicznych...")

        # 1. DXY
        if 'DXY_Index' in self.df.columns:
            dxy = self.df['DXY_Index']
            for days in [14, 28, 56]:
                self.features_df[f'DXY_Ret_{days}d'] = np.log(dxy / dxy.shift(days))
            sma_200_dxy = dxy.rolling(window=200).mean()
            self.features_df['DXY_Dist_SMA_200'] = (dxy / sma_200_dxy) - 1

        # 2. NASDAQ
        if 'NASDAQ_100' in self.df.columns and 'BTC_Price' in self.df.columns:
            nasdaq = self.df['NASDAQ_100']
            for days in [14, 28, 56]:
                self.features_df[f'NASDAQ_Ret_{days}d'] = np.log(nasdaq / nasdaq.shift(days))

            daily_ret_btc = np.log(self.df['BTC_Price'] / self.df['BTC_Price'].shift(1))
            daily_ret_nasdaq = np.log(nasdaq / nasdaq.shift(1))
            self.features_df['Corr_BTC_NASDAQ_28d'] = daily_ret_btc.rolling(window=28).corr(daily_ret_nasdaq)
            self.features_df['Corr_BTC_NASDAQ_56d'] = daily_ret_btc.rolling(window=56).corr(daily_ret_nasdaq)

        # 3. M2 Supply
        if 'M2_Supply' in self.df.columns:
            m2 = self.df['M2_Supply']
            for days in [56, 168]:
                self.features_df[f'M2_Ret_{days}d'] = np.log(m2 / m2.shift(days))

        return self

    def build_sentiment_features(self):
        """KONSOLIDACJA: Przetwarza cechy behawioralne czystej psychologii (F&G, Google Trends)"""
        print("-> Budowanie bloku cech sentymentu rynkowego (Fear & Greed + Google Trends)...")

        # 1. FEAR & GREED INDEX
        if 'Fear_Greed_Index' in self.df.columns:
            fg = self.df['Fear_Greed_Index']
            self.features_df['Fear_Greed_Index'] = fg
            self.features_df['F&G_SMA_14d'] = fg.rolling(window=14).mean()
            self.features_df['F&G_Delta_14d'] = fg - fg.shift(14)
            self.features_df['F&G_Delta_28d'] = fg - fg.shift(28)

        # 2. GOOGLE TRENDS BTC
        if 'Google_Trends_BTC' in self.df.columns:
            gt = self.df['Google_Trends_BTC']
            self.features_df['Google_Trends_BTC'] = gt
            self.features_df['Google_Trends_SMA_14d'] = gt.rolling(window=14).mean()
            self.features_df['Google_Trends_Delta_14d'] = gt - gt.shift(14)
            self.features_df['Google_Trends_Delta_28d'] = gt - gt.shift(28)

        return self

    def build_derivatives_features(self):
        """NOWA METODA: Przetwarza dane z rynku instrumentów pochodnych (Futures/Perpetuals)"""
        print("-> Budowanie bloku cech rynku derywatów (Funding Rates)...")

        if 'Funding_Rate_Last' in self.df.columns:
            funding = self.df['Funding_Rate_Last']

            # 1. Surowy Funding Rate (Zachowany jako kotwica mikrostruktury)
            self.features_df['Funding_Rate_Last'] = funding

            # 2. Szybki i głęboki koszt finansowania lewaryzacji (SMA 7d oraz SMA 28d)
            self.features_df['Funding_SMA_7d'] = funding.rolling(window=7).mean()
            self.features_df['Funding_SMA_28d'] = funding.rolling(window=28).mean()

        return self

    def save_features(self, start_date='2018-02-01'):
        """Odcina bufor historyczny, czyści ewentualne braki i zapisuje plik"""
        print(f"-> Odcinanie bufora historycznego. Start modelu od: {start_date}")
        self.features_df = self.features_df[self.features_df.index >= start_date]
        self.features_df.dropna(inplace=True)

        print(f"-> Zapisywanie gotowej macierzy cech do pliku: {self.output_filepath}")
        self.features_df.to_csv(self.output_filepath)

        print("\n--- PODGLĄD METRYK RYNKU DERYWATÓW ---")
        cols_to_show = ['Funding_Rate_Last', 'Funding_SMA_7d', 'Funding_SMA_28d']
        if all(col in self.features_df.columns for col in cols_to_show):
            print(self.features_df[cols_to_show].head())

        print(f"\nRozmiar nowego zbioru: {self.features_df.shape[1]} kolumn, {self.features_df.shape[0]} wierszy.")
        return self.features_df

    def run_pipeline(self):
        print("Rozpoczynam Inżynierię Cech (Feature Engineering)...")
        (self.build_price_derived_features()
         .build_macro_features()
         .build_sentiment_features()
         .build_derivatives_features()  # <--- Wpięcie nowego dedykowanego modułu
         .save_features())
        return self.features_df


if __name__ == "__main__":
    engineer = BitcoinFeatureEngineer('btc_ensemble_features.csv', 'btc_ml_features.csv')
    ml_dataset = engineer.run_pipeline()