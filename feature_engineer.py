import pandas as pd
import numpy as np


class BitcoinFeatureEngineer:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

        # Wczytanie danych z obsługą indeksu czasowego
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
        print("-> Budowanie bloku cech cenowych BTC...")
        if 'BTC_Price' not in self.df.columns: return self
        price = self.df['BTC_Price']

        # Usunięto: BTC_Ret_14d
        for days in [7, 28, 56]:
            self.features_df[f'BTC_Ret_{days}d'] = np.log(price / price.shift(days))

        sma_50 = price.rolling(window=50).mean()
        sma_200 = price.rolling(window=200).mean()
        self.features_df['Dist_SMA_50'] = (price / sma_50) - 1
        # Usunięto: Dist_SMA_100
        self.features_df['Dist_SMA_200'] = (price / sma_200) - 1
        self.features_df['Spread_50_200'] = (sma_50 / sma_200) - 1

        self.features_df['RSI_14'] = self._calculate_rsi(price, 14)

        # Usunięto: Dist_to_7d_High, Dist_to_14d_High, Dist_to_7d_Low, Dist_to_14d_Low
        for days in [28, 56]:
            self.features_df[f'Dist_to_{days}d_High'] = (price / price.rolling(window=days).max()) - 1
            self.features_df[f'Dist_to_{days}d_Low'] = (price / price.rolling(window=days).min()) - 1

        daily_ret = np.log(price / price.shift(1))
        for days in [14, 56]:
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
        print("-> Budowanie bloku cech makroekonomicznych...")
        if 'DXY_Index' in self.df.columns:
            dxy = self.df['DXY_Index']
            for days in [28, 56]:
                self.features_df[f'DXY_Ret_{days}d'] = np.log(dxy / dxy.shift(days))
            self.features_df['DXY_Dist_SMA_200'] = (dxy / dxy.rolling(window=200).mean()) - 1

        if 'NASDAQ_100' in self.df.columns and 'BTC_Price' in self.df.columns:
            nasdaq = self.df['NASDAQ_100']
            for days in [28, 56]:
                self.features_df[f'NASDAQ_Ret_{days}d'] = np.log(nasdaq / nasdaq.shift(days))
            daily_ret_btc = np.log(self.df['BTC_Price'] / self.df['BTC_Price'].shift(1))
            daily_ret_nasdaq = np.log(nasdaq / nasdaq.shift(1))
            self.features_df['Corr_BTC_NASDAQ_56d'] = daily_ret_btc.rolling(window=56).corr(daily_ret_nasdaq)

        if 'M2_Supply' in self.df.columns:
            m2 = self.df['M2_Supply']
            for days in [56, 168]:
                self.features_df[f'M2_Ret_{days}d'] = np.log(m2 / m2.shift(days))
        return self

    def build_sentiment_features(self):
        print("-> Budowanie bloku cech sentymentu rynkowego...")
        if 'Fear_Greed_Index' in self.df.columns:
            fg = self.df['Fear_Greed_Index']
            self.features_df['Fear_Greed_Index'] = fg
            self.features_df['F&G_Delta_14d'] = fg - fg.shift(14)

        if 'Google_Trends_BTC' in self.df.columns:
            gt = self.df['Google_Trends_BTC']
            self.features_df['Google_Trends_BTC'] = gt
            self.features_df['Google_Trends_Delta_14d'] = gt - gt.shift(14)
        return self

    def build_derivatives_features(self):
        print("-> Budowanie bloku cech rynku derywatów...")
        if 'Funding_Rate_Last' in self.df.columns:
            funding = self.df['Funding_Rate_Last']
            self.features_df['Funding_Rate_Last'] = funding
            # Usunięto: Funding_SMA_7d
            self.features_df['Funding_SMA_28d'] = funding.rolling(window=28).mean()
        return self

    def build_volume_features(self):
        print("-> Budowanie bloku cech wolumenowych i interakcji...")
        if 'BTC_Volume' in self.df.columns and 'BTC_Price' in self.df.columns:
            volume = self.df['BTC_Volume']
            price = self.df['BTC_Price']
            eps = 1e-8

            self.features_df['Vol_Ret_7d'] = np.log((volume + eps) / (volume.shift(7) + eps))
            self.features_df['Vol_Ret_28d'] = np.log((volume + eps) / (volume.shift(28) + eps))

            sma_14_vol = volume.rolling(window=14).mean()
            # Usunięto: Vol_Ratio_to_SMA_28d oraz sma_28_vol
            self.features_df['Vol_Ratio_to_SMA_14d'] = np.where(sma_14_vol == 0, 0, (volume / sma_14_vol) - 1)

            daily_ret_btc = np.log(price / price.shift(1))
            self.features_df['Signed_Vol_Ratio'] = self.features_df['Vol_Ratio_to_SMA_14d'] * np.sign(daily_ret_btc)

            if 'BB_Percent_28d' in self.features_df.columns:
                self.features_df['BB_Vol_Pressure'] = self.features_df['Vol_Ratio_to_SMA_14d'] * (
                            self.features_df['BB_Percent_28d'] - 0.5)
            if 'Volatility_14d' in self.features_df.columns:
                self.features_df['Vol_Climax_Index'] = self.features_df['Vol_Ratio_to_SMA_14d'] * self.features_df[
                    'Volatility_14d']
        return self

    def build_onchain_features(self):
        """KONSOLIDACJA: Przetwarza dane on-chain (Górnicy, Adopcja, Opłaty Sieciowe)"""
        print("-> Budowanie bloku cech on-chain (Górnicy, Adresy, Opłaty)...")
        eps = 1e-8  # Zabezpieczenie przed dzieleniem przez zero

        # --- 1. GÓRNICY (Hashrate & Difficulty) ---
        if 'Hashrate' in self.df.columns and 'Difficulty' in self.df.columns:
            hashrate = self.df['Hashrate']
            difficulty = self.df['Difficulty']

            sma_30_hash = hashrate.rolling(window=30).mean()
            sma_60_hash = hashrate.rolling(window=60).mean()
            self.features_df['Hash_Compression'] = np.where(sma_60_hash == 0, 0, (sma_30_hash / sma_60_hash) - 1)

            ratio = hashrate / difficulty
            self.features_df['Hash_Diff_Ratio'] = ratio.rolling(window=7).mean()

        # --- 2. ADOPCJA I AKTYWNOŚĆ (Unique Addresses & Transactions) ---
        if 'Unique_Addresses' in self.df.columns and 'Daily_Transactions' in self.df.columns and 'BTC_Price' in self.df.columns:
            addr = self.df['Unique_Addresses']
            tx = self.df['Daily_Transactions']
            price = self.df['BTC_Price']

            # Dywergencja Cenowo-Adresowa (Różnica dynamiki z 28 dni)
            price_ret_28 = np.log(price / price.shift(28))
            addr_ret_28 = np.log((addr + eps) / (addr.shift(28) + eps))
            self.features_df['Price_Address_Divergence_28d'] = price_ret_28 - addr_ret_28

            # Z-Score Aktywności Sieci (Anomalia z 56 dni)
            sma_56_addr = addr.rolling(window=56).mean()
            std_56_addr = addr.rolling(window=56).std()
            self.features_df['Network_Hype_ZScore_56d'] = np.where(std_56_addr == 0, 0,
                                                                   (addr - sma_56_addr) / std_56_addr)

            # Wskaźnik Spekulacyjnej Gęstości
            tx_per_addr = tx / (addr + eps)
            self.features_df['Tx_per_Address_SMA_7d'] = tx_per_addr.rolling(window=7).mean()

        # --- 3. OPŁATY SIECIOWE I ZATORY (Fees & Urgency) ---
        if 'Total_Fees_BTC' in self.df.columns and 'Miners_Revenue_USD' in self.df.columns and 'Daily_Transactions' in self.df.columns:
            fees_btc = self.df['Total_Fees_BTC']
            revenue_usd = self.df['Miners_Revenue_USD']
            tx = self.df['Daily_Transactions']
            price = self.df['BTC_Price']

            # A. Wskaźnik Presji Opłat (Fee-to-Revenue Ratio wygładzony 7-dniową średnią)
            fees_usd = fees_btc * price
            fee_to_revenue = fees_usd / (revenue_usd + eps)
            self.features_df['Fee_to_Revenue_Ratio_7d'] = fee_to_revenue.rolling(window=7).mean()

            # B. Koszt Pośpiechu: Z-Score średniej opłaty za transakcję (Z okna 56 dni)
            avg_fee_per_tx = fees_btc / (tx + eps)
            sma_56_fee = avg_fee_per_tx.rolling(window=56).mean()
            std_56_fee = avg_fee_per_tx.rolling(window=56).std()
            self.features_df['Fee_Urgency_ZScore_56d'] = np.where(std_56_fee == 0, 0,
                                                                  (avg_fee_per_tx - sma_56_fee) / std_56_fee)

        return self

    def build_halving_features(self):
        """NOWA METODA: Tworzy Zegar Makroekonomiczny oparty na cyklach Halvingu Bitcoina"""
        print("-> Budowanie bloku cykli Halvingowych (Kodowanie cykliczne)...")

        if 'Daily_Blocks_Mined' in self.df.columns:
            # 1. Odtworzenie całkowitej liczby wykopanych bloków (Total Blocks)
            anchor_date = pd.to_datetime('2020-05-11')
            anchor_blocks = 630027

            if anchor_date not in self.df.index:
                anchor_date = self.df.index[self.df.index.get_indexer([anchor_date], method='nearest')[0]]

            cumsum_blocks = self.df['Daily_Blocks_Mined'].cumsum()
            base_offset = anchor_blocks - cumsum_blocks.loc[anchor_date]
            self.features_df['Total_Blocks'] = cumsum_blocks + base_offset

            # 2. Obliczenie postępu cyklu Halvingowego (Halving co 210 000 bloków)
            cycle_length = 210000
            blocks_since_halving = self.features_df['Total_Blocks'] % cycle_length
            self.features_df['Halving_Progress'] = blocks_since_halving / cycle_length

            # 3. Kodowanie cykliczne (Sin/Cos)
            self.features_df['Halving_Sin'] = np.sin(2 * np.pi * self.features_df['Halving_Progress'])
            self.features_df['Halving_Cos'] = np.cos(2 * np.pi * self.features_df['Halving_Progress'])

            self.features_df.drop(columns=['Total_Blocks'], inplace=True)

        return self

    def save_features(self, start_date='2018-02-01'):
        """Odcina bufor historyczny, czyści ewentualne braki i zapisuje plik"""
        print(f"-> Odcinanie bufora historycznego. Start modelu od: {start_date}")
        self.features_df = self.features_df[self.features_df.index >= start_date]
        self.features_df.dropna(inplace=True)

        print(f"-> Zapisywanie gotowej macierzy cech do pliku: {self.output_filepath}")
        self.features_df.to_csv(self.output_filepath)

        print("\n--- PODGLĄD METRYK GORĄCZKI SIECIOWEJ ---")
        cols_to_show = ['Fee_to_Revenue_Ratio_7d', 'Fee_Urgency_ZScore_56d']
        if all(col in self.features_df.columns for col in cols_to_show):
            print(self.features_df[cols_to_show].head())

        print(
            f"\nRozmiar OSTATECZNEGO zbioru: {self.features_df.shape[1]} kolumn, {self.features_df.shape[0]} wierszy.")
        return self.features_df

    def run_pipeline(self):
        print("Rozpoczynam Inżynierię Cech (Feature Engineering)...")
        (self.build_price_derived_features()
         .build_macro_features()
         .build_sentiment_features()
         .build_derivatives_features()
         .build_volume_features()
         .build_onchain_features()
         .build_halving_features()
         .save_features())
        return self.features_df


# ==========================================
# URUCHOMIENIE SKRYPTU
# ==========================================
if __name__ == "__main__":
    engineer = BitcoinFeatureEngineer('btc_ensemble_features.csv', 'btc_ml_features.csv')
    ml_dataset = engineer.run_pipeline()