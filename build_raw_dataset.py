import pandas as pd
import yfinance as yf
from fredapi import Fred
import requests
from datetime import datetime, timedelta
import time


# ==========================================
# KONFIGURACJA GŁÓWNA
# ==========================================
FRED_API_KEY = 'f3ac7094f956fdb519c4f98c2453e476'
FETCH_START_DATE = '2011-01-01'
MODEL_START_DATE = '2018-02-01'  # Start modelu (Zbiega się z powstaniem Fear & Greed Index!)
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


class BitcoinDataIntegrator:
    def __init__(self):
        self.fred = Fred(api_key=FRED_API_KEY)
        self.df_main = pd.DataFrame()

    def get_market_data(self):
        """Pobiera dane Spot (OHLCV dla BTC) i Makro (Cena zamknięcia DXY, NASDAQ, Złoto, Ropa WTI) z Yahoo Finance"""
        print("-> Pobieranie danych z Yahoo Finance (w tym OHLC dla BTC oraz ceny Złota i Ropy WTI)...")
        tickers = ['BTC-USD', '^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F']

        # Pobieranie danych (używamy daty buforowej)
        data = yf.download(tickers, start=FETCH_START_DATE, end=END_DATE)

        # Ekstrakcja danych makro (tylko ceny zamknięcia - Close)
        macro_df = data['Close'][['^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F']].rename(columns={
            '^IXIC': 'NASDAQ_100',
            'DX-Y.NYB': 'DXY_Index',
            'GC=F': 'Gold_Close',
            'CL=F': 'WTI_Oil_Close'
        })

        # Ekstrakcja pełnego OHLCV dla Bitcoina
        btc_ohlcv = pd.DataFrame({
            'BTC_Open': data['Open']['BTC-USD'],
            'BTC_High': data['High']['BTC-USD'],
            'BTC_Low': data['Low']['BTC-USD'],
            'BTC_Close': data['Close']['BTC-USD'],
            'BTC_Volume': data['Volume']['BTC-USD']
        })

        # Łączenie w jeden DataFrame
        market_df = pd.concat([btc_ohlcv, macro_df], axis=1)
        market_df.index = pd.to_datetime(market_df.index).normalize()
        return market_df

    def get_macro_data(self):
        """Pobiera podaż pieniądza M2 z FRED"""
        print("-> Pobieranie podaży pieniądza M2 z FRED...")
        try:
            m2_data = self.fred.get_series('M2SL', observation_start=FETCH_START_DATE)
            m2_df = pd.DataFrame(m2_data, columns=['M2_Supply'])
            m2_df.index = pd.to_datetime(m2_df.index).normalize()
            return m2_df
        except Exception as e:
            print(f"Błąd FRED API: {e}")
            return pd.DataFrame()

    def get_fear_greed(self):
        """Pobiera Fear & Greed Index z API Alternative.me"""
        print("-> Pobieranie Sentymentu (Fear & Greed Index)...")

        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url).json()

        fng_data = {}
        for item in response['data']:
            date_str = datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d')
            fng_data[date_str] = int(item['value'])

        fng_df = pd.DataFrame.from_dict(fng_data, orient='index', columns=['Fear_Greed_Index'])
        fng_df.index = pd.to_datetime(fng_df.index).normalize()
        return fng_df

    def get_google_trends(self):
        """Pobiera historyczne zainteresowanie słowem 'Bitcoin' z Google Trends"""
        print("-> Pobieranie danych z Google Trends (Sentyment wyszukiwań)...")
        from pytrends.request import TrendReq

        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            kw_list = ["Bitcoin"]
            timeframe = f"{FETCH_START_DATE} {END_DATE}"
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')

            trends_df = pytrends.interest_over_time()

            if trends_df.empty:
                print("[!] Google Trends zwróciło pusty zbiór. Być może nałożono limit (Rate Limit).")
                return pd.DataFrame()

            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])

            trends_df.columns = ['Google_Trends_BTC']
            trends_df.index = pd.to_datetime(trends_df.index).normalize()

            daily_trends = trends_df.resample('D').ffill()
            return daily_trends

        except Exception as e:
            print(f"[!] Błąd API Google Trends: {e}")
            return pd.DataFrame()

    def get_bitmex_funding(self, start_date, end_date):
        """Pobiera historię Funding Rate z giełdy BitMEX w oparciu o silnik Pandas"""
        print("-> Pobieranie historii Funding Rate z BitMEX...")
        all_funding_dfs = []

        current_start = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while current_start < end_dt:
            url = "https://www.bitmex.com/api/v1/funding"
            params = {
                'symbol': 'XBTUSD',
                'count': 500,
                'startTime': current_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }

            try:
                response = requests.get(url, params=params)

                if response.status_code == 429:
                    print("   [Oczekiwanie] Limit zapytań API BitMEX, pauza 5 sekund...")
                    time.sleep(5)
                    continue
                elif response.status_code != 200:
                    print(f"   [!] API BitMEX odrzuciło zapytanie. Kod: {response.status_code}")
                    break

                data = response.json()
                if not data:
                    break

                temp_df = pd.DataFrame(data)
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']).dt.tz_localize(None)

                filtered_df = temp_df[temp_df['timestamp'] < end_dt]

                if not filtered_df.empty:
                    all_funding_dfs.append(filtered_df)

                last_ts = temp_df['timestamp'].iloc[-1]

                if last_ts <= current_start or len(data) < 500:
                    break

                current_start = last_ts + pd.Timedelta(seconds=1)
                time.sleep(1.5)

            except Exception as e:
                print(f"[!] Błąd przetwarzania danych BitMEX: {e}")
                break

        if not all_funding_dfs:
            print("[!] Nie udało się pobrać żadnych danych z BitMEX.")
            return pd.DataFrame()

        df = pd.concat(all_funding_dfs, ignore_index=True)
        df['date'] = df['timestamp'].dt.normalize()
        daily_funding = df.groupby('date')['fundingRate'].last().to_frame()
        return daily_funding

    def get_binance_funding(self, start_date, end_date):
        """Pobiera historię Funding Rate z Binance"""
        print("-> Pobieranie historii Funding Rate z Binance...")
        all_funding = []

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        while True:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': 'BTCUSDT', 'limit': 1000, 'startTime': start_ts}
            response = requests.get(url, params=params).json()

            if not response or len(response) == 0:
                break

            filtered_data = [d for d in response if d['fundingTime'] <= end_ts]
            all_funding.extend(filtered_data)

            last_ts = response[-1]['fundingTime']
            if last_ts <= start_ts or last_ts >= end_ts:
                break

            start_ts = last_ts + 1
            time.sleep(0.2)

        funding_df = pd.DataFrame(all_funding)
        funding_df['date'] = pd.to_datetime(funding_df['fundingTime'], unit='ms').dt.normalize()
        funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)

        daily_funding = funding_df.groupby('date')['fundingRate'].last().to_frame()
        return daily_funding

    def get_combined_funding(self):
        """Łączy dane z BitMEX (starsze) oraz Binance (nowsze) w jedną spójną serię"""
        print("\n-> Integracja finansowania na rynku derywatów (BitMEX + Binance)...")

        bitmex_df = self.get_bitmex_funding(FETCH_START_DATE, '2019-09-10')
        binance_df = self.get_binance_funding('2019-09-10', END_DATE)

        combined_df = pd.concat([bitmex_df, binance_df])
        combined_df.columns = ['Funding_Rate_Last']

        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        return combined_df


    def get_blockchaininfo_data(self):
        """Pobiera historyczne, surowe dane On-chain z Blockchain.info API"""
        print("-> Pobieranie surowych danych On-Chain...")

        # Słownik zawiera wyłącznie faktycznie istniejące endpointy zwracające surowe wartości
        charts = {
            # 1. Górnictwo i Bezpieczeństwo Sieci
            'hash-rate': 'Hashrate',
            'difficulty': 'Difficulty',
            'miners-revenue': 'Miners_Revenue_USD',

            # 2. Mempool i Przepustowość
            'mempool-size': 'Mempool_Size_Bytes',
            'mempool-count': 'Mempool_Tx_Count',
            'median-confirmation-time': 'Median_Conf_Time',

            # 3. Aktywność Transakcyjna
            'n-transactions': 'Tx_Count',
            'n-transactions-excluding-popular': 'Tx_Retail_Count',
            'n-transactions-per-block': 'Avg_Tx_Per_Block',
            'estimated-transaction-volume': 'Est_Tx_Volume_BTC',
            'avg-block-size': 'Avg_Block_Size_MB',

            # 4. Opłaty i Koszty
            'transaction-fees': 'Total_Fees_BTC',
            'cost-per-transaction-percent': 'Cost_Per_Tx_Percent',

            # 5. Użytkownicy i Stan Księgi
            'n-unique-addresses': 'Unique_Addresses',
            'utxo-count': 'UTXO_Count',

            # 6. Rynek i Podaż
            'total-bitcoins': 'Circulating_Supply',
            'trade-volume': 'Exchange_Trade_Volume_USD',
        }

        onchain_df = pd.DataFrame()

        for endpoint, col_name in charts.items():
            url = f"https://api.blockchain.info/charts/{endpoint}?timespan=all&sampled=true&format=json"

            try:
                response_raw = requests.get(url, timeout=10)
                response_raw.raise_for_status()
                response = response_raw.json()
            except Exception as e:
                print(f"[!] Błąd pobierania dla {col_name}: {e}")
                continue

            if 'values' not in response:
                print(f"[!] Pominięto '{col_name}': Brak klucza 'values' w odpowiedzi.")
                continue

            temp_dict = {datetime.fromtimestamp(item['x']).strftime('%Y-%m-%d'): item['y']
                         for item in response['values']}
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[col_name])
            temp_df.index = pd.to_datetime(temp_df.index).normalize()

            if onchain_df.empty:
                onchain_df = temp_df
            else:
                onchain_df = onchain_df.join(temp_df, how='outer')

        onchain_df.sort_index(inplace=True)
        onchain_df.ffill(inplace=True)

        return onchain_df

    def get_coinmetrics_data(self):
        """Pobiera darmowe dane On-Chain z API CoinMetrics (Bloki, Transakcje, Aktywne Adresy, Surowa Emisja)"""
        print("-> Pobieranie darmowych metryk on-chain z CoinMetrics...")

        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        all_data = []

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        params = {
            'assets': 'btc',
            # DODANO: IssTotUSD (Surowa wartość nowej emisji BTC w USD)
            'metrics': 'BlkCnt,TxCnt,AdrActCnt,IssTotUSD',
            'frequency': '1d',
            'start_time': f"{FETCH_START_DATE}T00:00:00Z",
            'end_time': f"{END_DATE}T00:00:00Z",
            'page_size': 1000
        }

        while True:
            try:
                response = requests.get(url, params=params, headers=headers)

                if response.status_code != 200:
                    print(f"   [!] Błąd API CoinMetrics. Kod: {response.status_code}.")
                    break

                json_data = response.json()
                data_chunk = json_data.get('data', [])

                if not data_chunk:
                    break

                all_data.extend(data_chunk)

                next_page_token = json_data.get('next_page_token')
                if not next_page_token:
                    break

                params['next_page_token'] = next_page_token
                time.sleep(0.2)

            except Exception as e:
                print(f"[!] Błąd połączenia z CoinMetrics: {e}")
                break

        if not all_data:
            print("   [!] Nie udało się pobrać żadnych danych z CoinMetrics.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None).dt.normalize()

        # Konwersja wszystkich pobranych metryk na wartości numeryczne
        for col in ['BlkCnt', 'IssTotUSD']:
            df[col] = pd.to_numeric(df.get(col), errors='coerce')

        # Zmiana nazw kolumn na surowe nazwy czytelne dla modelu
        df.rename(columns={
            'BlkCnt': 'Daily_Blocks_Mined',
            'IssTotUSD': 'Daily_Issuance_USD'  # <--- SUROWE DANE DOSTAWIONE DO CSV
        }, inplace=True)

        final_df = df.set_index('date')[
            ['Daily_Blocks_Mined', 'Daily_Issuance_USD']]
        final_df = final_df[~final_df.index.duplicated(keep='last')]

        return final_df

    def build_dataset(self):
        """Uruchamia funkcje, łączy tabele i przygotowuje dane pod inżynierię cech"""
        print("\nROZPOCZYNAM INTEGRACJĘ DANYCH...")

        market = self.get_market_data()
        macro = self.get_macro_data()
        fng = self.get_fear_greed()
        funding = self.get_combined_funding()
        onchain = self.get_blockchaininfo_data()
        trends = self.get_google_trends()
        blocks = self.get_coinmetrics_data()

        print("\n-> Łączenie zbiorów danych (Merging)...")
        dfs = [
            market,
            macro,
            trends,
            fng,
            funding,
            onchain,
            blocks
            ]

        self.df_main = dfs[0].join(dfs[1:], how='outer')

        print("-> Rozwiązywanie problemu brakujących danych w weekendy (Forward Fill)...")
        self.df_main.ffill(inplace=True)

        self.df_main = self.df_main[self.df_main.index >= FETCH_START_DATE]
        self.df_main = self.df_main[self.df_main.index <= END_DATE]

        self.df_main.dropna(subset=['BTC_Close'], inplace=True)

        print("\nGOTOWE! Zestawienie pierwszych 5 rekordów dla Modelu:")
        print(self.df_main.head())

        filename = 'btc_raw_data.csv'
        self.df_main.to_csv(filename)
        print(f"\nZapisano pełny zbiór danych do pliku: {filename}")

        return self.df_main


# ==========================================
# URUCHOMIENIE SKRYPTU
# ==========================================
if __name__ == "__main__":
    integrator = BitcoinDataIntegrator()
    dataset = integrator.build_dataset()