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
FETCH_START_DATE = '2014-09-01'
MODEL_START_DATE = '2014-09-01'  # Start modelu (Zbiega się z powstaniem Fear & Greed Index!)
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


class BitcoinDataIntegrator:
    def __init__(self):
        self.fred = Fred(api_key=FRED_API_KEY)
        self.df_main = pd.DataFrame()

    def get_yahoo_data(self):
        """Pobiera dane Spot (OHLCV dla BTC) i Makro (Cena zamknięcia DXY, NASDAQ, Złoto, Ropa WTI) z Yahoo Finance"""
        print("-> Pobieranie danych z Yahoo Finance (w tym OHLC dla BTC oraz ceny Złota i Ropy WTI)...")
        tickers = ['BTC-USD', '^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F', '^VIX', '^TNX']

        # Pobieranie danych (używamy daty buforowej)
        data = yf.download(tickers, start=FETCH_START_DATE, end=END_DATE)

        # Ekstrakcja danych makro (tylko ceny zamknięcia - Close)
        macro_df = data['Close'][['^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F', '^VIX', '^TNX']].rename(columns={
            '^IXIC': 'NASDAQ_100',
            'DX-Y.NYB': 'DXY_Index',
            'GC=F': 'Gold_Close',
            'CL=F': 'WTI_Oil_Close',
            '^VIX': 'VIX_Index',
            '^TNX': 'TNX'
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
        """Pobiera dane makroekonomiczne z FRED (M2, Stopy procentowe, Inflacja)"""
        print("-> Pobieranie danych makro z FRED (M2, FEDFUNDS, Core CPI)...")
        try:
            # 1. Podaż pieniądza M2 (M2SL)
            m2_data = self.fred.get_series('M2SL', observation_start=FETCH_START_DATE)
            m2_df = pd.DataFrame(m2_data, columns=['M2_Supply'])

            # 2. Stopy procentowe FED (FEDFUNDS)
            fed_data = self.fred.get_series('FEDFUNDS', observation_start=FETCH_START_DATE)
            fed_df = pd.DataFrame(fed_data, columns=['FEDFUNDS_Rate'])

            # 3. Inflacja bazowa (Core CPI - bez żywności i energii - CPILFESL)
            cpi_data = self.fred.get_series('CPILFESL', observation_start=FETCH_START_DATE)
            cpi_df = pd.DataFrame(cpi_data, columns=['Core_CPI'])

            # Łączenie w jeden DataFrame za pomocą join (dane miesięczne)
            macro_df = m2_df.join([fed_df, cpi_df], how='outer')
            macro_df.index = pd.to_datetime(macro_df.index).normalize()

            return macro_df

        except Exception as e:
            print(f"[!] Błąd FRED API: {e}")
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

    def get_bitmex_data(self, start_date, end_date):
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


    def get_bybit_data(self, start_date, end_date):
        """
        Pobiera historię Open Interest oraz Long/Short Ratio
        dla BTC z giełdy Bybit przy użyciu API v5.
        Zwraca połączoną dzienną serię danych (stan na koniec dnia).
        """
        print("-> Pobieranie wskaźników rynkowych z Bybit v5...")

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        # --- 1. POBIERANIE OPEN INTEREST ---
        print("   [1/2] Pobieranie Open Interest...")
        all_oi_data = []
        url_oi = "https://api.bybit.com/v5/market/open-interest"
        params_oi = {
            'category': 'linear', 'symbol': 'BTCUSDT',
            'intervalTime': '1d', 'limit': 500,
            'startTime': start_ts, 'endTime': end_ts
        }

        while True:
            try:
                res = requests.get(url_oi, params=params_oi, timeout=15)
                if res.status_code == 429: time.sleep(5); continue
                data = res.json()
                if data.get('retCode') != 0: break

                items = data.get('result', {}).get('list', [])
                if not items: break
                all_oi_data.extend(items)

                cursor = data.get('result', {}).get('nextPageCursor')
                if not cursor: break
                params_oi['cursor'] = cursor
                time.sleep(0.2)
            except Exception:
                break

        df_oi = pd.DataFrame(all_oi_data)
        if not df_oi.empty:
            df_oi['date'] = pd.to_datetime(df_oi['timestamp'].astype(float), unit='ms').dt.normalize()
            df_oi['Bybit_Open_Interest'] = df_oi['openInterest'].astype(float)
            df_oi = df_oi.groupby('date')['Bybit_Open_Interest'].last().to_frame()
        else:
            df_oi = pd.DataFrame(columns=['Bybit_Open_Interest'])



        # --- 2. POBIERANIE LONG/SHORT RATIO ---
        print("   [2/2] Pobieranie Long/Short Ratio...")
        all_ratio = []
        url_ratio = "https://api.bybit.com/v5/market/account-ratio"
        params_ratio = {
            'category': 'linear', 'symbol': 'BTCUSDT',
            'period': '1d', 'limit': 500,
            'startTime': start_ts, 'endTime': end_ts
        }

        while True:
            try:
                res = requests.get(url_ratio, params=params_ratio, timeout=15)
                if res.status_code == 429: time.sleep(5); continue
                data = res.json()

                items = data.get('result', {}).get('list', [])
                if not items: break
                all_ratio.extend(items)

                cursor = data.get('result', {}).get('nextPageCursor')
                if not cursor: break
                params_ratio['cursor'] = cursor
                time.sleep(0.2)
            except Exception:
                break

        df_ratio = pd.DataFrame(all_ratio)
        if not df_ratio.empty:
            df_ratio['date'] = pd.to_datetime(df_ratio['timestamp'].astype(float), unit='ms').dt.normalize()
            df_ratio['Bybit_Long_Short_Ratio'] = (
                        df_ratio['buyRatio'].astype(float) / df_ratio['sellRatio'].astype(float))
            df_ratio = df_ratio.groupby('date')['Bybit_Long_Short_Ratio'].last().to_frame()
        else:
            df_ratio = pd.DataFrame(columns=['Bybit_Long_Short_Ratio'])

        # --- ŁĄCZENIE DANYCH (MERGE) ---
        print("   [+] Łączenie i czyszczenie danych...")
        dfs = [df_oi, df_ratio]
        merged_df = pd.concat(dfs, axis=1, join='outer').sort_index()

        # Odfiltrowanie do ścisłego przedziału dat
        merged_df = merged_df[(merged_df.index >= start_date) & (merged_df.index <= end_date)]

        print("-> Zakończono! Zwracam połączony DataFrame wskaźników.")
        return merged_df

    def get_defillama_data(self):
        """
        Pobiera kompleksowe dane o płynności, ryzyku i przepływach kapitału z API DefiLlama:
        1. Podaż Stablecoinów (Stablecoin_Total_MCap)
        2. Total Value Locked (DeFi_Global_TVL)
        3. Wolumen na giełdach DEX (DEX_Daily_Volume)
        4. Dzienne opłaty sieciowe (DeFi_Global_Daily_Fees)
        5. Straty z ataków hakerskich (DeFi_Daily_Hacks_Loss_USD)
        """
        print("-> Pobieranie maksymalnego zestawu danych (7 wskaźników) z DefiLlama...")
        dfs = []

        # 1. Całkowita kapitalizacja Stablecoinów
        try:
            res = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15)
            if res.status_code == 200:
                parsed = [{'date': pd.to_datetime(int(i['date']), unit='s').normalize(),
                           'Stablecoin_Total_MCap': float(i.get('totalCirculating', {}).get('peggedUSD', 0))}
                          for i in res.json() if i.get('totalCirculating', {}).get('peggedUSD')]
                dfs.append(pd.DataFrame(parsed).set_index('date'))
        except Exception as e:
            print(f"   [!] Błąd API Stablecoinów: {e}")

        # 2. Global DeFi TVL
        try:
            res = requests.get("https://api.llama.fi/charts", timeout=15)
            if res.status_code == 200:
                parsed = [{'date': pd.to_datetime(int(i['date']), unit='s').normalize(),
                           'DeFi_Global_TVL': float(i['totalLiquidityUSD'])}
                          for i in res.json()]
                dfs.append(pd.DataFrame(parsed).set_index('date'))
        except Exception as e:
            print(f"   [!] Błąd API TVL: {e}")

        # 3. Global DEX Volume
        try:
            res = requests.get("https://api.llama.fi/overview/dexs?excludeTotalDataChart=false&dataType=dailyVolume",
                               timeout=15)
            if res.status_code == 200:
                parsed = [{'date': pd.to_datetime(int(i[0]), unit='s').normalize(),
                           'DEX_Daily_Volume': float(i[1])}
                          for i in res.json().get('totalDataChart', [])]
                dfs.append(pd.DataFrame(parsed).set_index('date'))
        except Exception as e:
            print(f"   [!] Błąd API DEX Volume: {e}")

        # 4. Globalne opłaty (Daily Fees)
        try:
            res = requests.get("https://api.llama.fi/overview/fees?excludeTotalDataChart=false&dataType=dailyFees",
                               timeout=15)
            if res.status_code == 200:
                parsed = [{'date': pd.to_datetime(int(i[0]), unit='s').normalize(),
                           'DeFi_Global_Daily_Fees': float(i[1])}
                          for i in res.json().get('totalDataChart', [])]
                dfs.append(pd.DataFrame(parsed).set_index('date'))
        except Exception as e:
            print(f"   [!] Błąd API Daily Fees: {e}")

        # 5. Ataki hakerskie i exploity (Hacks)
        try:
            res = requests.get("https://api.llama.fi/hacks", timeout=15)
            if res.status_code == 200:
                data = res.json()
                hacks_list = data.get('hacks', data) if isinstance(data, dict) else data
                parsed = []
                for h in hacks_list:
                    ts = h.get('date')
                    raw_amount = h.get('amount') if h.get('amount') is not None else h.get('amountLost', 0)
                    try:
                        amount = float(raw_amount)
                    except:
                        amount = 0.0
                    if ts:
                        try:
                            date_val = pd.to_datetime(int(ts), unit='s').normalize()
                        except:
                            date_val = pd.to_datetime(ts).normalize()
                        parsed.append({'date': date_val, 'DeFi_Daily_Hacks_Loss_USD': amount})
                if parsed:
                    df_hacks = pd.DataFrame(parsed).groupby('date')['DeFi_Daily_Hacks_Loss_USD'].sum().to_frame()
                    dfs.append(df_hacks)
        except Exception as e:
            print(f"   [!] Błąd API Hacks: {e}")


        if not dfs:
            print("   [!] Nie udało się pobrać żadnych danych z DefiLlama.")
            return pd.DataFrame()

        # Łączenie wszystkich wskaźników z DefiLlama w jedną tabelę
        df_final = dfs[0].join(dfs[1:], how='outer')
        df_final.sort_index(inplace=True)
        df_final = df_final[~df_final.index.duplicated(keep='last')]

        # KLUCZOWE ZABEZPIECZENIE (Brak ataków / Brak ogłoszeń VC = 0 USD)
        for col in ['DeFi_Daily_Hacks_Loss_USD', 'DeFi_Daily_VC_Raises_USD']:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        return df_final

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
            'metrics': 'BlkCnt, IssTotUSD',
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

        market = self.get_yahoo_data()
        macro = self.get_macro_data()
        fng = self.get_fear_greed()
        funding = self.get_bitmex_data(FETCH_START_DATE, END_DATE)
        bybit = self.get_bybit_data(FETCH_START_DATE, END_DATE)
        stable = self.get_defillama_data()
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
            bybit,
            stable,
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