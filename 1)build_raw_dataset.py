import pandas as pd
import yfinance as yf
from fredapi import Fred
import requests
from datetime import datetime, timedelta
import time
import random
from pytrends.request import TrendReq

# ==========================================
# Konfiguracja Główna
# ==========================================
FRED_API_KEY = 'f3ac7094f956fdb519c4f98c2453e476'
CCDATA_API_KEY = '23c6099e959e8b974663970bad395a95d15ade135e47953330a69d5f2ba915d0'
FETCH_START_DATE = '2010-12-01'
MODEL_START_DATE = '2011-01-01'
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


class BitcoinDataIntegrator:
    # ==========================================
    # Inicjalizacja Klasy
    # ==========================================
    def __init__(self):
        self.fred = Fred(api_key=FRED_API_KEY)
        self.df_main = pd.DataFrame()

    # ==========================================
    # Pobieranie Danych: Yahoo Finance
    # ==========================================
    def get_yahoo_data(self):
        print("-> Pobieranie danych z Yahoo Finance (w tym OHLC dla BTC oraz ceny Złota i Ropy WTI)...")
        tickers = ['BTC-USD', '^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F', '^VIX', '^TNX']
        yf_end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download(tickers, start=FETCH_START_DATE, end=yf_end_date)

        macro_df = data['Close'][['^IXIC', 'DX-Y.NYB', 'GC=F', 'CL=F', '^VIX', '^TNX']].rename(columns={
            '^IXIC': 'NASDAQ_100',
            'DX-Y.NYB': 'DXY_Index',
            'GC=F': 'Gold_Close',
            'CL=F': 'WTI_Oil_Close',
            '^VIX': 'VIX_Index',
            '^TNX': 'TNX'
            })

        # btc_ohlcv = pd.DataFrame({
        #     'BTC_Open': data['Open']['BTC-USD'],
        #     'BTC_High': data['High']['BTC-USD'],
        #     'BTC_Low': data['Low']['BTC-USD'],
        #     'BTC_Close': data['Close']['BTC-USD'],
        # })
        #
        # market_df = pd.concat([macro_df], axis=1)
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()

        return macro_df


    # ==========================================
    # Pobieranie Danych: CCData (CryptoCompare) dla BTC
    # ==========================================

    def get_cryptocompare_btc_data(self):
        print("-> Pobieranie historycznych danych OHLC dla BTC z CCData (CCCAGG Index)...")
        url = "https://min-api.cryptocompare.com/data/v2/histoday"

        all_data = []
        end_ts = int(datetime.strptime(END_DATE, '%Y-%m-%d').timestamp())
        start_ts = int(datetime.strptime(FETCH_START_DATE, '%Y-%m-%d').timestamp())

        current_to_ts = end_ts

        # Dodanie nagłówków, aby obejść blokady (np. Cloudflare)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        # Jeśli podano klucz API, autoryzuj żądanie
        if CCDATA_API_KEY:
            headers["authorization"] = f"Apikey {CCDATA_API_KEY}"

        while current_to_ts > start_ts:
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': 2000,
                'toTs': current_to_ts
            }

            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)

                # Bezpieczne ładowanie JSON, w razie bloku WAF dostaniemy HTML
                try:
                    data = response.json()
                except ValueError:
                    print(f"[!] Błąd parsowania JSON. Kod statusu: {response.status_code}")
                    print(f"[!] Surowa odpowiedź: {response.text[:300]}")
                    break

                if data.get('Response') != 'Success':
                    # Pobieramy całą odpowiedź, by zobaczyć faktyczny błąd, jeśli 'Message' nie istnieje
                    error_msg = data.get('Message', 'Brak zdefiniowanej wiadomości o błędzie')
                    print(f"[!] Błąd API CryptoCompare: {error_msg}")
                    print(f"[!] Pełny payload API: {data}")
                    break

                # Ważne: Sprawdzamy czy struktura danych na pewno istnieje
                if 'Data' not in data or 'Data' not in data['Data']:
                    print("[!] Otrzymano nieoczekiwaną strukturę danych API.")
                    print(f"[!] Podgląd: {data}")
                    break

                chunk = data['Data']['Data']
                if not chunk:
                    break

                all_data.extend(chunk)

                oldest_in_chunk = chunk[0]['time']

                if oldest_in_chunk >= current_to_ts:
                    break

                current_to_ts = oldest_in_chunk - 86400

                # Wydłużony sleep dla darmowego API bez klucza (CryptoCompare bywa wrażliwe)
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"[!] Błąd połączenia z API CryptoCompare: {e}")
                break
            except Exception as e:
                print(f"[!] Nieoczekiwany błąd: {e}")
                break

        if not all_data:
            print("[!] Nie pobrano danych z CryptoCompare.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['time'], unit='s').dt.normalize()

        df = df.drop_duplicates(subset=['date']).sort_values('date')
        df = df[df['date'] >= pd.to_datetime(FETCH_START_DATE)]

        df.rename(columns={
            'open': 'BTC_Open',
            'high': 'BTC_High',
            'low': 'BTC_Low',
            'close': 'BTC_Close',
            'volumeto': 'BTC_Volume_USD'
        }, inplace=True)

        df.set_index('date', inplace=True)

        return df[['BTC_Open', 'BTC_High', 'BTC_Low', 'BTC_Close', 'BTC_Volume_USD']]

    # ==========================================
    # Pobieranie Danych: Makroekonomia (FRED)
    # ==========================================
    def get_macro_data(self):
        print("-> Pobieranie danych makro z FRED (ALFRED + DFF Shift)...")
        macro_dfs = []

        series_ids_vintage = {
            'M2SL': 'M2_Supply',
            'CPILFESL': 'Core_CPI'
        }

        for series_id, col_name in series_ids_vintage.items():
            try:
                df_releases = self.fred.get_series_all_releases(series_id)
                df_releases['realtime_start'] = pd.to_datetime(df_releases['realtime_start']).dt.normalize()
                df_clean = df_releases.groupby('realtime_start')['value'].last().to_frame(name=col_name)
                df_clean = df_clean[df_clean.index >= pd.to_datetime(FETCH_START_DATE)]

                macro_dfs.append(df_clean)
                time.sleep(0.5)
            except Exception as e:
                print(f"   [!] Błąd pobierania ALFRED dla {series_id}: {e}")

        try:
            dff_series = self.fred.get_series('DFF', observation_start=FETCH_START_DATE)
            dff_df = dff_series.to_frame(name='DFF_Rate')
            dff_df.index = pd.to_datetime(dff_df.index) + pd.tseries.offsets.BDay(1)
            dff_df.index = dff_df.index.normalize()
            dff_df = dff_df[~dff_df.index.duplicated(keep='last')]
            macro_dfs.append(dff_df)
        except Exception as e:
            print(f"   [!] Błąd pobierania DFF: {e}")

        if not macro_dfs:
            print("[!] Nie udało się pobrać żadnych danych makro z FRED.")
            return pd.DataFrame()

        macro_df = macro_dfs[0].join(macro_dfs[1:], how='outer')
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()

        return macro_df

    # ==========================================
    # Pobieranie Danych: Sentyment (Fear & Greed)
    # ==========================================
    def get_fear_greed(self):
        print("-> Pobieranie Sentymentu (Fear & Greed Index)...")

        url = "https://api.alternative.me/fng/?limit=0&format=json"
        try:
            response = requests.get(url, timeout=15).json()

            fng_data = {}
            for item in response['data']:
                date_str = datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d')
                fng_data[date_str] = int(item['value'])

            fng_df = pd.DataFrame.from_dict(fng_data, orient='index', columns=['Fear_Greed_Index'])
            fng_df.index = pd.to_datetime(fng_df.index).tz_localize(None).normalize()
            return fng_df
        except Exception as e:
            print(f"[!] Błąd pobierania Fear & Greed: {e}")
            return pd.DataFrame()

    # ==========================================
    # Pobieranie Danych: Google Trends
    # ==========================================
    def get_google_trends(self):
        print("-> Pobieranie danych z Google Trends (własne okienkowanie i skalowanie)...")

        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            kw_list = ["Bitcoin"]

            start_dt = pd.to_datetime(FETCH_START_DATE)
            end_dt = pd.to_datetime(END_DATE)

            window_days = 200
            overlap_days = 30
            step_days = window_days - overlap_days

            current_start = start_dt
            all_trends = []

            while current_start < end_dt:
                current_end = current_start + pd.Timedelta(days=window_days)
                if current_end > end_dt:
                    current_end = end_dt

                timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
                print(f"   Pobieranie okna: {timeframe}...")

                max_retries = 4
                df_window = pd.DataFrame()

                for attempt in range(max_retries):
                    try:
                        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')
                        df_window = pytrends.interest_over_time()
                        break

                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg:
                            wait_time = (attempt + 1) * 15 + random.uniform(3.0, 7.0)
                            print(
                                f"      [Próba {attempt + 1}/{max_retries} nieudana]: Limit zapytań (429). Czekam {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"      [Próba {attempt + 1}/{max_retries} nieudana]: {error_msg}")
                            time.sleep(10)

                if not df_window.empty:
                    if 'isPartial' in df_window.columns:
                        df_window = df_window.drop(columns=['isPartial'])
                    df_window.columns = ['Google_Trends_BTC']
                    all_trends.append(df_window)
                else:
                    print(f"   [!] Ostrzeżenie: Puste dane dla okna {timeframe}.")

                if current_end == end_dt:
                    break

                current_start = current_start + pd.Timedelta(days=step_days)

                time.sleep(random.uniform(3.0, 7.0))

            if not all_trends:
                print("[!] Nie udało się pobrać żadnych danych z Google Trends.")
                return pd.DataFrame()

            print("   Łączenie i matematyczne skalowanie okien...")
            final_df = all_trends[0].copy()

            for i in range(1, len(all_trends)):
                prev_df = final_df
                curr_df = all_trends[i].copy()

                overlap_index = prev_df.index.intersection(curr_df.index)

                if not overlap_index.empty:
                    prev_mean = prev_df.loc[overlap_index, 'Google_Trends_BTC'].mean()
                    curr_mean = curr_df.loc[overlap_index, 'Google_Trends_BTC'].mean()

                    if curr_mean > 0:
                        scaling_factor = prev_mean / curr_mean
                        curr_df['Google_Trends_BTC'] = curr_df['Google_Trends_BTC'] * scaling_factor

                new_days = curr_df[~curr_df.index.isin(final_df.index)]
                final_df = pd.concat([final_df, new_days])

            max_val = final_df['Google_Trends_BTC'].max()
            if max_val > 0:
                final_df['Google_Trends_BTC'] = (final_df['Google_Trends_BTC'] / max_val) * 100

            daily_trends = final_df.resample('D').ffill()
            daily_trends.index = pd.to_datetime(daily_trends.index).tz_localize(None).normalize()

            return daily_trends

        except Exception as e:
            print(f"[!] Błąd przetwarzania Google Trends: {e}")
            return pd.DataFrame()

    # ==========================================
    # Pobieranie Danych: BitMEX Funding Rate
    # ==========================================
    def get_bitmex_data(self, start_date, end_date):
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
                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 429:
                    time.sleep(30)
                    continue
                elif response.status_code != 200:
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
            return pd.DataFrame()

        df = pd.concat(all_funding_dfs, ignore_index=True)
        df['date'] = df['timestamp'].dt.normalize()

        daily_funding_max = df.groupby('date')['fundingRate'].max().rename('BitMEX_Funding_Rate_Max')
        daily_funding_min = df.groupby('date')['fundingRate'].min().rename('BitMEX_Funding_Rate_Min')

        daily_funding = pd.concat([daily_funding_max, daily_funding_min], axis=1)
        daily_funding.index = pd.to_datetime(daily_funding.index).tz_localize(None).normalize()

        return daily_funding

    # ==========================================
    # Pobieranie Danych: Bybit Wskaźniki Rynkowe
    # ==========================================
    def get_bybit_data(self, start_date, end_date):
        print("-> Pobieranie wskaźników rynkowych z Bybit v5...")

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

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
                if res.status_code == 429:
                    time.sleep(30)
                    continue
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
                if res.status_code == 429:
                    time.sleep(30)
                    continue
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

        dfs = [df_oi, df_ratio]
        merged_df = pd.concat(dfs, axis=1, join='outer').sort_index()
        merged_df = merged_df[(merged_df.index >= start_date) & (merged_df.index <= end_date)]
        merged_df.index = pd.to_datetime(merged_df.index).tz_localize(None).normalize()

        return merged_df

    # ==========================================
    # Pobieranie Danych: DefiLlama
    # ==========================================
    def get_defillama_data(self):
        print("-> Pobieranie danych z DefiLlama...")

        expected_columns = [
            'Stablecoin_Total_MCap',
            'DeFi_Global_TVL',
            'DEX_Daily_Volume',
            'DeFi_Global_Daily_Fees',
            'DeFi_Daily_Hacks_Loss_USD'
        ]

        data_dict = {col: pd.Series(dtype=float, name=col) for col in expected_columns}

        try:
            res = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15)
            if res.status_code == 200:
                parsed = {pd.to_datetime(int(i['date']), unit='s').normalize(): float(
                    i.get('totalCirculating', {}).get('peggedUSD', 0))
                          for i in res.json() if i.get('totalCirculating', {}).get('peggedUSD')}
                data_dict['Stablecoin_Total_MCap'] = pd.Series(parsed, name='Stablecoin_Total_MCap')
        except Exception:
            pass

        try:
            res = requests.get("https://api.llama.fi/charts", timeout=15)
            if res.status_code == 200:
                parsed = {pd.to_datetime(int(i['date']), unit='s').normalize(): float(i['totalLiquidityUSD'])
                          for i in res.json()}
                data_dict['DeFi_Global_TVL'] = pd.Series(parsed, name='DeFi_Global_TVL')
        except Exception:
            pass

        try:
            res = requests.get("https://api.llama.fi/overview/dexs?excludeTotalDataChart=false&dataType=dailyVolume",
                               timeout=15)
            if res.status_code == 200:
                parsed = {pd.to_datetime(int(i[0]), unit='s').normalize(): float(i[1])
                          for i in res.json().get('totalDataChart', [])}
                data_dict['DEX_Daily_Volume'] = pd.Series(parsed, name='DEX_Daily_Volume')
        except Exception:
            pass

        try:
            res = requests.get("https://api.llama.fi/overview/fees?excludeTotalDataChart=false&dataType=dailyFees",
                               timeout=15)
            if res.status_code == 200:
                parsed = {pd.to_datetime(int(i[0]), unit='s').normalize(): float(i[1])
                          for i in res.json().get('totalDataChart', [])}
                data_dict['DeFi_Global_Daily_Fees'] = pd.Series(parsed, name='DeFi_Global_Daily_Fees')
        except Exception:
            pass

        try:
            res = requests.get("https://api.llama.fi/hacks", timeout=15)
            if res.status_code == 200:
                data = res.json()
                hacks_list = data.get('hacks', data) if isinstance(data, dict) else data
                parsed_hacks = {}
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

                        if date_val in parsed_hacks:
                            parsed_hacks[date_val] += amount
                        else:
                            parsed_hacks[date_val] = amount

                data_dict['DeFi_Daily_Hacks_Loss_USD'] = pd.Series(parsed_hacks, name='DeFi_Daily_Hacks_Loss_USD')
        except Exception:
            pass

        df_final = pd.DataFrame(data_dict)
        df_final.index.name = 'date'

        df_final.index = pd.to_datetime(df_final.index).tz_localize(None).normalize()
        df_final.sort_index(inplace=True)
        df_final = df_final[~df_final.index.duplicated(keep='last')]

        return df_final

    # ==========================================
    # Pobieranie Danych: Blockchain.info On-Chain
    # ==========================================
    def get_blockchaininfo_data(self):
        print("-> Pobieranie surowych danych On-Chain...")

        charts = {
            # 'trade-volume': 'Exchange_Trade_Volume_USD',
            'hash-rate': 'Hashrate', 'difficulty': 'Difficulty', 'miners-revenue': 'Miners_Revenue_USD',
            'mempool-size': 'Mempool_Size_Bytes', 'mempool-count': 'Mempool_Tx_Count',
            'median-confirmation-time': 'Median_Conf_Time',
            'n-transactions': 'Tx_Count',
            'n-transactions-per-block': 'Avg_Tx_Per_Block', 'estimated-transaction-volume': 'Est_Tx_Volume_BTC',
            'avg-block-size': 'Avg_Block_Size_MB', 'transaction-fees': 'Total_Fees_BTC',
            'cost-per-transaction-percent': 'Cost_Per_Tx_Percent', 'n-unique-addresses': 'Unique_Addresses',
            'utxo-count': 'UTXO_Count', 'total-bitcoins': 'Circulating_Supply',
        }

        onchain_df = pd.DataFrame()

        for endpoint, col_name in charts.items():
            url = f"https://api.blockchain.info/charts/{endpoint}?timespan=all&sampled=false&format=json"

            try:
                response_raw = requests.get(url, timeout=10)
                response_raw.raise_for_status()
                response = response_raw.json()
            except Exception:
                continue

            if 'values' not in response:
                continue

            temp_dict = {datetime.fromtimestamp(item['x']).strftime('%Y-%m-%d'): item['y']
                         for item in response['values']}
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[col_name])

            if onchain_df.empty:
                onchain_df = temp_df
            else:
                onchain_df = onchain_df.join(temp_df, how='outer')

        onchain_df.sort_index(inplace=True)
        onchain_df.ffill(inplace=True)
        onchain_df.index = pd.to_datetime(onchain_df.index).tz_localize(None).normalize()

        return onchain_df

    # ==========================================
    # Pobieranie Danych: CoinMetrics On-Chain
    # ==========================================
    def get_coinmetrics_data(self):
        print("-> Pobieranie darmowych metryk on-chain z CoinMetrics...")

        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        all_data = []

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        params = {
            'assets': 'btc',
            'metrics': 'BlkCnt, IssTotUSD',
            'frequency': '1d',
            'end_time': f"{END_DATE}T00:00:00Z",
            'page_size': 1000
        }

        while True:
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code != 200: break

                json_data = response.json()
                data_chunk = json_data.get('data', [])
                if not data_chunk: break

                all_data.extend(data_chunk)
                next_page_token = json_data.get('next_page_token')
                if not next_page_token: break

                params['next_page_token'] = next_page_token
                time.sleep(0.2)
            except Exception:
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None).dt.normalize()

        for col in ['BlkCnt', 'IssTotUSD']:
            df[col] = pd.to_numeric(df.get(col), errors='coerce')

        df.rename(columns={'BlkCnt': 'Daily_Blocks_Mined', 'IssTotUSD': 'Daily_Issuance_USD'}, inplace=True)
        final_df = df.set_index('date')[['Daily_Blocks_Mined', 'Daily_Issuance_USD']]
        final_df = final_df[~final_df.index.duplicated(keep='last')]

        final_df.sort_index(inplace=True)

        final_df['Total_Blocks'] = final_df['Daily_Blocks_Mined'].cumsum()

        final_df['Daily_Blocks_Mined'] = final_df['Daily_Blocks_Mined'].fillna(0).astype(int)
        final_df['Total_Blocks'] = final_df['Total_Blocks'].fillna(0).astype(int)

        return final_df

    # ==========================================
    # Budowa Głównego Zestawienia (Integracja)
    # ==========================================
    def build_dataset(self):
        print("\nROZPOCZYNAM INTEGRACJĘ DANYCH...")

        btc_ohlc = self.get_cryptocompare_btc_data()
        market = self.get_yahoo_data()
        macro = self.get_macro_data()
        bitmex = self.get_bitmex_data(FETCH_START_DATE, END_DATE)
        bybit = self.get_bybit_data(FETCH_START_DATE, END_DATE)
        coinmetrics = self.get_coinmetrics_data()
        blockchaininfo = self.get_blockchaininfo_data()
        defillama = self.get_defillama_data()
        trends = self.get_google_trends()
        fng = self.get_fear_greed()


        print("\n-> Łączenie zbiorów danych (Merging)...")
        dfs = [
            btc_ohlc,
            market,
            blockchaininfo,
            coinmetrics,
            macro,
            bitmex,
            bybit,
            defillama,
            trends,
            fng
            ]
        dfs_valid = [df for df in dfs if not df.empty]

        if not dfs_valid:
            print("[!] Wszystkie źródła zwróciły puste zbiory.")
            return pd.DataFrame()

        self.df_main = dfs_valid[0].join(dfs_valid[1:], how='outer')

        print("-> Wypełnianie braków dla zdarzeń punktowych i wolumenów...")
        event_cols = ['DeFi_Daily_Hacks_Loss_USD', 'DEX_Daily_Volume', 'DeFi_Global_Daily_Fees']

        fill_events_dict = {col: 0 for col in event_cols if col in self.df_main.columns}
        self.df_main.fillna(value=fill_events_dict, inplace=True)

        print("-> Rozwiązywanie problemu brakujących danych w weekendy (Forward Fill)...")
        self.df_main.ffill(inplace=True)

        print(f"-> Przycinanie zbioru danych do docelowego horyzontu: {MODEL_START_DATE} do {END_DATE}")
        self.df_main = self.df_main[self.df_main.index >= MODEL_START_DATE]
        self.df_main = self.df_main[self.df_main.index <= END_DATE]


        print("-> Imputacja danych historycznych (inteligentne usuwanie NaN dla początkowych lat)...")


        if 'Fear_Greed_Index' in self.df_main.columns:
            self.df_main['Fear_Greed_Index'] = self.df_main['Fear_Greed_Index'].fillna(50)

        if 'Bybit_Long_Short_Ratio' in self.df_main.columns:
            self.df_main['Bybit_Long_Short_Ratio'] = self.df_main['Bybit_Long_Short_Ratio'].fillna(1.0)

        cols_to_zero_fill = [
            'Bybit_Open_Interest',
            'BitMEX_Funding_Rate_Max', 'BitMEX_Funding_Rate_Min',
            'Stablecoin_Total_MCap', 'DeFi_Global_TVL',
            'Mempool_Size_Bytes', 'Mempool_Tx_Count'
        ]

        fill_dict = {col: 0 for col in cols_to_zero_fill if col in self.df_main.columns}
        self.df_main.fillna(value=fill_dict, inplace=True)

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
# Uruchomienie Skryptu
# ==========================================
if __name__ == "__main__":
    integrator = BitcoinDataIntegrator()
    dataset = integrator.build_dataset()