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
FETCH_START_DATE = '2017-07-01'  # 7 miesięcy bufora przed modelem (na wyliczenie SMA 200)
MODEL_START_DATE = '2018-02-01'  # Start modelu (Zbiega się z powstaniem Fear & Greed Index!)
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


class BitcoinDataIntegrator:
    def __init__(self):
        self.fred = Fred(api_key=FRED_API_KEY)
        self.df_main = pd.DataFrame()

    def get_market_data(self):
        """Pobiera dane Spot i Makro (Cena, Wolumen, DXY, NASDAQ) z Yahoo Finance"""
        print("-> Pobieranie danych z Yahoo Finance...")
        tickers = {
            'BTC-USD': 'BTC_Price',
            '^IXIC': 'NASDAQ_100',
            'DX-Y.NYB': 'DXY_Index'
        }

        # Pobieranie danych (używamy daty buforowej)
        data = yf.download(list(tickers.keys()), start=FETCH_START_DATE, end=END_DATE)

        # Ekstrakcja cen zamknięcia i wolumenu (tylko dla BTC)
        close_prices = data['Close'].rename(columns=tickers)
        btc_volume = data['Volume'][['BTC-USD']].rename(columns={'BTC-USD': 'BTC_Volume'})

        # Łączenie w jeden DataFrame
        market_df = pd.concat([close_prices, btc_volume], axis=1)
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
        from pytrends.request import TrendReq  # Importujemy lokalnie, by zachować porządek

        try:
            # Inicjalizacja klienta. Używamy strefy czasowej UTC (tz=0)
            pytrends = TrendReq(hl='en-US', tz=0)

            # Słowo kluczowe
            kw_list = ["Bitcoin"]

            # Budowa zapytania z naszymi datami z konfiguracji
            timeframe = f"{FETCH_START_DATE} {END_DATE}"
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')

            # Pobranie danych
            trends_df = pytrends.interest_over_time()

            if trends_df.empty:
                print("[!] Google Trends zwróciło pusty zbiór. Być może nałożono limit (Rate Limit).")
                return pd.DataFrame()

            # Usunięcie zbędnej kolumny systemowej 'isPartial' jeśli istnieje
            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])

            trends_df.columns = ['Google_Trends_BTC']
            trends_df.index = pd.to_datetime(trends_df.index).normalize()

            # SKALOWANIE: Google zwraca dane tygodniowe.
            # Rozbijamy je na codzienne (resample) i kopiujemy wartość na kolejne dni tygodnia (ffill)
            daily_trends = trends_df.resample('D').ffill()
            return daily_trends

        except Exception as e:
            print(f"[!] Błąd API Google Trends (najczęściej blokada IP/Too Many Requests): {e}")
            return pd.DataFrame()

    def get_bitmex_funding(self, start_date, end_date):
        """Pobiera historię Funding Rate z giełdy BitMEX w oparciu o silnik Pandas"""
        print("-> Pobieranie historii Funding Rate z BitMEX...")
        all_funding_dfs = []

        # Pandas automatycznie radzi sobie z formatami dat
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

                # Konwersja paczki danych prosto do Pandas DataFrame
                temp_df = pd.DataFrame(data)

                # Zmiana tekstowych timestampów na obiekty daty i usunięcie strefy czasowej (dla kompatybilności z Binance)
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']).dt.tz_localize(None)

                # Wektorowe odfiltrowanie danych nowszych niż nasza data końcowa
                filtered_df = temp_df[temp_df['timestamp'] < end_dt]

                if not filtered_df.empty:
                    all_funding_dfs.append(filtered_df)

                # Wyciągnięcie ostatniego czasu jako punktu startowego do kolejnej pętli
                last_ts = temp_df['timestamp'].iloc[-1]

                if last_ts <= current_start or len(data) < 500:
                    break

                current_start = last_ts + pd.Timedelta(seconds=1)
                time.sleep(1.5)  # Opóźnienie zapobiegające błędom 429

            except Exception as e:
                print(f"[!] Błąd przetwarzania danych BitMEX: {e}")
                break

        if not all_funding_dfs:
            print("[!] Nie udało się pobrać żadnych danych z BitMEX.")
            return pd.DataFrame()

        # Połączenie wszystkich małych tabelek w jedną wielką
        df = pd.concat(all_funding_dfs, ignore_index=True)
        df['date'] = df['timestamp'].dt.normalize()

        # Wyciągamy ostatni odczyt z każdego dnia
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

        # Binance odpaliło futures 10 września 2019
        bitmex_df = self.get_bitmex_funding(FETCH_START_DATE, '2019-09-10')
        binance_df = self.get_binance_funding('2019-09-10', END_DATE)

        combined_df = pd.concat([bitmex_df, binance_df])
        combined_df.columns = ['Funding_Rate_Last']

        # Zabezpieczenie na wypadek dubli przy zmianie giełdy
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        return combined_df

    def get_onchain_data(self):
        """Pobiera historyczne dane On-chain z Blockchain.info API"""
        print("-> Pobieranie danych On-Chain (Hashrate, Trudność, Adresy, Opłaty)...")

        # Używamy tylko stabilnych, działających endpointów
        charts = {
            'hash-rate': 'Hashrate',
            'difficulty': 'Difficulty',
            'n-unique-addresses': 'Unique_Addresses',
            'transaction-fees': 'Total_Fees_BTC'
        }

        onchain_df = pd.DataFrame()

        for endpoint, col_name in charts.items():
            url = f"https://api.blockchain.info/charts/{endpoint}?timespan=all&sampled=true&format=json"

            try:
                response_raw = requests.get(url)
                response = response_raw.json()
            except Exception as e:
                print(f"[!] Błąd połączenia dla {col_name}: {e}")
                continue

            if 'values' not in response:
                print(f"[!] Pominięto '{col_name}': API odrzuciło zapytanie.")
                continue

            temp_dict = {datetime.fromtimestamp(item['x']).strftime('%Y-%m-%d'): item['y'] for item in response['values']}
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[col_name])
            temp_df.index = pd.to_datetime(temp_df.index).normalize()

            if onchain_df.empty:
                onchain_df = temp_df
            else:
                onchain_df = onchain_df.join(temp_df, how='outer')

        return onchain_df

    def build_dataset(self):
        """Uruchamia funkcje, łączy tabele i przygotowuje dane pod inżynierię cech"""
        print("\nROZPOCZYNAM INTEGRACJĘ DANYCH...")

        market = self.get_market_data()
        macro = self.get_macro_data()
        fng = self.get_fear_greed()
        funding = self.get_combined_funding()
        onchain = self.get_onchain_data()
        trends = self.get_google_trends()  # <--- NOWA METODA

        print("\n-> Łączenie zbiorów danych (Merging)...")
        dfs = [market, macro, trends, fng, funding, onchain]  # <--- DODANO DO LISTY
        self.df_main = dfs[0].join(dfs[1:], how='outer')

        # 1. Filtrowanie do daty pobierania buforowego (2018 rok)
        self.df_main = self.df_main[self.df_main.index >= FETCH_START_DATE]
        self.df_main = self.df_main[self.df_main.index <= END_DATE]

        # 2. ZABEZPIECZENIE: Usuwanie wiersza dla wczoraj, jeśli Yahoo nie dało jeszcze ceny
        self.df_main.dropna(subset=['BTC_Price'], inplace=True)

        print("-> Rozwiązywanie problemu brakujących danych w weekendy (Forward Fill)...")
        self.df_main.ffill(inplace=True)

        # ==========================================
        # TUTAJ BĘDZIE MIEJSCE NA INŻYNIERIĘ CECH (Wskaźniki techniczne z użyciem bufora)
        # ==========================================

        print(f"-> Przycinanie zbioru do docelowej daty startowej modelu: {MODEL_START_DATE}...")
        # 3. Odcięcie "brudnego" okresu rozgrzewkowego. Zostawiamy czysty ML dataset.
        # self.df_main = self.df_main[self.df_main.index >= MODEL_START_DATE] # <--- USUŃ TĘ LINIĘ
        # self.df_main.dropna(inplace=True) # <--- USUŃ TĘ LINIĘ

        print("\nGOTOWE! Zestawienie pierwszych 5 rekordów dla Modelu:")
        print(self.df_main.head())

        filename = 'btc_ensemble_features.csv'
        self.df_main.to_csv(filename)
        print(f"\nZapisano pełny zbiór danych do pliku: {filename}")

        return self.df_main


# ==========================================
# URUCHOMIENIE SKRYPTU
# ==========================================
if __name__ == "__main__":
    integrator = BitcoinDataIntegrator()
    dataset = integrator.build_dataset()