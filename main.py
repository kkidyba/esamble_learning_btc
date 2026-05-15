import pandas as pd
import yfinance as yf
from fredapi import Fred
import requests
from datetime import datetime
import time

# ==========================================
# KONFIGURACJA GŁÓWNA
# ==========================================
FRED_API_KEY = 'f3ac7094f956fdb519c4f98c2453e476'  # Zarejestruj się na fred.stlouisfed.org
START_DATE = '2019-01-01'
END_DATE = '2026-05-01'


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

        # Pobieranie danych
        data = yf.download(list(tickers.keys()), start=START_DATE, end=END_DATE)

        # Ekstrakcja cen zamknięcia i wolumenu (tylko dla BTC)
        close_prices = data['Close'].rename(columns=tickers)
        btc_volume = data['Volume'][['BTC-USD']].rename(columns={'BTC-USD': 'BTC_Volume'})

        # Łączenie w jeden DataFrame
        market_df = pd.concat([close_prices, btc_volume], axis=1)
        market_df.index = pd.to_datetime(market_df.index).normalize()  # Normalizacja daty (usunięcie godzin)
        return market_df

    def get_macro_data(self):
        """Pobiera podaż pieniądza M2 z FRED"""
        print("-> Pobieranie podaży pieniądza M2 z FRED...")
        try:
            m2_data = self.fred.get_series('M2SL', observation_start=START_DATE)
            m2_df = pd.DataFrame(m2_data, columns=['M2_Supply'])
            m2_df.index = pd.to_datetime(m2_df.index).normalize()
            return m2_df
        except Exception as e:
            print(f"Błąd FRED API: Zastąp 'TUTAJ_WKLEJ_SWOJ_KLUCZ_FRED' prawidłowym kluczem. ({e})")
            return pd.DataFrame()

    def get_fear_greed(self):
        """Pobiera Fear & Greed Index z API Alternative.me"""
        print("-> Pobieranie Sentymentu (Fear & Greed Index)...")

        # ZMIANA TUTAJ: Usunięto parametr date_format z końca URL
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url).json()

        fng_data = {}
        for item in response['data']:
            # Teraz API zwraca czysty Unix Timestamp, więc int() zadziała bez błędu
            date_str = datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d')
            fng_data[date_str] = int(item['value'])

        fng_df = pd.DataFrame.from_dict(fng_data, orient='index', columns=['Fear_Greed_Index'])
        fng_df.index = pd.to_datetime(fng_df.index).normalize()
        return fng_df

    def get_binance_funding(self):
        """Pobiera historyczny Funding Rate z Binance (Public API) omijając limity"""
        print("-> Pobieranie pełnej historii Funding Rate z Binance...")
        all_funding = []

        # Konwersja daty startowej na milisekundy (wymagane przez Binance)
        start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)

        while True:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': 'BTCUSDT', 'limit': 1000, 'startTime': start_ts}
            response = requests.get(url, params=params).json()

            if not response or len(response) == 0:
                break

            all_funding.extend(response)

            # Pobranie czasu z ostatniego rekordu jako nowy punkt startowy
            last_ts = response[-1]['fundingTime']
            if last_ts <= start_ts:
                break

            start_ts = last_ts + 1
            time.sleep(0.2)  # Zabezpieczenie przed zablokowaniem IP (Rate Limiting)

        # Konwersja listy słowników na DataFrame
        funding_df = pd.DataFrame(all_funding)
        funding_df['date'] = pd.to_datetime(funding_df['fundingTime'], unit='ms').dt.normalize()
        funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)

        # ZMIANA: Zamiast .mean() używamy .last(), aby wziąć ostatni odczyt z danego dnia
        daily_funding = funding_df.groupby('date')['fundingRate'].last().to_frame()

        # ZMIANA NAZWY KOLUMNY DLA PORZĄDKU
        daily_funding.columns = ['Funding_Rate_Last']

        return daily_funding

    def get_binance_open_interest(self):
        """Pobiera historyczny Open Interest z Binance Futures"""
        print("-> Pobieranie historii Open Interest z Binance...")
        all_oi = []

        # Konwersja daty startowej
        start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)

        while True:
            # POPRAWKA 1: Prawidłowy adres URL dla historycznych danych OI
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {
                'symbol': 'BTCUSDT',
                'period': '1d',
                'limit': 500,
                'startTime': start_ts
            }

            # Pobranie surowej odpowiedzi
            response_raw = requests.get(url, params=params)

            # POPRAWKA 2: Zabezpieczenie przed błędem serwera (np. Rate Limiting)
            if response_raw.status_code != 200:
                print(
                    f"Błąd API Binance OI: Giełda zwróciła kod {response_raw.status_code}. Treść: {response_raw.text}")
                break

            response = response_raw.json()

            if not response or type(response) != list or len(response) == 0:
                break

            all_oi.extend(response)

            last_ts = response[-1]['timestamp']
            if last_ts <= start_ts:
                break

            # POPRAWKA 3: Zwiększenie opóźnienia do 0.5s dla endpointów data/
            start_ts = last_ts + 1
            time.sleep(0.5)

        # Zabezpieczenie, jeśli lista jest pusta
        if not all_oi:
            print("Nie udało się pobrać danych Open Interest.")
            return pd.DataFrame()

        oi_df = pd.DataFrame(all_oi)
        oi_df['date'] = pd.to_datetime(oi_df['timestamp'], unit='ms').dt.normalize()

        # Pobieramy zarówno wartość w sztukach BTC, jak i w dolarach (USD)
        oi_df['Open_Interest_BTC'] = oi_df['sumOpenInterest'].astype(float)
        oi_df['Open_Interest_USD'] = oi_df['sumOpenInterestValue'].astype(float)

        # Grupujemy (na wypadek duplikatów) i ustawiamy indeks daty
        daily_oi = oi_df.groupby('date')[['Open_Interest_BTC', 'Open_Interest_USD']].last()
        return daily_oi

    def get_binance_long_short_ratio(self):
        """Pobiera historyczny Global Long/Short Account Ratio z Binance Futures"""
        print("-> Pobieranie historii Long/Short Ratio z Binance...")
        all_ls_ratio = []

        # Konwersja daty startowej na timestamp
        start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)

        while True:
            # Endpoint dla historii Long/Short Ratio
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': 'BTCUSDT',
                'period': '1d',  # Świece dzienne
                'limit': 500,  # Max limit to 500
                'startTime': start_ts
            }

            response_raw = requests.get(url, params=params)

            if response_raw.status_code != 200:
                print(f"Błąd API Binance L/S Ratio: Kod {response_raw.status_code}. Treść: {response_raw.text}")
                break

            response = response_raw.json()

            if not response or type(response) != list or len(response) == 0:
                break

            all_ls_ratio.extend(response)

            last_ts = response[-1]['timestamp']
            if last_ts <= start_ts:
                break

            start_ts = last_ts + 1
            time.sleep(0.5)  # Bezpieczne opóźnienie dla endpointów analitycznych

        if not all_ls_ratio:
            print("Nie udało się pobrać danych Long/Short Ratio.")
            return pd.DataFrame()

        ls_df = pd.DataFrame(all_ls_ratio)
        ls_df['date'] = pd.to_datetime(ls_df['timestamp'], unit='ms').dt.normalize()

        # Pobieramy główny wskaźnik proporcji (wartość > 1 oznacza więcej Longów, < 1 więcej Shortów)
        ls_df['Long_Short_Ratio'] = ls_df['longShortRatio'].astype(float)

        # Opcjonalnie: pobieramy też dokładne udziały procentowe
        ls_df['Long_Account_Pct'] = ls_df['longAccount'].astype(float)
        ls_df['Short_Account_Pct'] = ls_df['shortAccount'].astype(float)

        daily_ls = ls_df.groupby('date')[['Long_Short_Ratio', 'Long_Account_Pct', 'Short_Account_Pct']].last()
        return daily_ls

    def get_onchain_data(self):
        """Pobiera historyczne dane On-chain z Blockchain.info API"""
        print("-> Pobieranie danych On-Chain (Hashrate, Trudność, Adresy, Opłaty)...")

        # Oczyszczony słownik - tylko stabilne i dostępne publicznie endpointy
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

            # Zabezpieczenie przed błędną odpowiedzią z serwera
            if 'values' not in response:
                print(f"[!] Pominięto '{col_name}': API odrzuciło zapytanie.")
                continue  # To ta kluczowa instrukcja, która zapobiega KeyError!

            # Ekstrakcja wartości
            temp_dict = {datetime.fromtimestamp(item['x']).strftime('%Y-%m-%d'): item['y'] for item in
                         response['values']}
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[col_name])
            temp_df.index = pd.to_datetime(temp_df.index).normalize()

            if onchain_df.empty:
                onchain_df = temp_df
            else:
                onchain_df = onchain_df.join(temp_df, how='outer')

        return onchain_df

    def build_dataset(self):
        """Uruchamia wszystkie funkcje, łączy tabele i czyści dane"""
        print("\nROZPOCZYNAM INTEGRACJĘ DANYCH...")

        market = self.get_market_data()
        macro = self.get_macro_data()
        fng = self.get_fear_greed()
        funding = self.get_binance_funding()
        onchain = self.get_onchain_data()

        # UWAGA: Wykluczamy z automatycznego API Open Interest oraz L/S Ratio.
        # Z powodu limitu 30 dni w darmowym REST API, zrujnowałoby to nasz zbiór danych
        # (funkcja dropna usunęłaby wszystkie wiersze z lat 2019-2026).
        # Te dwie specyficzne metryki dla lat 2019+ należy pobrać ręcznie jako pliki CSV z data.binance.vision.

        print("\n-> Łączenie zbiorów danych (Merging)...")
        dfs = [market, macro, fng, funding, onchain]
        self.df_main = dfs[0].join(dfs[1:], how='outer')

        # Filtrowanie dat od września 2019
        self.df_main = self.df_main[self.df_main.index >= START_DATE]
        self.df_main = self.df_main[self.df_main.index <= END_DATE]

        print("-> Rozwiązywanie problemu brakujących danych (Forward Fill)...")
        self.df_main.ffill(inplace=True)

        # Usuwanie wierszy, które mogły pozostać puste na samym początku zbioru
        self.df_main.dropna(inplace=True)

        print("\nGOTOWE! Zestawienie pierwszych 5 rekordów:")
        print(self.df_main.head())

        # Zapis do pliku
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