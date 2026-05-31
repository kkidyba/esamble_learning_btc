import numpy as np
import pandas as pd


class DCAPortfolioSimulator:
    def __init__(self, initial_capital=1000.0, dca_amount=1000.0, fee_rate=0.001, mar=0.0):
        """
        Uniwersalny symulator portfela dla strategii kryptowalutowych.

        Parametry:
        - initial_capital: Kapitał startowy w USD (domyślnie 1000$).
        - dca_amount: Kwota dodawana do portfela przy zmianie miesiąca (domyślnie 1000$).
        - fee_rate: Koszt transakcyjny + slippage (np. 0.001 to 0.1%).
        - mar: Minimum Acceptable Return dla Sortino (dzienna stopa zwrotu, domyślnie 0.0).
        """
        self.initial_capital = initial_capital
        self.dca_amount = dca_amount
        self.fee_rate = fee_rate
        self.mar = mar

    def simulate(self, dates, prices, signals, execution_prices=None):
        """
        Główna pętla symulacyjna. (Poprawka: Usunięto Look-Ahead Bias).

        Parametry:
        - dates: Daty dla identyfikacji przełomu miesiąca (DCA).
        - prices: Ceny zamknięcia do codziennej wyceny portfela (Mark-to-Market).
        - signals: Sygnały wygenerowane dla danych z danego dnia.
        - execution_prices: Ceny, po których faktycznie realizujemy zlecenia (np. Open).
          Jeśli None, system zrealizuje zlecenie po cenie z 'prices' (T+1).
        """
        prices_arr = np.asarray(prices)
        signals_arr = np.asarray(signals)

        # Jeśli nie podano osobnych cen egzekucji (np. danych Open),
        # używamy głównej tablicy cen, ale transakcja i tak będzie opóźniona o 1 okres.
        exec_prices_arr = np.asarray(execution_prices) if execution_prices is not None else prices_arr

        dates_dt = pd.to_datetime(dates)
        if isinstance(dates_dt, pd.DatetimeIndex):
            months = dates_dt.month.values
        else:
            months = dates_dt.dt.month.values

        n = len(prices_arr)

        equity_curve = np.zeros(n)
        true_returns = np.zeros(n)

        usd_balance = float(self.initial_capital)
        btc_balance = 0.0

        for i in range(n):
            current_val_price = float(prices_arr[i])
            current_exec_price = float(exec_prices_arr[i])
            dca_today = 0.0

            # --- POPRAWKA: LOOK-AHEAD BIAS ---
            # Decyzję podejmujemy na podstawie sygnału z dnia wczorajszego (i-1).
            # W dniu i=0 nie mamy sygnału z przeszłości, więc nie wykonujemy żadnych akcji.
            active_signal = signals_arr[i - 1] if i > 0 else np.nan
            # ---------------------------------

            # 1. Sprawdzenie warunku wpłaty DCA (początek miesiąca)
            if i > 0 and months[i] != months[i - 1]:
                dca_today = float(self.dca_amount)
                usd_balance += dca_today

            # 2. Egzekucja transakcji w oparciu o aktywny sygnał (z T-1) i cenę egzekucji z dzisiaj (T)
            if pd.isna(active_signal):
                pass

            elif active_signal == 1.0:
                if usd_balance > 0:
                    btc_bought = (usd_balance * (1.0 - self.fee_rate)) / current_exec_price
                    btc_balance += btc_bought
                    usd_balance = 0.0

            elif active_signal == 0.0:
                if btc_balance > 0:
                    usd_received = (btc_balance * current_exec_price) * (1.0 - self.fee_rate)
                    usd_balance += usd_received
                    btc_balance = 0.0

            # 3. Wycena aktualnego portfela (Mark-to-Market) po dzisiejszej cenie zamknięcia
            current_equity = usd_balance + (btc_balance * current_val_price)
            equity_curve[i] = current_equity

            # 4. Obliczenie prawdziwej stopy zwrotu (bez wpływu wkładu DCA)
            if i > 0:
                prev_equity = equity_curve[i - 1]
                if prev_equity > 0:
                    true_returns[i] = (current_equity - dca_today) / prev_equity - 1.0
                else:
                    true_returns[i] = 0.0
            else:
                true_returns[i] = 0.0

        valid_returns = true_returns[1:]
        sortino = self._calculate_sortino(valid_returns)

        total_invested = self.initial_capital + (sum(months[1:] != months[:-1]) * self.dca_amount)
        final_equity = equity_curve[-1]
        roi = (final_equity / total_invested) - 1.0 if total_invested > 0 else 0

        return {
            'Sortino': sortino,
            'ROI': roi,
            'Final_Equity': final_equity,
            'Total_Invested': total_invested,
            'Equity_Curve': equity_curve,
            'Returns': true_returns
        }

    def _calculate_sortino(self, returns_array):
        """Kalkulacja wskaźnika Sortino z parametrem MAR."""
        if len(returns_array) == 0:
            return 0.0

        excess_returns = returns_array - self.mar
        mean_return = np.mean(excess_returns)

        downside_returns = np.minimum(0, excess_returns)
        downside_variance = np.mean(downside_returns ** 2)

        if downside_variance == 0:
            return 99.9 if mean_return > 0 else 0.0

        downside_deviation = np.sqrt(downside_variance)
        sortino = (mean_return / downside_deviation) * np.sqrt(365)
        return float(sortino)