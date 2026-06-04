import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DCAPortfolioSimulator:
    def __init__(self, initial_capital=1000.0, dca_amount=1000.0, fee_rate=0.001, mar=0.0, leverage=0):
        """
        Uniwersalny symulator portfela dla strategii kryptowalutowych.

        Parametry:
        - initial_capital: Kapitał startowy w USD (domyślnie 1000$).
        - dca_amount: Kwota dodawana do portfela przy zmianie miesiąca (domyślnie 1000$).
        - fee_rate: Koszt transakcyjny + slippage (np. 0.001 to 0.1%).
        - mar: Minimum Acceptable Return dla Sortino (dzienna stopa zwrotu, domyślnie 0.0).
        - leverage: Dźwignia finansowa.
            0 -> rynek spot (long, brak shortów).
            1 -> rynek futures (long i short na dźwigni 1x).
            2 -> rynek futures (long i short na dźwigni 2x) itd.
        """
        self.initial_capital = initial_capital
        self.dca_amount = dca_amount
        self.fee_rate = fee_rate
        self.mar = mar
        self.leverage = leverage

    def simulate(self, dates, prices, signals, execution_prices=None):
        """
        Główna pętla symulacyjna. Zwraca wyniki i dane potrzebne do wizualizacji.
        """
        prices_arr = np.asarray(prices)
        signals_arr = np.asarray(signals)
        exec_prices_arr = np.asarray(execution_prices) if execution_prices is not None else prices_arr

        dates_dt = pd.to_datetime(dates)
        if isinstance(dates_dt, pd.DatetimeIndex):
            months = dates_dt.month.values
        else:
            months = dates_dt.dt.month.values

        n = len(prices_arr)

        equity_curve = np.zeros(n)
        true_returns = np.zeros(n)
        btc_balances_arr = np.zeros(n)

        usd_balance = float(self.initial_capital)
        btc_balance = 0.0
        entry_price = 0.0
        current_state = 0  # 1: Long, 0: Flat, -1: Short

        for i in range(n):
            current_val_price = float(prices_arr[i])
            current_exec_price = float(exec_prices_arr[i])
            dca_today = 0.0

            active_signal = signals_arr[i - 1] if i > 0 else np.nan

            # 1. Sprawdzenie warunku wpłaty DCA
            if i > 0 and months[i] != months[i - 1]:
                dca_today = float(self.dca_amount)
                usd_balance += dca_today

            # 2. Egzekucja transakcji (sygnał T-1, cena T)
            if pd.isna(active_signal):
                pass
            elif self.leverage == 0:
                # --- RYNEK SPOT ---
                if active_signal == 1.0:
                    if usd_balance > 0:
                        btc_bought = (usd_balance * (1.0 - self.fee_rate)) / current_exec_price
                        btc_balance += btc_bought
                        usd_balance = 0.0
                        current_state = 1
                elif active_signal == 0.0:
                    if btc_balance > 0:
                        usd_received = (btc_balance * current_exec_price) * (1.0 - self.fee_rate)
                        usd_balance += usd_received
                        btc_balance = 0.0
                        current_state = 0
            else:
                # --- RYNEK FUTURES (DŹWIGNIA) ---
                target_state = 1 if active_signal == 1.0 else -1

                # MTM: Obliczanie aktualnego kapitału uwzględniającego zyski/straty z otwartej pozycji
                exec_equity = usd_balance + btc_balance * (current_exec_price - entry_price)

                # Weryfikacja ewentualnej likwidacji pozycji (kapitał MTM spada poniżej zera)
                if exec_equity <= 0:
                    usd_balance = 0.0
                    btc_balance = 0.0
                    entry_price = 0.0
                    current_state = 0
                    exec_equity = 0.0

                if exec_equity > 0:
                    if current_state != target_state:
                        # a) Zamykamy dotychczasową pozycję (jeśli istnieje)
                        if btc_balance != 0:
                            # Realizacja PnL (dodajemy zysk/zdejmujemy stratę do marginu)
                            usd_balance += btc_balance * (current_exec_price - entry_price)
                            close_fee = abs(btc_balance) * current_exec_price * self.fee_rate
                            usd_balance -= close_fee
                            btc_balance = 0.0
                            entry_price = 0.0

                        # Aktualizacja dostępnego kapitału operacyjnego po zamknięciu
                        exec_equity = usd_balance

                        # b) Otwieramy nową pozycję w zgodzie z nowym sygnałem
                        if exec_equity > 0:
                            position_value_usd = exec_equity * self.leverage
                            target_btc = position_value_usd / current_exec_price

                            if target_state == -1:
                                target_btc = -target_btc

                            open_fee = abs(target_btc) * current_exec_price * self.fee_rate
                            usd_balance -= open_fee
                            btc_balance = target_btc
                            entry_price = current_exec_price
                            current_state = target_state

                    else:
                        # Stan się nie zmienia. Jeśli jednak wpadło nowe DCA, dorzucamy do pozycji,
                        # aby zachować stały docelowy poziom lewarowania kapitału.
                        if dca_today > 0 and current_state != 0:
                            extra_position_value = dca_today * self.leverage
                            extra_btc = extra_position_value / current_exec_price

                            if current_state == -1:
                                extra_btc = -extra_btc

                            open_fee = abs(extra_btc) * current_exec_price * self.fee_rate
                            usd_balance -= open_fee

                            # Aktualizacja średniej ceny wejścia dla zrebalansowanej pozycji
                            total_btc = btc_balance + extra_btc
                            if total_btc != 0:
                                old_cost = btc_balance * entry_price
                                new_cost = extra_btc * current_exec_price
                                entry_price = (old_cost + new_cost) / total_btc

                            btc_balance = total_btc

            # Zapisanie aktualnego rozmiaru pozycji (na minusie dla shorta)
            btc_balances_arr[i] = btc_balance

            # 3. Wycena aktualnego portfela (Mark-to-Market na koniec dnia)
            if self.leverage == 0:
                current_equity = usd_balance + (btc_balance * current_val_price)
            else:
                current_equity = usd_balance + btc_balance * (current_val_price - entry_price)
                if current_equity < 0:
                    current_equity = 0.0  # Zero kapitału = likwidacja, wykres nie zejdzie poniżej 0

            equity_curve[i] = current_equity

            # 4. Obliczenie prawdziwej stopy zwrotu
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
            'Returns': true_returns,
            'BTC_Balances': btc_balances_arr,
            'Dates': dates_dt,
            'Prices': prices_arr
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

    def _calculate_pure_dca_equity(self, dates_dt, prices_arr, execution_prices=None):
        """Wektoryzowana funkcja generująca benchmark Czystego DCA i skumulowane wpłaty."""
        exec_prices_arr = np.asarray(execution_prices) if execution_prices is not None else prices_arr

        if isinstance(dates_dt, pd.DatetimeIndex):
            months = dates_dt.month.values
        else:
            months = dates_dt.dt.month.values

        n = len(prices_arr)
        btc_purchases = np.zeros(n)
        fiat_invested_step = np.zeros(n)

        if n > 0:
            btc_purchases[0] = (self.initial_capital * (1.0 - self.fee_rate)) / exec_prices_arr[0]
            fiat_invested_step[0] = self.initial_capital

        if n > 1:
            month_changes = months[1:] != months[:-1]
            change_indices = np.where(month_changes)[0] + 1

            btc_purchases[change_indices] = (self.dca_amount * (1.0 - self.fee_rate)) / exec_prices_arr[change_indices]
            fiat_invested_step[change_indices] = self.dca_amount

        btc_balance_curve = np.cumsum(btc_purchases)
        fiat_invested_curve = np.cumsum(fiat_invested_step)
        pure_dca_equity = btc_balance_curve * prices_arr

        return pure_dca_equity, fiat_invested_curve

    def plot_simulation_results(self, results, execution_prices=None):
        """Generuje kompleksowy wykres analizy portfela."""
        dates = results['Dates']
        prices = results['Prices']
        equity_curve = results['Equity_Curve']
        btc_balances = results['BTC_Balances']

        pure_dca_equity, fiat_invested_curve = self._calculate_pure_dca_equity(dates, prices, execution_prices)

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # --- LEWA OŚ: CENA BTC ---
        color_btc = '#FF9900'
        ax1.set_xlabel('Data', fontsize=12)
        ax1.set_ylabel('Cena BTC (USD) - Skala Logarytmiczna', color=color_btc, fontsize=12)
        line_btc = ax1.plot(dates, prices, color=color_btc, label='Cena BTC', linewidth=1.5, alpha=0.6)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color_btc)

        fig.autofmt_xdate()

        # --- PRAWA OŚ: PORTFELE W USD ---
        ax2 = ax1.twinx()
        color_sim = '#1f77b4'
        color_dca = '#2ca02c'
        color_fiat = '#7f7f7f'

        ax2.set_ylabel('Wartość Portfela / Kapitał (USD)', color='black', fontsize=12)

        line_sim = ax2.plot(dates, equity_curve, color=color_sim, label='Symulowany Portfel (Strategia)', linewidth=2.5)
        line_dca = ax2.plot(dates, pure_dca_equity, color=color_dca, label='Czyste DCA (Benchmark)', linewidth=2,
                            linestyle='--')
        line_fiat = ax2.plot(dates, fiat_invested_curve, color=color_fiat, label='Skumulowane Wpłaty (Fiat)',
                             linewidth=1.5, drawstyle='steps-post')

        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(bottom=0)

        # --- TŁO: OKRESY POZA RYNKIEM ---
        out_of_market = btc_balances == 0

        fill_bg = ax1.fill_between(dates, ax1.get_ylim()[0], ax1.get_ylim()[1],
                                   where=out_of_market, facecolor='red', alpha=0.1,
                                   label='Poza rynkiem (100% USD)')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10, framealpha=0.9)

        # Dynamiczny tytuł zależny od ustawień dźwigni
        title_suffix = f'(Dźwignia: {self.leverage}x)' if self.leverage > 0 else '(Spot)'
        plt.title(f'Wyniki Symulacji Strategii vs Benchmark Czystego DCA {title_suffix}', fontsize=16, pad=15)
        plt.grid(True, which='major', axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()