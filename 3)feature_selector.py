import pandas as pd


class FeatureSelector:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        print(f"Załadowano dane z '{file_path}'. Początkowy wymiar: {self.df.shape}")

    def remove_highly_correlated_features(self, threshold: float = 0.85,
                                          output_file: str = "btc_ml_features_cleaned.csv"):

        corr_matrix = self.df.corr().abs()

        to_drop = set()
        drop_details = []

        columns = corr_matrix.columns
        for i in range(len(columns)):
            col_i = columns[i]

            if col_i in to_drop:
                continue

            for j in range(i + 1, len(columns)):
                col_j = columns[j]

                if col_j in to_drop:
                    continue

                if corr_matrix.iloc[i, j] > threshold:
                    to_drop.add(col_j)
                    correlation_value = round(corr_matrix.iloc[i, j], 3)
                    drop_details.append(f"{col_j} (skorelowana z: {col_i} na poziomie {correlation_value})")

        print(f"\n--- RAPORT REDUKCJI KORELACJI (Próg > {threshold}) ---")
        if not to_drop:
            print("Nie znaleziono cech o zbyt wysokiej korelacji. Nic nie usunięto.")
        else:
            print(f"Znaleziono {len(to_drop)} nadmiarowych cech do usunięcia:\n")
            for detail in sorted(drop_details):
                print(f"[-] USUNIĘTO: {detail}")

        df_cleaned = self.df.drop(columns=list(to_drop))

        df_cleaned.to_csv(output_file)
        print(f"\nSukces! Zapisano oczyszczony zbiór.")
        print(f"Ilość cech przed: {self.df.shape[1]}")
        print(f"Ilość cech po:   {df_cleaned.shape[1]}")
        print(f"Plik wyjściowy:  {output_file}")

        return df_cleaned


if __name__ == "__main__":
    selector = FeatureSelector("btc_ml_features.csv")
    df_clean = selector.remove_highly_correlated_features(threshold=0.85)
    print("!!!Double check!!!")
    selector = FeatureSelector("btc_ml_features_cleaned.csv")
    df_clean = selector.remove_highly_correlated_features(threshold=0.85)