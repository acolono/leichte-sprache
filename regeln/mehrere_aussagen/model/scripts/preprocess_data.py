"""
Data Preprocessing Pipeline für StaGE Statement Segmentation
Bereitet die CSV-Daten für BERT-Klassifikation vor.

Autor: Senior Data Scientist
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


import logging

logger = logging.getLogger(__name__)
class StaGEDataPreprocessor:
    """
    Preprocessor für StaGE Statement Segmentation Dataset.
    Transformiert CSV-Daten in BERT-kompatibles Format.
    """

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)

    def load_data(self, filename: str = "train.csv") -> pd.DataFrame:
        """Lädt CSV-Daten und bereinigt sie."""
        filepath = self.data_dir / filename
        logger.info(" Lade Daten von: %s", filepath)

        df = pd.read_csv(filepath)
        logger.info(" %s Einträge geladen", len(df))
        logger.info(" Spalten: %s", df.columns.tolist())

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt und validiert die Daten."""
        logger.info("\n Bereinige Daten...")

        # Entferne Zeilen mit fehlenden Werten
        df_clean = df.dropna(subset=["phrase", "num_statements"]).copy()

        # Konvertiere num_statements zu int
        df_clean["num_statements"] = df_clean["num_statements"].astype(int)

        # Filtere ungültige Werte (negative oder zu hohe Werte)
        df_clean = df_clean[df_clean["num_statements"] > 0]
        df_clean = df_clean[df_clean["num_statements"] <= 10]  # Max 10 Statements

        # Bereinige Textfeld
        df_clean["phrase"] = df_clean["phrase"].apply(self._clean_text)

        logger.info(" %s saubere Einträge (%s entfernt)", len(df_clean), len(df) - len(df_clean))

        return df_clean

    def _clean_text(self, text: str) -> str:
        """Bereinigt einen einzelnen Text."""
        if not isinstance(text, str):
            return ""

        # Entferne LaTeX-Befehle wie \newline
        text = re.sub(r"\\newline", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # Normalisiere Whitespace
        text = " ".join(text.split())

        return text.strip()

    def create_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt binäre Labels: 0 = ein Statement, 1 = mehrere Statements.
        Dies ist die Hauptaufgabe für unsere Regel.
        """
        logger.info("\n Erstelle binäre Labels...")

        df_labeled = df.copy()
        df_labeled["has_multiple_statements"] = (
            df_labeled["num_statements"] > 1
        ).astype(int)

        # Statistiken
        single_statements = (df_labeled["has_multiple_statements"] == 0).sum()
        multiple_statements = (df_labeled["has_multiple_statements"] == 1).sum()

        logger.info(" Labels erstellt:")
        logger.info(" Ein Statement: %s (%.1f%)", single_statements, single_statements / len(df_labeled) * 100)
        logger.info(" Mehrere Statements: %s (%.1f%)", multiple_statements, multiple_statements / len(df_labeled) * 100)

        return df_labeled

    def create_multiclass_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt Multi-Class Labels für detailliertere Klassifikation.
        0 = 1 Statement, 1 = 2 Statements, 2 = 3 Statements, 3 = 4+ Statements
        """
        logger.info("\n Erstelle Multi-Class Labels...")

        df_labeled = df.copy()

        def map_to_class(num_statements):
            if num_statements == 1:
                return 0
            elif num_statements == 2:
                return 1
            elif num_statements == 3:
                return 2
            else:  # 4+
                return 3

        df_labeled["statement_class"] = df_labeled["num_statements"].apply(map_to_class)

        # Statistiken
        for i in range(4):
            count = (df_labeled["statement_class"] == i).sum()
            label = f"{i + 1}" if i < 3 else "4+"
            logger.info(" %s Statement(s): %s (%.1f%)", label, count, count / len(df_labeled) * 100)

        return df_labeled

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Teilt Daten in Train/Val/Test mit stratifizierter Sampling."""
        from sklearn.model_selection import train_test_split

        logger.info("\n Teile Daten: %.0% Train / %.0% Val / %.0% Test", train_ratio, val_ratio, test_ratio)

        # Stratify nach has_multiple_statements für ausgewogene Splits
        stratify_col = (
            "has_multiple_statements"
            if "has_multiple_statements" in df.columns
            else "num_statements"
        )

        # Erst Train + (Val+Test) Split
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=df[stratify_col],
        )

        # Dann Val + Test Split
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=random_state,
            stratify=temp_df[stratify_col],
        )

        logger.info(" Train: %s | Val: %s | Test: %s", len(train_df), len(val_df), len(test_df))

        return train_df, val_df, test_df

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analysiert das Dataset und gibt Statistiken zurück."""
        logger.info("\n DATASET ANALYSE")
        logger.info("=" * 60)

        analysis = {
            "total_samples": len(df),
            "num_statements_dist": df["num_statements"].value_counts().to_dict(),
            "avg_phrase_length": df["phrase"].str.len().mean(),
            "avg_word_count": df["phrase"].str.split().str.len().mean(),
            "unique_topics": df["topic"].nunique() if "topic" in df.columns else 0,
        }

        logger.info(" Gesamtzahl Samples: %s", analysis['total_samples'])
        logger.info(" Durchschnittliche Satzlänge: %.1f Zeichen", analysis['avg_phrase_length'])
        logger.info(" Durchschnittliche Wortanzahl: %.1f Wörter", analysis['avg_word_count'])

        if analysis["unique_topics"] > 0:
            logger.info(" Unique Topics: %s", analysis['unique_topics'])

        logger.info("\n Verteilung der Statement-Anzahl:")
        for num_stmts, count in sorted(analysis["num_statements_dist"].items()):
            pct = count / analysis["total_samples"] * 100
            logger.info(" %s Statement(s): %4d (%5.1f%)", num_stmts, count, pct)

        logger.info("=" * 60)

        return analysis

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "../data",
    ):
        """Speichert verarbeitete Daten."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("\n Speichere verarbeitete Daten...")

        train_df.to_csv(output_path / "train_processed.csv", index=False)
        val_df.to_csv(output_path / "val_processed.csv", index=False)
        test_df.to_csv(output_path / "test_processed.csv", index=False)

        logger.info(" Gespeichert in: %s", output_path)
        logger.info(" - train_processed.csv (%s samples)", len(train_df))
        logger.info(" - val_processed.csv (%s samples)", len(val_df))
        logger.info(" - test_processed.csv (%s samples)", len(test_df))


def main():
    """Hauptfunktion für Datenvorverarbeitung."""
    logger.info(" StaGE Statement Segmentation - Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # Initialisiere Preprocessor
    preprocessor = StaGEDataPreprocessor(data_dir="../data")

    # Lade und bereinige Daten
    df = preprocessor.load_data("train.csv")
    df_clean = preprocessor.clean_data(df)

    # Analysiere Dataset
    preprocessor.analyze_dataset(df_clean)

    # Erstelle Labels (Binary und Multi-Class)
    df_binary = preprocessor.create_binary_labels(df_clean)
    df_multiclass = preprocessor.create_multiclass_labels(df_binary)

    # Split Daten
    train_df, val_df, test_df = preprocessor.split_data(df_multiclass)

    # Speichere verarbeitete Daten
    preprocessor.save_processed_data(train_df, val_df, test_df)

    logger.info("\n Preprocessing abgeschlossen!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
