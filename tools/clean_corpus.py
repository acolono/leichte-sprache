# clean_corpus.py
# Comprehensive corpus cleaning script for Leichte-Sprache perplexity model training
# Removes parsing artifacts, normalizes text, and prepares clean training data

import re
import sys
import unicodedata
from pathlib import Path
from typing import Tuple


import logging

logger = logging.getLogger(__name__)
class CorpusCleaner:
    """
    Advanced corpus cleaning for German Leichte-Sprache text.
    Handles parsing artifacts, normalization, and German-specific text processing.
    """

    def __init__(self):
        # Compile regex patterns for efficiency
        # Pattern for parsing artifacts: -- _ $( ... )
        self.parsing_artifact_pattern = re.compile(r"-- _ \$\$\(.*?\)")

        # Alternative pattern that matches the actual format we found
        self.parsing_artifact_pattern_v2 = re.compile(r"-- _ \$\$\(.*?\)")

        # Simpler pattern to catch all variations: -- _ $( ... )
        self.parsing_artifact_pattern_v3 = re.compile(r"-- _ \$\$\(.*?\)")

        self.multiple_spaces_pattern = re.compile(r" +")
        self.german_quotes_pattern = re.compile(r'["„“]')
        self.special_chars_pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")

        # German-specific normalization rules
        self.german_normalization = {
            "ß": "ss",
            "„": '"',
            "“": '"',
            "–": "-",
            "—": "-",
            "…": "...",
            "´": "'",
            "`": "'",
            "‘": "'",
            "’": "'",
        }

    def _normalize_german_text(self, text: str) -> str:
        """
        Apply German-specific text normalization.
        """
        # Replace German-specific characters
        for old, new in self.german_normalization.items():
            text = text.replace(old, new)

        # Normalize Unicode (NFKC form)
        text = unicodedata.normalize("NFKC", text)

        return text

    def _clean_parsing_artifacts(self, text: str) -> str:
        """
        Remove dependency parsing artifacts from text.
        Patterns like: -- _ $( _ 0 ROOT _ _ 8 Urbi Urbi _ NE _ 0 S _ _ 9 et et _ NE _ 8 APP _ _ 10 orbi Orbi _ NE _ 9 APP _ _ 11 .
        """
        # Remove the parsing artifacts - try all patterns
        cleaned_text = text

        # Try all three patterns
        cleaned_text = self.parsing_artifact_pattern.sub("", cleaned_text)
        cleaned_text = self.parsing_artifact_pattern_v2.sub("", cleaned_text)
        cleaned_text = self.parsing_artifact_pattern_v3.sub("", cleaned_text)

        # Clean up any remaining artifacts or double spaces
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def _clean_sentence(self, sentence: str) -> str:
        """
        Clean a single sentence with comprehensive processing.
        """
        if not sentence or not sentence.strip():
            return ""

        # Step 1: Remove parsing artifacts
        sentence = self._clean_parsing_artifacts(sentence)

        # Step 2: German-specific normalization
        sentence = self._normalize_german_text(sentence)

        # Step 3: General text cleaning
        sentence = self.special_chars_pattern.sub(" ", sentence)
        sentence = self.multiple_spaces_pattern.sub(" ", sentence)

        # Step 4: Final cleanup
        sentence = sentence.strip()

        return sentence

    def clean_corpus_file(
        self, input_file: Path, output_file: Path
    ) -> Tuple[int, int, int]:
        """
        Clean a corpus file and save cleaned version.

        Returns:
            Tuple of (total_lines, cleaned_lines, artifact_lines_removed)
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file '{input_file}' not found")

        logger.info(" Loading corpus from '%s'...", input_file)

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        cleaned_lines = []
        artifact_count = 0
        empty_after_cleaning = 0

        logger.info(" Processing %s lines...", total_lines)

        for i, line in enumerate(lines, 1):
            # Skip empty lines
            if not line.strip():
                continue

            # Clean the sentence
            original_line = line.strip()
            cleaned_line = self._clean_sentence(original_line)

            # Check if parsing artifacts were removed
            if (
                self.parsing_artifact_pattern.search(original_line)
                or self.parsing_artifact_pattern_v2.search(original_line)
                or self.parsing_artifact_pattern_v3.search(original_line)
            ):
                artifact_count += 1

            # Skip lines that become empty after cleaning
            if not cleaned_line:
                empty_after_cleaning += 1
                continue

            cleaned_lines.append(cleaned_line)

            # Progress reporting
            if i % 10000 == 0:
                logger.info(" Processed %s/%s lines (%.1%)", i, total_lines, i / total_lines)

        # Save cleaned corpus
        logger.info(" Saving cleaned corpus to '%s'...", output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            for line in cleaned_lines:
                f.write(line + "\n")

        final_cleaned_count = len(cleaned_lines)

        logger.info(" Corpus cleaning completed!")
        logger.info(" Statistics:")
        logger.info(" Total lines processed: %s", total_lines)
        logger.info(" Lines with parsing artifacts: %s", artifact_count)
        logger.info(" Lines removed (empty after cleaning): %s", empty_after_cleaning)
        logger.info(" Clean sentences saved: %s", final_cleaned_count)
        logger.info(" Retention rate: %.1%", final_cleaned_count / total_lines)

        return total_lines, final_cleaned_count, artifact_count

    def validate_cleaned_corpus(self, file_path: Path) -> bool:
        """
        Validate that a cleaned corpus file has no remaining artifacts.
        """
        if not file_path.exists():
            logger.error(" File '%s' not found", file_path)
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for any remaining parsing artifacts
        has_artifacts = (
            bool(self.parsing_artifact_pattern.search(content))
            or bool(self.parsing_artifact_pattern_v2.search(content))
            or bool(self.parsing_artifact_pattern_v3.search(content))
        )

        if has_artifacts:
            logger.info(" Found remaining parsing artifacts in cleaned file")
            return False

        line_count = content.count("\n")
        logger.info(" Cleaned corpus validation passed")
        logger.info(" Lines: %s", line_count)
        logger.info(" No parsing artifacts detected")

        return True


def main():
    """
    Main function: Clean the Leichte-Sprache corpus for better model training.
    """
    logger.info(" Leichte-Sprache Corpus Cleaning")
    logger.info("=" * 50)

    # Configuration
    INPUT_FILE = Path("mein_leichte_sprache_korpus.txt")
    OUTPUT_FILE = Path("mein_leichte_sprache_korpus_clean.txt")

    try:
        # Initialize cleaner
        cleaner = CorpusCleaner()

        # Clean the corpus
        total, cleaned, artifacts = cleaner.clean_corpus_file(INPUT_FILE, OUTPUT_FILE)

        # Validate the result
        if cleaner.validate_cleaned_corpus(OUTPUT_FILE):
            logger.info("\n Success! Cleaned corpus is ready for model training.")
            logger.info(" Improved data quality: Removed %s parsing artifacts", artifacts)
            logger.info(" Usage: python train_perplexity_model.py --corpus mein_leichte_sprache_korpus_clean.txt")

        return True

    except Exception as e:
        logger.error(" Error during corpus cleaning: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    sys.exit(0 if success else 1)
