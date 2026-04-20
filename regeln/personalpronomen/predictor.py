"""Inference pipeline for personal pronoun classification.

PyTorch port of the MLX-based predictor from the external personalpronomen package.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .pronoun_model import ModelConfig, PronounClassifier, create_model


class PronounPredictor:
    """Inference pipeline for pronoun classification."""

    def __init__(
        self,
        model_path: Path,
        label_encoders_path: Path,
        tokenizer_name: str = "bert-base-german-cased",
        max_length: int = 128,
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to model checkpoint (.pt file)
            label_encoders_path: Path to label encoders JSON
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
        """
        self.max_length = max_length

        # Load label encoders
        with open(label_encoders_path, "r") as f:
            self.label_encoders = json.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Create and load model
        self.config = self._create_config()
        self.model = self._load_model(model_path)

    def _create_config(self) -> ModelConfig:
        """Create model configuration from label encoders."""
        config = ModelConfig()
        config.num_person_classes = self.label_encoders["person"]["num_classes"]
        config.num_gender_classes = self.label_encoders["gender"]["num_classes"]
        config.num_number_classes = self.label_encoders["number"]["num_classes"]
        config.num_case_classes = self.label_encoders["case"]["num_classes"]
        config.num_polite_classes = self.label_encoders["polite"]["num_classes"]
        config.num_reflex_classes = self.label_encoders["reflex"]["num_classes"]
        return config

    def _load_model(self, model_path: Path) -> PronounClassifier:
        """Load trained model from checkpoint."""
        model = create_model(self.config)
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def find_pronouns(self, text: str) -> List[Tuple[str, int]]:
        """Find personal pronouns in text.

        Args:
            text: Input text

        Returns:
            List of (pronoun, position) tuples
        """
        # Common German personal pronouns
        pronouns = {
            "ich",
            "mich",
            "mir",
            "meiner",
            "du",
            "dich",
            "dir",
            "deiner",
            "er",
            "ihn",
            "ihm",
            "seiner",
            "sie",
            "Sie",
            "ihr",
            "Ihr",
            "ihre",
            "Ihre",
            "ihrer",
            "Ihrer",
            "es",
            "sein",
            "wir",
            "uns",
            "unser",
            "ihr",
            "euch",
            "euer",
            "sie",
            "Sie",
            "ihnen",
            "Ihnen",
            "sich",
        }

        words = text.split()
        found_pronouns = []

        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = word.strip(".,!?;:")
            if clean_word in pronouns or clean_word.lower() in pronouns:
                found_pronouns.append((word, i))

        return found_pronouns

    def build_context_window(
        self, sentences: List[str], current_idx: int
    ) -> Tuple[str, Dict[str, int]]:
        """Build a 3-sentence context window around the current sentence.

        Args:
            sentences: List of all sentences
            current_idx: Index of the current sentence

        Returns:
            Tuple of (context_text, sentence_boundaries)
                context_text: Combined text of previous + current + next sentences
                sentence_boundaries: Dict with start positions of each sentence
        """
        prev_sent = sentences[current_idx - 1] if current_idx > 0 else ""
        curr_sent = sentences[current_idx]
        next_sent = (
            sentences[current_idx + 1] if current_idx < len(sentences) - 1 else ""
        )

        # Build context with sentence boundary markers
        context_parts = []
        sentence_boundaries = {}

        if prev_sent:
            sentence_boundaries["prev"] = len(context_parts)
            context_parts.extend(prev_sent.split())

        sentence_boundaries["current"] = len(context_parts)
        context_parts.extend(curr_sent.split())

        if next_sent:
            sentence_boundaries["next"] = len(context_parts)
            context_parts.extend(next_sent.split())

        context_text = " ".join(context_parts)
        return context_text, sentence_boundaries

    def extract_candidates(
        self,
        context_text: str,
        pronoun_position: int,
        sentence_boundaries: Dict[str, int],
    ) -> List[Tuple[str, int]]:
        """Extract candidate antecedents from the context.

        Args:
            context_text: The extended context (3 sentences)
            pronoun_position: Position of the pronoun in the context
            sentence_boundaries: Sentence boundary positions

        Returns:
            List of (candidate, position) tuples
        """
        # Common German nouns that could be antecedents
        # These should match gender/number with pronouns
        words = context_text.split()
        candidates = []

        # Look for nouns (simplified - capitalized words in German)
        for i, word in enumerate(words):
            clean_word = word.strip(".,!?;:")

            # Skip the pronoun itself
            if i == pronoun_position:
                continue

            # Simple heuristic: capitalized words that aren't sentence-initial
            # and aren't pronouns themselves
            is_sentence_start = (
                i == 0
                or i == sentence_boundaries.get("current", -1)
                or i == sentence_boundaries.get("next", -1)
            )

            # Check if it's a capitalized word (potential noun in German)
            # and not a pronoun
            if clean_word and clean_word[0].isupper():
                # Check if it's not a pronoun
                if not self.find_pronouns(clean_word):
                    # Add if it's not at sentence start (unless it could still be a noun)
                    if not is_sentence_start or len(clean_word) > 2:
                        candidates.append((clean_word, i))

        # Prioritize candidates from current and previous sentences
        # (most likely antecedents are recent)
        current_start = sentence_boundaries.get("current", 0)
        candidates.sort(
            key=lambda x: (
                abs(x[1] - pronoun_position),  # Distance from pronoun
                x[1] < current_start,  # Prefer previous sentence
            )
        )

        return candidates[:10]  # Limit to top 10 candidates

    def get_article_for_noun(self, gender: str, number: str, case: str = "Nom") -> str:
        """Get the appropriate German article for a noun.

        Args:
            gender: Gender of the noun (Masc, Fem, Neut)
            number: Number (Sing, Plur)
            case: Grammatical case (Nom, Acc, Dat, Gen)

        Returns:
            Appropriate definite article
        """
        # For plural, article is always "Die" in nominative/accusative
        if number == "Plur":
            if case in ("Nom", "Acc"):
                return "Die"
            elif case == "Dat":
                return "Den"
            else:  # Gen
                return "Der"

        # Singular articles by gender and case
        articles = {
            "Masc": {"Nom": "Der", "Acc": "Den", "Dat": "Dem", "Gen": "Des"},
            "Fem": {"Nom": "Die", "Acc": "Die", "Dat": "Der", "Gen": "Der"},
            "Neut": {"Nom": "Das", "Acc": "Das", "Dat": "Dem", "Gen": "Des"},
        }

        return articles.get(gender, {}).get(case, "Die")

    def should_replace_pronoun(
        self, prediction: Dict, sentence_position: int, candidates: List[Dict]
    ) -> Optional[Dict]:
        """Determine if a pronoun should be replaced with its antecedent.

        Args:
            prediction: The pronoun prediction with features
            sentence_position: Position of the pronoun in the sentence (0-indexed)
            candidates: List of candidate antecedents

        Returns:
            Dictionary with replacement suggestion, or None if no replacement needed
        """
        features = prediction.get("features", {})

        # Only suggest replacement for 3rd person pronouns
        person = features.get("person", {}).get("value")
        if person != "3":
            return None

        # Don't replace polite "Sie" (2nd person formal)
        polite = features.get("polite", {}).get("value")
        if polite == "Form":
            return None

        # Only suggest replacement for sentence-initial pronouns (position 0 or 1)
        if sentence_position > 1:
            return None

        # Need at least one candidate antecedent
        if not candidates:
            return None

        # Get pronoun's gender and number
        pronoun_gender = features.get("gender", {}).get("value")
        pronoun_number = features.get("number", {}).get("value")

        # Find best matching candidate - prefer those that agree in number
        # For German, plural "sie" (they) should match plural nouns
        best_candidate = None
        best_score = -1

        for candidate in candidates:
            score = 0
            word = candidate.get("word", "").strip(".,!?;:")

            # In German, plural nouns often end in -er, -en, -e, or are unchanged
            # "Helfer" is typically plural when referring to multiple helpers
            # This is a heuristic - proper noun morphology would be better
            is_likely_plural = (
                word.endswith("er")
                or word.endswith("en")
                or word.endswith("e")
                or word.endswith("s")
            )

            # Score based on agreement
            if pronoun_number == "Plur" and is_likely_plural:
                score += 2
            elif pronoun_number == "Sing" and not is_likely_plural:
                score += 2

            # Prefer earlier candidates (from previous sentence context)
            # Lower position = earlier in text = more likely antecedent for subject
            if candidate.get("position", 999) < 10:
                score += 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # If no scored candidate, fall back to first
        if best_candidate is None and candidates:
            best_candidate = candidates[0]

        if not best_candidate:
            return None

        # Generate replacement suggestion
        noun = best_candidate.get("word", "").strip(".,!?;:")

        # Determine article based on pronoun's features
        # Use nominative case for sentence-initial position
        article = self.get_article_for_noun(
            pronoun_gender if pronoun_gender != "NONE" else "Masc",
            pronoun_number if pronoun_number != "NONE" else "Sing",
            "Nom",
        )

        replacement = f"{article} {noun}"

        return {
            "should_replace": True,
            "replacement_suggestion": replacement,
            "replacement_reason": f"3rd person pronoun at sentence start refers to '{noun}'",
            "antecedent": noun,
        }

    def predict_sentence(
        self,
        sentence: str,
        pronoun_positions: Optional[List[int]] = None,
        prev_sentence: Optional[str] = None,
        next_sentence: Optional[str] = None,
    ) -> List[Dict]:
        """Predict pronoun features for a sentence with optional context.

        Args:
            sentence: Input sentence
            pronoun_positions: Optional list of word positions containing pronouns
            prev_sentence: Optional previous sentence for context
            next_sentence: Optional next sentence for context

        Returns:
            List of predictions for each pronoun
        """
        # Build context if adjacent sentences provided
        if prev_sentence or next_sentence:
            sentences = [s for s in [prev_sentence, sentence, next_sentence] if s]
            curr_idx = 1 if prev_sentence else 0
            context_text, sentence_boundaries = self.build_context_window(
                sentences, curr_idx
            )
        else:
            context_text = sentence
            sentence_boundaries = {"current": 0}

        # Tokenize context
        encoding = self.tokenizer(
            context_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        # Find pronouns in the current sentence if not provided
        if pronoun_positions is None:
            found_pronouns = self.find_pronouns(sentence)
            if not found_pronouns:
                return []
            pronoun_positions = [pos for _, pos in found_pronouns]

        if not pronoun_positions:
            return []

        words = context_text.split()
        current_start = sentence_boundaries.get("current", 0)

        # Adjust pronoun positions to context coordinates
        context_positions = [current_start + pos for pos in pronoun_positions]

        # Filter out invalid positions
        valid_positions = [
            (pos, orig_pos)
            for pos, orig_pos in zip(context_positions, pronoun_positions)
            if pos < len(words)
        ]
        if not valid_positions:
            return []

        # Create batch of inputs - one copy per pronoun
        batch_size = len(valid_positions)
        input_ids = torch.tensor([encoding["input_ids"]] * batch_size, dtype=torch.long)
        attention_mask = torch.tensor([encoding["attention_mask"]] * batch_size, dtype=torch.long)

        # Create pronoun masks and candidate masks for all pronouns
        pronoun_masks = []
        candidate_masks = []
        offset_mapping = encoding["offset_mapping"]

        for context_pos, _ in valid_positions:
            # Create pronoun mask for this specific pronoun
            pronoun_mask = [0.0] * self.max_length

            # Map word position to token positions
            char_start = sum(len(w) + 1 for w in words[:context_pos])
            char_end = char_start + len(words[context_pos])

            for token_idx, (start, end) in enumerate(offset_mapping):
                if start is not None and end is not None:
                    if start >= char_start and end <= char_end + 1:
                        pronoun_mask[token_idx] = 1.0

            # If no tokens matched, use approximate position
            if sum(pronoun_mask) == 0:
                approx_pos = min(context_pos + 1, len(encoding["input_ids"]) - 2)
                pronoun_mask[approx_pos] = 1.0

            pronoun_masks.append(pronoun_mask)

            # Extract candidates and create candidate mask
            candidates = self.extract_candidates(
                context_text, context_pos, sentence_boundaries
            )
            candidate_mask = [0.0] * self.max_length

            if candidates:
                for cand_word, cand_pos in candidates[:5]:  # Top 5 candidates
                    # Map candidate position to token positions
                    cand_char_start = sum(len(w) + 1 for w in words[:cand_pos])
                    cand_char_end = cand_char_start + len(words[cand_pos])

                    for token_idx, (start, end) in enumerate(offset_mapping):
                        if start is not None and end is not None:
                            if start >= cand_char_start and end <= cand_char_end + 1:
                                candidate_mask[token_idx] = 1.0

            candidate_masks.append(candidate_mask)

        pronoun_mask_batch = torch.tensor(pronoun_masks, dtype=torch.float32)
        candidate_mask_batch = torch.tensor(candidate_masks, dtype=torch.float32) if candidate_masks else None

        # Single forward pass for all pronouns with candidate masks
        with torch.no_grad():
            logits = self.model(
                input_ids,
                attention_mask,
                pronoun_mask_batch,
                candidate_mask_batch,
            )

        # Decode predictions for all pronouns
        predictions = []
        sentence_words = sentence.split()

        for idx, (context_pos, orig_pos) in enumerate(valid_positions):
            prediction = {
                "token": sentence_words[orig_pos]
                if orig_pos < len(sentence_words)
                else words[context_pos],
                "position": orig_pos,
                "features": {},
                "context_used": prev_sentence is not None or next_sentence is not None,
            }

            # Add candidate information if available
            candidates = self.extract_candidates(
                context_text, context_pos, sentence_boundaries
            )
            if candidates:
                prediction["candidates"] = [
                    {"word": word, "position": pos} for word, pos in candidates[:3]
                ]

            for task, task_logits in logits.items():
                pred_idx = torch.argmax(task_logits[idx]).item()
                id_to_label = self.label_encoders[task]["id_to_label"]
                pred_label = id_to_label[str(pred_idx)]

                # Convert label to string for all heads.
                # This is critical for the reflex head where JSON deserializes
                # booleans (false/true) to Python bool. str(True) -> "True".
                # For other heads it's a no-op since values are already strings.
                pred_label = str(pred_label)

                # Get confidence
                probs = torch.softmax(task_logits[idx], dim=-1)
                confidence = probs[pred_idx].item()

                prediction["features"][task] = {
                    "value": pred_label,
                    "confidence": confidence,
                }

            # Check if this pronoun should be replaced with its antecedent
            replacement_info = self.should_replace_pronoun(
                prediction,
                orig_pos,
                prediction.get("candidates", []),
            )
            if replacement_info:
                prediction.update(replacement_info)

            predictions.append(prediction)

        return predictions

    def predict_text(self, text: str, use_context: bool = True) -> Dict:
        """Predict pronouns in a longer text with multi-sentence context.

        Args:
            text: Input text (can be multiple sentences)
            use_context: Whether to use multi-sentence context (default: True)

        Returns:
            Dictionary with predictions for all sentences
        """
        # Split into sentences - handles missing spaces after punctuation
        # Use regex to split on sentence-ending punctuation followed by capital letter
        sentences = re.split(r"([.!?])\s*(?=[A-Z\u00c4\u00d6\u00dc])", text)

        # Recombine punctuation with preceding sentence and clean up
        cleaned_sentences = []
        i = 0
        while i < len(sentences):
            sent = sentences[i].strip()
            # If next element is punctuation, append it
            if i + 1 < len(sentences) and sentences[i + 1] in ".!?":
                sent = sent + sentences[i + 1]
                i += 2
            else:
                i += 1
            if sent and sent not in ".!?":
                cleaned_sentences.append(sent)
        sentences = cleaned_sentences

        results = {"text": text, "sentences": [], "multi_sentence_context": use_context}

        for i, sentence in enumerate(sentences):
            pronouns = self.find_pronouns(sentence)

            if pronouns:
                # Get adjacent sentences for context if enabled
                if use_context:
                    prev_sentence = sentences[i - 1] if i > 0 else None
                    next_sentence = sentences[i + 1] if i < len(sentences) - 1 else None

                    predictions = self.predict_sentence(
                        sentence,
                        prev_sentence=prev_sentence,
                        next_sentence=next_sentence,
                    )
                else:
                    predictions = self.predict_sentence(sentence)

                results["sentences"].append(
                    {
                        "sentence": sentence,
                        "sentence_index": i,
                        "pronouns": predictions,
                        "has_context": use_context
                        and ((i > 0) or (i < len(sentences) - 1)),
                    }
                )

        return results

    def format_prediction(self, prediction: Dict) -> str:
        """Format a prediction for display.

        Args:
            prediction: Single pronoun prediction

        Returns:
            Formatted string
        """
        lines = [f"Token: '{prediction['token']}' (position {prediction['position']})"]

        # Show if multi-sentence context was used
        if prediction.get("context_used"):
            lines.append("  [Using multi-sentence context]")

        # Show replacement suggestion if available
        if prediction.get("should_replace"):
            lines.append(
                f"  ** REPLACE WITH: '{prediction['replacement_suggestion']}' **"
            )
            lines.append(f"     Reason: {prediction['replacement_reason']}")

        # Show candidate antecedents if available
        if "candidates" in prediction and prediction["candidates"]:
            candidates_str = ", ".join(
                [f"'{c['word']}'" for c in prediction["candidates"][:3]]
            )
            lines.append(f"  Candidates: {candidates_str}")

        for feature, info in prediction["features"].items():
            value = info["value"]
            confidence = info["confidence"]

            if value != "NONE":
                # Convert value to string for formatting (handles bool values)
                value_str = str(value)
                lines.append(
                    f"  {feature:8s}: {value_str:10s} (confidence: {confidence:.3f})"
                )

        return "\n".join(lines)
