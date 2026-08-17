# trainer.py
import random
import re
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from .config import DeepERConfig
from .schema import flatten_record


class DeepLinkageTrainer:
    def __init__(self, config: DeepERConfig):
        self.config = config
        print(f"Loading base model for training: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)

        # Standard 19th/early-20th century enumerator abbreviations & nicknames
        self.historical_abbreviations = {
            "GEORGE": ["Geo.", "Geo", "Georg"],
            "CHARLES": ["Chas.", "Chas", "Charley"],
            "EDWARD": ["Edw.", "Edw", "Ed"],
            "WILLIAM": ["Wm.", "Wm", "Will"],
            "JAMES": ["Jas.", "Jas", "Jim"],
            "JOHN": ["Jno.", "Jno", "Johnnie"],
            "JOSEPH": ["Jos.", "Jos", "Joe"],
            "BENJAMIN": ["Benj.", "Benj", "Ben"],
            "THEODORE": ["Theo.", "Theo"],
            "MATHEW": ["Matt.", "Matthew", "Matt"],
            "MATTHEW": ["Matt.", "Mathew", "Matt"],
            "MARGARET": ["Maggie", "Margt.", "Marg't"],
            "MILDRED": ["Millie", "Mil."],
            "ROBERT": ["Robt.", "Robt", "Bob"],
            "THOMAS": ["Thos.", "Thos", "Tom"],
        }

        # Phonetic substitution rules for historical surname drift
        self.phonetic_rules = [
            (r"ph", "f"),
            (r"ie$", "y"),
            (r"y$", "ie"),
            (r"ll", "l"),
            (r"pp", "p"),
            (r"tt", "t"),
            (r"nn", "n"),
            (r"mm", "m"),
            (r"ss", "s"),
            (r"ck", "c"),
            (r"'", ""),  # O'Brien -> OBrien
        ]

        # O(1) memory indices for Hard-Negative Mining
        self.household_index = defaultdict(list)
        self.surname_town_index = defaultdict(list)

    def _build_hard_negative_indices(self, df: pd.DataFrame):
        """
        Builds O(1) lookup tables to instantaneously sample sibling/relative hard negatives.
        """
        self.household_index.clear()
        self.surname_town_index.clear()

        for idx, row in df.iterrows():
            last = str(row.get("lastname", "")).strip().lower()
            addr = str(row.get("address", "")).strip().lower()
            town = str(row.get("town", "")).strip().lower()
            first = str(row.get("firstname", "")).strip().lower()

            if last and addr:
                self.household_index[(last, addr)].append((first, idx))
            if last and town:
                self.surname_town_index[(last, town)].append((first, idx))

    def _sample_hard_negative(self, row: pd.Series, df: pd.DataFrame) -> str:
        """
        Mines a challenging negative example:
        Priority 1: Same Household Sibling (same lastname + address, different firstname)
        Priority 2: Same Town Cousin/Namesake (same lastname + town, different firstname)
        Priority 3: Random fallback record from the dataset
        """
        last = str(row.get("lastname", "")).strip().lower()
        addr = str(row.get("address", "")).strip().lower()
        town = str(row.get("town", "")).strip().lower()
        first = str(row.get("firstname", "")).strip().lower()

        # Priority 1: Household Sibling
        hh_candidates = [
            idx
            for (f_name, idx) in self.household_index.get((last, addr), [])
            if f_name != first
        ]
        if hh_candidates:
            return flatten_record(df.iloc[random.choice(hh_candidates)])

        # Priority 2: Surname in same Town
        st_candidates = [
            idx
            for (f_name, idx) in self.surname_town_index.get((last, town), [])
            if f_name != first
        ]
        if st_candidates:
            return flatten_record(df.iloc[random.choice(st_candidates)])

        # Priority 3: Fallback random sample
        return flatten_record(df.iloc[random.randint(0, len(df) - 1)])

    def _apply_structural_dropout(self, row: pd.Series) -> pd.Series:
        corrupted_row = row.copy()
        firstname = str(corrupted_row.get("firstname", ""))
        if len(firstname.split()) > 1 and random.random() < 0.4:
            corrupted_row["firstname"] = firstname.split()[0]

        address = str(corrupted_row.get("address", ""))
        parts = address.split(" ", 1)
        if len(parts) > 1 and parts[0].isdigit() and random.random() < 0.3:
            corrupted_row["address"] = parts[1]
        return corrupted_row

    def _apply_historical_shorthand(self, name_str: str) -> str:
        tokens = name_str.split()
        if not tokens:
            return name_str
        first_upper = tokens[0].upper().replace(".", "")
        if first_upper in self.historical_abbreviations and random.random() < 0.5:
            tokens[0] = random.choice(self.historical_abbreviations[first_upper])
        return " ".join(tokens)

    def _apply_phonetic_drift(self, text: str) -> str:
        mutated = text
        for pattern, replacement in self.phonetic_rules:
            if random.random() < 0.25:
                mutated = re.sub(pattern, replacement, mutated, flags=re.IGNORECASE)
        return mutated

    def _apply_visual_ocr_noise(self, text: str) -> str:
        if not text:
            return text
        chars = list(text)
        swaps = {
            "m": "rn",
            "rn": "m",
            "1": "l",
            "l": "1",
            "c": "e",
            "e": "c",
            "0": "O",
            "S": "5",
            "g": "a",
            "a": "g",
            "J": "x",
            "u": "v",
            "v": "u",
        }
        for i in range(len(chars)):
            if random.random() < self.config.corruption_rate:
                if chars[i] in swaps:
                    chars[i] = swaps[chars[i]]
                elif chars[i].isalpha() and random.random() < 0.15:
                    chars[i] = random.choice(["a", "e", "i", "o", "u"])
        return "".join(chars)

    def _simulate_historical_record(self, row: pd.Series) -> str:
        """
        4-Layer positive generator combining structural dropout, shorthand,
        phonetics, and visual OCR confusions.
        """
        corrupted_row = self._apply_structural_dropout(row)
        firstname = str(corrupted_row.get("firstname", ""))
        corrupted_row["firstname"] = self._apply_historical_shorthand(firstname)

        serialized = flatten_record(corrupted_row)

        if random.random() < 0.4:
            serialized = self._apply_phonetic_drift(serialized)
        return self._apply_visual_ocr_noise(serialized)

    def train_on_data(self, df: pd.DataFrame):
        df = df.fillna("")
        training_examples = []

        print("Indexing households for Hard-Negative Triplet mining...")
        self._build_hard_negative_indices(df)

        print(
            "Generating (Anchor, Positive, Hard-Negative) triplets from master dataset..."
        )
        for _ in range(self.config.epochs_of_synthetic_data):
            for _, row in df.iterrows():
                anchor_text = flatten_record(row)
                positive_text = self._simulate_historical_record(row)
                hard_negative_text = self._sample_hard_negative(row, df)

                training_examples.append(
                    InputExample(
                        texts=[anchor_text, positive_text, hard_negative_text]
                    )
                )

        train_dataloader = DataLoader(
            training_examples,
            shuffle=True,
            batch_size=self.config.training_batch_size,
        )
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        print("Fine-tuning vector model with Hard-Negative Triplet Loss...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.training_epochs,
            warmup_steps=100,
            show_progress_bar=True,
        )

        self.model.save(self.config.custom_model_out_path)
        print(
            f"✅ Triplet-trained model saved to {self.config.custom_model_out_path}!"
        )