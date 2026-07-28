import pandas as pd
import random
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from .config import DeepERConfig


class DeepLinkageTrainer:
    def __init__(self, config: DeepERConfig):
        self.config = config
        print(f"Loading base model for training: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)

    def _flatten_schema(self, df: pd.DataFrame) -> list[str]:
        """Uses the exact same flattening logic as the Engine."""
        df = df.fillna("")
        sentences = []
        for _, row in df.iterrows():
            sentence = f"Name: {row.get('firstname', '')} {row.get('lastname', '')}, " \
                       f"Address: {row.get('address', '')}, " \
                       f"Town: {row.get('town', '')}"
            sentences.append(sentence.strip())
        return sentences

    def _simulate_ocr_errors(self, text: str) -> str:
        """Artificially corrupts text based on config thresholds."""
        if not text: return text
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < self.config.corruption_rate:
                swaps = {'m': 'rn', 'rn': 'm', '1': 'l', 'l': '1', 'c': 'e', 'e': 'c', '0': 'O', 'S': '5'}
                char = chars[i]
                if char in swaps:
                    chars[i] = swaps[char]
                elif random.random() < 0.3:
                    chars[i] = ''
        return "".join(chars)

    def train_on_data(self, df: pd.DataFrame):
        sentences = self._flatten_schema(df)
        training_examples = []

        print("Generating synthetic training pairs...")
        for _ in range(self.config.epochs_of_synthetic_data):
            for original_text in sentences:
                corrupted_text = self._simulate_ocr_errors(original_text)
                training_examples.append(InputExample(texts=[original_text, corrupted_text]))

        train_dataloader = DataLoader(
            training_examples,
            shuffle=True,
            batch_size=self.config.training_batch_size
        )
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        print("Commencing contrastive learning loop...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.training_epochs,
            warmup_steps=10,
            show_progress_bar=True
        )

        self.model.save(self.config.custom_model_out_path)
        print(f"✅ Custom model saved to {self.config.custom_model_out_path}!")