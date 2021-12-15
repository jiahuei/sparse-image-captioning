"""
@Author  :   JiaHuei
@Time    :   2021/12/15 13:09:34
"""

import unittest
import os
from sparse_caption.utils.config import Config
from sparse_caption.tokenizer import TOKENIZER_REGISTRY, get_tokenizer, Tokenizer
from sparse_caption.data import get_dataset
from .paths import TEST_DIRPATH, TEST_DATA_DIRPATH


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.base_log_dir = os.path.join(TEST_DIRPATH, "experiments")
        self.config = Config(
            start_from="",
            dataset="mscoco_testing",
            dataset_dir=TEST_DATA_DIRPATH,
            log_dir=self.base_log_dir,
            vocab_size=512,
            retokenize_captions=False,
            radix_base=768,
            logging_level="INFO",
        )

    def test_tokenizer(self):
        self.assertTrue(len(TOKENIZER_REGISTRY) > 1, "There should be more than 1 tokenizer in `TOKENIZER_REGISTRY`.")

        for tok_name in TOKENIZER_REGISTRY:
            config = self.config
            config.log_dir = os.path.join(self.base_log_dir, tok_name)
            config.tokenizer = tok_name
            tokenizer_dir = os.path.join(config.log_dir, "tokenizer")

            # Prepare training data
            self.data = get_dataset(config.dataset)(config)
            self.data.prepare_data()

            # Train the tokenizer
            with self.subTest(f"Testing tokenizer: {tok_name}"):
                tokenizer = get_tokenizer(config.tokenizer)(config)
                self.assertIsInstance(tokenizer, Tokenizer, "Tokenizer object is not a subclass of `Tokenizer`.")
                tokenizer_files = os.listdir(tokenizer_dir)
                self.assertTrue(len(tokenizer_files) > 0, "Tokenizer directory should not be empty.")
                self.assertTrue(
                    any(_.endswith(".model") for _ in tokenizer_files),
                    "Tokenizer directory should contain a `*.model` file.",
                )


if __name__ == "__main__":
    unittest.main()
