import unittest

from lowparllm.ai.llm.generation_config import GenerationConfig
from lowparllm.core.config import AppConfig


class GenerationConfigTests(unittest.TestCase):
    def test_default_and_unlimited_output_options(self) -> None:
        generation = GenerationConfig.from_app_config(AppConfig())
        self.assertEqual(generation.num_ctx, 4096)
        self.assertEqual(generation.num_predict, -1)
        self.assertEqual(generation.as_options()["num_predict"], -1)

    def test_configurable_limits_are_passed_to_ollama(self) -> None:
        generation = GenerationConfig.from_app_config(
            AppConfig(context_size=8192, num_predict=1200, temperature=0.25, top_p=0.9)
        )
        self.assertEqual(
            generation.as_options(),
            {
                "temperature": 0.25,
                "top_p": 0.9,
                "top_k": 20,
                "num_ctx": 8192,
                "num_predict": 1200,
            },
        )

    def test_invalid_generation_settings_are_rejected(self) -> None:
        invalid_configs = (
            {"temperature": float("nan")},
            {"temperature": 2.1},
            {"top_p": -0.1},
            {"context_size": 128},
            {"num_predict": 0},
            {"num_predict": -2},
            {"keep_alive": " "},
        )
        for values in invalid_configs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationConfig.from_app_config(AppConfig(**values))


if __name__ == "__main__":
    unittest.main()
