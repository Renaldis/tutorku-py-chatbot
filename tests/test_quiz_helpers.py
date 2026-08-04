import unittest

from routers.quiz import clean_json_output


class QuizHelperTest(unittest.TestCase):
    def test_clean_json_output_parses_markdown_json_block(self):
        raw_output = """```json
[
  {
    "question": "Apa fungsi primary key?",
    "options": {"A": "Unik", "B": "Duplikat"},
    "correct_answer": "A",
  }
]
```"""

        questions = clean_json_output(raw_output)

        self.assertIsInstance(questions, list)
        self.assertEqual(questions[0]["question"], "Apa fungsi primary key?")
        self.assertEqual(questions[0]["correct_answer"], "A")

    def test_clean_json_output_returns_parse_error_for_invalid_json(self):
        result = clean_json_output("bukan json")

        self.assertIn("raw_output", result)
        self.assertIn("parse_error", result)


if __name__ == "__main__":
    unittest.main()
