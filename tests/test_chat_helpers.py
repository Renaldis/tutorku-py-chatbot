import unittest

from routers.chat import build_material_filter, format_chat_history


class ChatHelperTest(unittest.TestCase):
    def test_build_material_filter_limits_retrieval_to_requested_material(self):
        material_filter = build_material_filter("material-123")

        self.assertEqual(
            material_filter,
            {
                "must": [
                    {
                        "key": "metadata.material_id",
                        "match": {"value": "material-123"},
                    }
                ]
            },
        )

    def test_format_chat_history_preserves_role_and_content(self):
        history = [
            {"role": "user", "content": "Apa itu normalisasi?"},
            {
                "role": "assistant",
                "content": "Normalisasi adalah proses perancangan basis data.",
            },
        ]

        self.assertEqual(
            format_chat_history(history),
            "user: Apa itu normalisasi?\nassistant: Normalisasi adalah proses perancangan basis data.",
        )


if __name__ == "__main__":
    unittest.main()
