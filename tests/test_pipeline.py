import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import main
from model_inference import find_similar_words, image_to_prompt
from nlp_preprocessing import preprocess_text
from image_processing import load_image


class PipelineTests(unittest.TestCase):
    def test_missing_image_is_rejected_before_pillow(self):
        with self.assertRaises(FileNotFoundError):
            load_image("does-not-exist.png")

    def test_preprocessing_preserves_vocabulary_keys(self):
        self.assertEqual(preprocess_text("  New York \n"), "New York")

    def test_clip_inference_uses_rgb(self):
        image = Mock()
        interrogator = Mock()
        interrogator.interrogate_fast.return_value = "A cat"
        self.assertEqual(image_to_prompt(image, interrogator), "A cat")
        image.convert.assert_called_once_with("RGB")
        interrogator.interrogate_fast.assert_called_once_with(image.convert.return_value)

    def test_word2vec_lookup(self):
        model = Mock()
        model.wv.most_similar.return_value = [("kitten", 0.9)]
        self.assertEqual(find_similar_words(model, "cat"), [("kitten", 0.9)])
        model.wv.most_similar.assert_called_once_with("cat", topn=10)

    def test_pipeline_success_and_unknown_word(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown), contextlib.ExitStack() as stack:
                image = Mock()
                stack.enter_context(patch.object(main, "load_image", return_value=image))
                display = stack.enter_context(patch.object(main, "display_image"))
                stack.enter_context(patch.object(main, "load_clip_model"))
                stack.enter_context(patch.object(main, "image_to_prompt", return_value="A cat"))
                load_model = stack.enter_context(patch.object(main, "load_word2vec_model"))
                lookup = stack.enter_context(patch.object(main, "find_similar_words"))
                if unknown:
                    lookup.side_effect = KeyError("cat")
                else:
                    lookup.return_value = [("kitten", 0.9)]
                stack.enter_context(patch("builtins.input", return_value=" cat "))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main.main(["photo.png", "--model-path", "trained.model"])
                display.assert_called_once_with(image)
                load_model.assert_called_once_with("trained.model")
                lookup.assert_called_once_with(load_model.return_value, "cat")
                image.close.assert_called_once()
                self.assertIn("A cat", output.getvalue())
                self.assertIn(
                    "not in the Word2Vec vocabulary" if unknown else "kitten - similarity: 0.90",
                    output.getvalue(),
                )

    def test_empty_search_text_stops_before_model_loading(self):
        image = Mock()
        with patch.object(main, "load_image", return_value=image), \
                patch.object(main, "display_image"), \
                patch.object(main, "load_clip_model") as load_clip, \
                patch("builtins.input", return_value="   "), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = main.main(["photo.png"])
        self.assertEqual(result, 1)
        load_clip.assert_not_called()
        self.assertIn("Search text cannot be empty", output.getvalue())
        image.close.assert_called_once()

    def test_missing_model_is_reported(self):
        image = Mock()
        with patch.object(main, "load_image", return_value=image), \
                patch.object(main, "display_image"), \
                patch.object(main, "load_clip_model"), \
                patch.object(main, "image_to_prompt", return_value="A cat"), \
                patch.object(main, "load_word2vec_model", side_effect=FileNotFoundError("missing")), \
                patch("builtins.input", return_value="cat"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = main.main(["photo.png", "--model-path", "missing.model"])
        self.assertEqual(result, 1)
        self.assertIn("Unable to load Word2Vec model", output.getvalue())
        image.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
