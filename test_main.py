import os

import pytest

import main


def test_validate_image_path_missing_file():
    with pytest.raises(FileNotFoundError):
        main.validate_image_path("/definitely/missing/file.png")


def test_validate_search_text_empty_string():
    with pytest.raises(ValueError):
        main.validate_search_text("   ")


def test_validate_search_text_all_punctuation():
    with pytest.raises(ValueError):
        main.validate_search_text("!!!@@@")
