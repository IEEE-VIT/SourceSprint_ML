# SourceSprint_Juice
Repository with issues for SourceSprint<br><br>

This code combines image processing and natural language processing techniques to extract textual information from an image and find similar words or phrases. It loads and displays an image, extracts text content using the CLIP model, and preprocesses the text for analysis. After training a Word2Vec model, it accepts user input and searches for exact or partial matches within the processed text, reporting any findings with a confidence level of 80 or higher.

The expected final result is given in image.png

## Pipeline modules

- `image_processing.py`: loads RGB images and displays them.
- `nlp_preprocessing.py`: trims search text while preserving vocabulary keys.
- `model_inference.py`: loads CLIP and Word2Vec and runs their inference calls.
- `main.py`: handles command-line arguments and coordinates the pipeline.

Run with Pillow, clip-interrogator, and gensim installed, and a trained Word2Vec
model available locally:

```bash
python main.py image.png --model-path word2vec_model.model
```

The Word2Vec model is not included in this repository. CLIP loads its configured
weights on first use. Importing the modules does not load models, open an image
viewer, or prompt for input.

Run the tests without downloading model weights:

```bash
python -m unittest discover -s tests -v
```
