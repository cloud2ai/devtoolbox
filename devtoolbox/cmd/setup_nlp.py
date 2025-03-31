# devtoolbox/setup_nlp.py
import subprocess
import sys

# Define supported languages
LANGS = ['en', 'zh']

def download_spacy_models():
    """Download required spaCy models."""
    models = [f"{lang}_core_{'web' if lang in ['en', 'zh'] else 'news'}_sm"
             for lang in LANGS]

    for model in models:
        try:
            print(f"Downloading {model}...")
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model])
            print(f"Successfully downloaded {model}")
        except Exception as e:
            print(f"Error downloading {model}: {e}")


if __name__ == "__main__":
    download_spacy_models()
