#!/usr/bin/env python3
"""Download word-embedding models from Hugging Face Hub.

By default every supported language is downloaded.  Pass ``--lang`` to
restrict which languages are fetched.

Usage examples
--------------
Download all languages::

    python download_models.py

Download only English and Spanish::

    python download_models.py --lang en es

Authenticate with a private / gated repository::

    HF_TOKEN=hf_... python download_models.py
    # or
    python download_models.py --hf-token hf_...

Override the source repository for a specific language::

    HF_REPO_EN=my-org/my-en-vectors HF_FILE_EN=vectors.vec python download_models.py --lang en
"""

import argparse
import os
import sys

# Default Hugging Face Hub model configuration per language:
#   { lang: (repo_id, filename_in_repo) }
# These defaults can be overridden with environment variables:
#   HF_REPO_<LANG> and HF_FILE_<LANG>  (e.g. HF_REPO_EN, HF_FILE_EN)
LANG_HF_MODELS: dict[str, tuple[str, str]] = {
    'en': (
        os.environ.get('HF_REPO_EN', 'facebook/fasttext-en-vectors'),
        os.environ.get('HF_FILE_EN', 'model.bin'),
    ),
    'he': (
        os.environ.get('HF_REPO_HE', 'facebook/fasttext-he-vectors'),
        os.environ.get('HF_FILE_HE', 'model.bin'),
    ),
    'es': (
        os.environ.get('HF_REPO_ES', 'facebook/fasttext-es-vectors'),
        os.environ.get('HF_FILE_ES', 'model.bin'),
    ),
}

EMBEDDINGS_DIR = os.environ.get('EMBEDDINGS_DIR', 'embeddings')


def download_model(
    lang: str,
    repo_id: str | None = None,
    filename: str | None = None,
    token: str | None = None,
    embeddings_dir: str = EMBEDDINGS_DIR,
) -> str:
    """Download the model for *lang* from Hugging Face Hub.

    Parameters
    ----------
    lang:
        Language code (``'en'``, ``'he'``, ``'es'``, …).
    repo_id:
        Hugging Face repository ID (e.g. ``'facebook/fasttext-en-vectors'``).
        Falls back to the value configured in :data:`LANG_HF_MODELS`.
    filename:
        Filename inside the repository (e.g. ``'model.bin'``).
        Falls back to the value configured in :data:`LANG_HF_MODELS`.
    token:
        Hugging Face API token for private / gated repositories.
        Defaults to the ``HF_TOKEN`` environment variable.
    embeddings_dir:
        Local directory where the file will be stored.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        ) from exc

    default_repo, default_file = LANG_HF_MODELS.get(lang, ('', ''))
    repo_id = repo_id or default_repo
    filename = filename or default_file

    if not repo_id or not filename:
        raise ValueError(
            f"No Hugging Face model configured for language '{lang}'. "
            "Pass --repo-id and --filename explicitly."
        )

    os.makedirs(embeddings_dir, exist_ok=True)

    print(f"[{lang}] Downloading from {repo_id}/{filename} …")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=embeddings_dir,
    )
    print(f"[{lang}] Saved to {path}")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download word-embedding models from Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--lang',
        nargs='+',
        default=list(LANG_HF_MODELS.keys()),
        metavar='LANG',
        help=(
            "Language codes to download (default: all supported languages). "
            f"Known languages: {', '.join(LANG_HF_MODELS)}."
        ),
    )
    parser.add_argument(
        '--hf-token',
        default=os.environ.get('HF_TOKEN'),
        metavar='TOKEN',
        help=(
            "Hugging Face API token for private / gated repositories. "
            "Alternatively, set the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        '--repo-id',
        default=None,
        metavar='REPO_ID',
        help=(
            "Override the Hugging Face repository ID. "
            "Only useful when downloading a single --lang."
        ),
    )
    parser.add_argument(
        '--filename',
        default=None,
        metavar='FILENAME',
        help=(
            "Override the filename inside the repository. "
            "Only useful when downloading a single --lang."
        ),
    )
    parser.add_argument(
        '--embeddings-dir',
        default=EMBEDDINGS_DIR,
        metavar='DIR',
        help=f"Directory where models are saved (default: {EMBEDDINGS_DIR!r}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if (args.repo_id or args.filename) and len(args.lang) > 1:
        print(
            "Error: --repo-id / --filename can only be used when downloading "
            "a single language. Specify exactly one --lang value.",
            file=sys.stderr,
        )
        sys.exit(1)

    for lang in args.lang:
        download_model(
            lang,
            repo_id=args.repo_id,
            filename=args.filename,
            token=args.hf_token,
            embeddings_dir=args.embeddings_dir,
        )


if __name__ == '__main__':
    main()
