import json
import logging
import os
import urllib.request as request
import zipfile

# from nltk.tokenize import sent_tokenize


TMP_ZIP_FILE = "/tmp/python-docs.zip"
TMP_SENT_TOKENIZED = "/tmp/python-docs-sent-tokenized.json"


def setup_logging():
    LOG_FMT = (
        f"%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format=LOG_FMT,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    return


def _get_python_documentation_corpus() -> str:
    if not os.path.exists(TMP_ZIP_FILE):
        request.urlretrieve("https://docs.python.org/3/archives/python-3.12.3-docs-text.zip", TMP_ZIP_FILE)

    lines = []
    with zipfile.ZipFile(TMP_ZIP_FILE, "r") as zipref:
        for fref in zipref.filelist:
            with zipref.open(fref) as f:
                lines.extend(f.readlines())
    return b" ".join(line.strip() for line in lines).decode("utf-8")


def get_python_documentation_sentences() -> list[str]:
    with open("./py_docs_sentences.json", "r") as f:
        return json.load(f)
    # if os.path.exists(TMP_SENT_TOKENIZED):
    #     with open(TMP_SENT_TOKENIZED, "r") as f:
    #         return json.load(f)

    # corpus = _get_python_documentation_corpus()
    # tokenized = sent_tokenize(corpus)

    # with open(TMP_SENT_TOKENIZED, "w") as f:
    #     json.dump(tokenized, f)

    # return tokenized
