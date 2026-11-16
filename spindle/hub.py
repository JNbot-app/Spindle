"""Download and read public dataset files (Hugging Face Hub and plain URLs), with a local cache."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get('SPINDLE_CACHE', Path.home() / '.cache' / 'spindle'))


def hf_file(repo, name):
    """Local path of one file of a Hugging Face dataset repo."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type='dataset')


def url_file(url):
    """Local path of a downloaded URL."""
    path = CACHE / 'urls' / (hashlib.sha1(url.encode()).hexdigest()[:16] + '-' + url.rsplit('/', 1)[-1])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        path.with_suffix(path.suffix + '.part').write_bytes(data)
        path.with_suffix(path.suffix + '.part').rename(path)
    return path


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path


def read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pylist()


def hf_jsonl(repo, name):
    return read_jsonl(hf_file(repo, name))


def hf_parquet(repo, name, columns=None):
    return read_parquet(hf_file(repo, name), columns)


def cached(name, build):
    """Rows from CACHE/name.jsonl, built once with `build()`."""
    path = CACHE / f'{name}.jsonl'
    if not path.exists():
        print(f'  preparing {name}', flush=True)
        write_jsonl(path, build())
    return read_jsonl(path)
