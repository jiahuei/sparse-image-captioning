# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors and jiahuei. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Utilities for file download and caching.
"""
import hashlib
import os
import shutil
import tarfile
import zipfile
import six
import requests
import json
import logging
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from .misc import humanise_number

logger = logging.getLogger(__name__)


def file_size(path, suffix="B"):
    num = os.path.getsize(path)
    return humanise_number(num)


def list_dir(path):
    return sorted(os.path.join(path, _) for _ in os.listdir(path))


def list_files(path):
    files = [os.path.join(root, f) for root, dirs, files in os.walk(path) for f in files]
    return files


def read_file(path):
    with open(path, "r") as f:
        data = [_.rstrip() for _ in f.readlines()]
    return data


def read_json(path, utf8=True):
    encoding = "utf8" if utf8 else None
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)
    return data


def dump_json(path, data, utf8=True, lf_newline=True, **json_kwargs):
    encoding = "utf8" if utf8 else None
    newline = "\n" if lf_newline else None
    if utf8 and json_kwargs.get("ensure_ascii", True):
        logger.warning(
            f"Opening file handler for `{path}` in UTF-8 mode, " "but `ensure_ascii` is set to True (default) for JSON"
        )
    with open(path, "w", encoding=encoding, newline=newline) as f:
        json.dump(data, f, **json_kwargs)
    return path


def dumps_file(path, string, utf8=True, lf_newline=True):
    encoding = "utf8" if utf8 else None
    newline = "\n" if lf_newline else None
    with open(path, "w", encoding=encoding, newline=newline) as f:
        f.write(string)
    return path


def tqdm_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    with tqdm(...) as t:
        reporthook = tqdm_hook(t)
        urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def load_pil_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def get_file(
    fname,
    origin,
    dest_dir,
    md5_hash=None,
    file_hash=None,
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
):
    """Downloads a file from a URL if it not already in the cache.

    The file at the url `origin` is downloaded to the `dest_dir`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `dest_dir/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    Arguments:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        dest_dir: Location to store the files.
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        Path to the downloaded file
    """
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    os.makedirs(dest_dir, exist_ok=True)
    fpath = os.path.join(dest_dir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated because the "
                    + hash_algorithm
                    + " file hash does not match the original value of "
                    + file_hash
                    + " so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        print(f"Downloading: `{origin}` to `{fpath}`.")
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                # urlretrieve(origin, fpath, dl_progress)
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=os.path.basename(fpath),
                ) as t:
                    urlretrieve(origin, fpath, tqdm_hook(t))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if extract:
        _extract_archive(fpath, dest_dir, archive_format)
    return fpath


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Arguments:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        else:
            raise Exception("Invalid archive type.")

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def zip_dir(target_dir, save_dir):
    out = os.path.join(save_dir, os.path.basename(target_dir) + ".zip")
    logger.info(f"Zipping `{target_dir}` into `{out}`")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in list_files(target_dir):
            zf.write(f, f.replace(os.path.dirname(target_dir), "", 1))
