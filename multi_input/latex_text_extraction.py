import os
import zipfile
import tempfile

# This import is only needed if you want to support .7z archives.
# Make sure you have py7zr installed (pip install py7zr). 
try:
    import py7zr
except ImportError:
    py7zr = None


class latex_text_extraction:
    """
    Extract the raw .tex contents from a compressed .zip or .7z archive.

    Usage:
        extractor = latex_text_extraction()
        tex_dict = extractor.text_extraction("/path/to/archive.zip")
        # tex_dict is a dict where keys are archive‐internal paths (like "main.tex"),
        # and values are the full text of each .tex file inside.

    Raises:
      - FileNotFoundError       if the given path doesn’t exist.
      - ValueError              if the extension isn’t .zip or .7z.
      - RuntimeError            if py7zr isn’t installed but a .7z is passed.
      - zipfile.BadZipFile      if the .zip is corrupted.
      - py7zr.exceptions.tarx7zError if the .7z is corrupted.
    """

    def text_extraction(self, path: str) -> dict:
        # 1) Check that the file exists:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: '{path}'")

        # 2) Determine extension:
        _, ext = os.path.splitext(path)
        ext = ext.lower().lstrip(".")

        if ext == "zip":
            return self._extract_from_zip(path)
        elif ext == "7z":
            return self._extract_from_7z(path)
        else:
            raise ValueError(f"Unsupported archive format: '.{ext}'. Only .zip and .7z are supported.")

    def _extract_from_zip(self, zip_path: str) -> dict:
        """
        Open a .zip file, locate all .tex entries, and return a dict:
            { internal_path: content_string, … }
        """
        tex_contents = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Iterate over all files in the archive:
            for member in zf.namelist():
                if member.lower().endswith(".tex"):
                    # Read bytes, decode as utf-8 (or fallback to latin1 if decoding fails)
                    raw_bytes = zf.read(member)
                    try:
                        text = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_bytes.decode("latin-1")
                    tex_contents[member] = text
        return tex_contents

    def _extract_from_7z(self, sevenz_path: str) -> dict:
        """
        Open a .7z file (requires py7zr), locate all .tex entries,
        and return a dict: { internal_path: content_string, … }
        """
        if py7zr is None:
            raise RuntimeError("py7zr is not installed. Install it via 'pip install py7zr' to handle .7z files.")

        tex_contents = {}
        # py7zr allows streaming read of individual files without extracting to disk:
        with py7zr.SevenZipFile(sevenz_path, mode="r") as archive:
            all_info = archive.readall()  # returns a dict: { filename: io.BytesIO, … }
            for member_name, bio in all_info.items():
                if member_name.lower().endswith(".tex"):
                    raw_bytes = bio.read()
                    try:
                        text = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_bytes.decode("latin-1")
                    tex_contents[member_name] = text
        return tex_contents
