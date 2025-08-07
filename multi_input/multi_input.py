import re

class MultiInput:

  """
  Download the compresses latex source code given the bib, arxiv id or arxiv link
  """
    
  def extract_bib_from_string(self, bib_str: str) -> dict:
      """
      Parse a single-entry BibTeX string into a Python dict.

      Example input (bib_str):
          @misc{1802.08773,
            Author  = {Jiaxuan You and Rex Ying and Xiang Ren and William L. Hamilton and Jure Leskovec},
            Title   = {GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models},
            Year    = {2018},
            Eprint  = {arXiv:1802.08773},
          }

      Returns a dict like:
      {
        "ENTRYTYPE": "misc",
        "ID":        "1802.08773",
        "author":    "Jiaxuan You and Rex Ying and ...",
        "title":     "GraphRNN: Generating Realistic ...",
        "year":      "2018",
        "eprint":    "arXiv:1802.08773"
      }

      Raises:
        ValueError if the string does not look like a well‐formed single BibTeX entry.
      """
      s = bib_str.strip()

      # 1) Match the “@type{key,” header:
      m_header = re.match(r'@\s*(?P<type>\w+)\s*{\s*(?P<key>[^,]+)\s*,', s)
      if not m_header:
          raise ValueError(
              "Invalid BibTeX entry header. "
              "Expected something like: @misc{<ID>, … }"
          )

      entry_type = m_header.group("type").lower()
      entry_key  = m_header.group("key").strip()

      # 2) Find the very first “{” in the entire string (that opens the entry),
      #    then scan forward to locate its matching “}” by counting nested braces.
      first_brace = s.find('{')
      if first_brace == -1:
          raise ValueError("Could not find the opening brace '{' for the BibTeX entry.")

      brace_level = 0
      end_brace   = -1
      for i, ch in enumerate(s[first_brace:], start=first_brace):
          if ch == '{':
              brace_level += 1
          elif ch == '}':
              brace_level -= 1
              if brace_level == 0:
                  end_brace = i
                  break

      if end_brace == -1:
          raise ValueError("Could not find matching closing '}' for the BibTeX entry.")

      # Everything between the first “{” and its matching “}” is the block:
      block = s[first_brace+1 : end_brace].strip()

      # 3) That block begins with “<ID>,” (we already extracted the ID). Remove up to the first comma:
      comma_pos = block.find(',')
      if comma_pos == -1:
          raise ValueError("Expected a comma after the entry key inside the braces.")
      fields_str = block[comma_pos+1 :].strip()

      # 4) Split fields_str on commas that occur at brace‐level zero.
      fields = {}
      current = ""
      level = 0

      for ch in fields_str:
          if ch == '{':
              level += 1
              current += ch
          elif ch == '}':
              level -= 1
              current += ch
          elif ch == ',' and level == 0:
              # We have a complete “key = value” chunk
              chunk = current.strip()
              current = ""
              if chunk:
                  m_field = re.match(r'(?P<k>\w+)\s*=\s*(?P<v>.+)$', chunk)
                  if m_field:
                      key = m_field.group("k").strip().lower()
                      val = m_field.group("v").strip()
                      # Strip outer braces or quotes if present:
                      if val.startswith("{") and val.endswith("}"):
                          val = val[1:-1].strip()
                      elif val.startswith('"') and val.endswith('"'):
                          val = val[1:-1].strip()
                      fields[key] = val
              # else: skip empty chunk
          else:
              current += ch

      # If anything remains in current after the loop, that's the last field:
      last_chunk = current.strip()
      if last_chunk:
          m_field = re.match(r'(?P<k>\w+)\s*=\s*(?P<v>.+)$', last_chunk)
          if m_field:
              key = m_field.group("k").strip().lower()
              val = m_field.group("v").strip()
              if val.startswith("{") and val.endswith("}"):
                  val = val[1:-1].strip()
              elif val.startswith('"') and val.endswith('"'):
                  val = val[1:-1].strip()
              fields[key] = val

      # 5) Build the final dict:
      bib_dict = {
          "ENTRYTYPE": entry_type,
          "ID":        entry_key
      }
      bib_dict.update(fields)
      return bib_dict


  def arxiv_url_to_id(self, arxiv_url: str) -> str:
      """
      Extract the arXiv ID (with optional version) from a full arXiv URL.

      Supported URL patterns include (but aren’t limited to):
        - https://arxiv.org/abs/1802.08773
        - http://arxiv.org/abs/1802.08773v2
        - https://arxiv.org/pdf/1802.08773
        - https://arxiv.org/pdf/1802.08773.pdf
        - http://arxiv.org/pdf/1802.08773v3.pdf
        - https://arxiv.org/abs/math/0301234
        - https://arxiv.org/pdf/math/0301234
        - https://arxiv.org/pdf/math.GT/0301234v2.pdf

      Returns:
        The “bare” arXiv identifier (e.g. "1802.08773", "1802.08773v2",
        "math/0301234" or "math.GT/0301234v1").

      Raises:
        ValueError if the URL doesn’t match any known arXiv pattern.
      """
      # 1) Strip off query strings or fragments:
      url = arxiv_url.split('?')[0].split('#')[0]

      # 2) Regex patterns for “new” vs. “old” ID formats,
      #    allowing /pdf/<id> (with or without “.pdf”) and /abs/<id>.

      patterns = [
          # — New‐style “abs” URL (e.g. /abs/1802.08773 or /abs/1802.08773v2)
          r"""arxiv\.org/
              abs/
              (?P<id>\d{4}\.\d{4,5}(?:v\d+)?)$
            """,

          # — New‐style “pdf” URL (e.g. /pdf/1802.08773 or /pdf/1802.08773.pdf or /pdf/1802.08773v2.pdf)
          r"""arxiv\.org/
              pdf/
              (?P<id>\d{4}\.\d{4,5}(?:v\d+)?)
              (?:\.pdf)?$
            """,

          # — Old‐style “abs” URL (e.g. /abs/math/0301234 or /abs/math.GT/0301234v2)
          r"""arxiv\.org/
              abs/
              (?P<id>[A-Za-z\-]+(?:\.[A-Za-z\-]+)?/\d{7}(?:v\d+)?)$
            """,

          # — Old‐style “pdf” URL (e.g. /pdf/math/0301234 or /pdf/math/0301234.pdf or /pdf/math.GT/0301234v2.pdf)
          r"""arxiv\.org/
              pdf/
              (?P<id>[A-Za-z\-]+(?:\.[A-Za-z\-]+)?/\d{7}(?:v\d+)?)
              (?:\.pdf)?$
            """
      ]

      for pat in patterns:
          m = re.search(pat, url, re.VERBOSE)
          if m:
              return m.group('id')

      raise ValueError(f"Could not extract arXiv ID from URL: {arxiv_url}")


  def extract_arxiv_id_from_bib(self, bib_entry: dict) -> str:
      """
      Given a single Bib entry (as a JSON‐style dict), return its arXiv ID (with version, if any).
      If no Eprint/eprint field is found, raises a KeyError.
      If the Eprint value isn’t in the form "arXiv:<id>", returns the raw field as a fallback.

      Example input dict:
        {
          "ENTRYTYPE": "misc",
          "ID":       "1802.08773",
          "author":   "...",
          "title":    "...",
          "year":     "2018",
          "Eprint":   "arXiv:1802.08773"
        }

      Returns:
        The bare arXiv identifier, e.g. "1802.08773" or "1802.08773v2".

      Raises:
        KeyError if neither "Eprint" nor "eprint" is present.
      """
      # Try to find any key named "Eprint" or "eprint" (case‐insensitive would be more robust):
      for key in ("Eprint", "eprint"):
          if key in bib_entry:
              raw = bib_entry[key].strip()
              # If it starts with "arXiv:" (case‐insensitive), strip that off:
              m = re.match(r'(?i)arxiv\s*:\s*(?P<id>.+)', raw)
              if m:
                  return m.group("id").strip()
              else:
                  # Fallback: return whatever is in Eprint if it doesn’t match "arXiv:<id>"
                  return raw

      # If we get here, neither "Eprint" nor "eprint" was in the dict:
      raise KeyError("No Eprint/eprint field found in this Bib entry")
  

  def extract_arxiv_id(self, input: str, input_type: str) -> str:
        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = self.extract_bib_from_string(input)
            arxiv_id = self.extract_arxiv_id_from_bib(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = self.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")
        return arxiv_id