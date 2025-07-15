import re

pattern = re.compile(
    r'arxiv:'                                # now lowercase
    r'(?P<id>'
      r'(?:\d{4}\.\d{4,5}(?:v\d+)?)'
      r'|'
      r'(?:[a-z\-]+\/\d{7}(?:v\d+)?)'
    r')'
)

text = "arXiv preprint arXiv:2502.01456".lower()
m = pattern.search(text)
if m:
    print("Found arXiv ID:", m.group('id'))
