import re

s = "text_123"
m = re.match(r'^text_(\d+)$', s)
if m:
    num = int(m.group(1))  # 123
    print(num)