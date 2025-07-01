import or_edits_demo
from datetime import datetime

client = or_edits_demo.api.OpenReviewClient(baseurl='https://api2.openreview.net')
note_edits = client.get_note_edits(note_id="odjMSBSWRt")
