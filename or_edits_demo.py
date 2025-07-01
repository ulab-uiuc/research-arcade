from openreview.api import OpenReviewClient

# 1. Instantiate (with credentials if you need private data)
client = OpenReviewClient(
    baseurl='https://api2.openreview.net'
)

note_id = "odjMSBSWRt"

ref_id = "VfUGyD5DpM"

note_edits = client.get_note_edits(ref_id, ref_id=True)

note = client.get_note()

for node_edit in note_edits:
    print("----------------------------")
    print(node_edit)
