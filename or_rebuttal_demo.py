from openreview.api import OpenReviewClient

client = OpenReviewClient(baseurl='https://api2.openreview.net')
paper_forum = 'odjMSBSWRt'

all_forum_notes = client.get_notes(forum=paper_forum)

rebuttals = []
for note in all_forum_notes:
    print("-----")
    note_json = note.to_json()
    print(note)
    inv = note_json.get('invitation', '')
    if inv.endswith('/Rebuttal'):
        rebuttals.append(note_json)

for r in rebuttals:
    ts = r['tcdate'] / 1000
    text = r['content'].get('rebuttal', r['content'].get('comment', '[no text]'))
    print(f"Rebuttal (inv={r['invitation']}) by {r['signatures']} at {ts}:")
    print(text)
    print('-' * 40)
    