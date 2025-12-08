import json

def ms_to_hhmmss(ms: int) -> str:
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

# Paste your full JSON into this variable:
data = {
    "speaker_timeframes": [
    {
    "speakerName": "pallavika nesargikar",
    "start": 25287,
    "end": 32430
    }]
}

segments = []
for idx, seg in enumerate(data["speaker_timeframes"], start=1):
    segments.append({
        "segment_id": idx,
        "start": ms_to_hhmmss(seg["start"]),
        "end": ms_to_hhmmss(seg["end"]),
        "label": seg["speakerName"]
    })

output = {"segments": segments}

# write to file
with open("segments.json", "w") as f:
    json.dump(output, f, indent=2)
