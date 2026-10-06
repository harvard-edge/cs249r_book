import json
import time
import os

targets = {
    "Chapter 1": "d47c3a11-3d11-4de4-963f-3a0a23f4e5cc",
    "Chapter 2": "3ff3ad67-38d4-4ec4-a647-9a0e1464a8c8",
    "Chapter 3": "6f1bbcbb-c977-4781-848a-80b6194fee68",
    "Chapter 4": "791a8451-dd3a-4126-91d3-780324e2cfd7",
    "Chapter 5": "e1d6225d-57e1-47ff-991f-68b8554de8f6",
    "Chapter 6": "c03b6bef-9e58-4829-a6d7-ff64f1193b20",
    "Chapter 7": "89a36d5a-d8aa-4644-996f-477db99b7850",
    "Chapter 8": "6da820b2-2b38-46e7-a3da-723b4ddbec24",
    "Chapter 9": "d0390255-ae5b-4a15-b64c-42f567756b88",
    "Chapter 10": "ea516bbe-d89c-49bd-b596-f2f4d144a124",
    "Chapter 11": "7d2ba327-406d-49be-b348-8140d1749a15",
    "Chapter 12": "8f69d04f-e94c-41b9-ac4f-3fcc130551de",
    "Chapter 13": "16d71882-70e2-4246-a95c-da27e4f9bebd",
    "Chapter 14": "37ee7b06-f650-4de7-8238-0170e98f1783",
    "Chapter 15": "05a06411-c966-4734-9018-88b710e5633a",
    "Chapter 16": "6b7ef4c1-2029-4215-9afa-4f64bf172d1e",
    "Chapter 17": "482f0eab-aef0-4fec-adda-bcc260bf7093"
}

completed = set()

for i in range(60):
    for role, conv_id in targets.items():
        if role in completed:
            continue
        log_path = f'/Users/VJ/.gemini/antigravity-cli/brain/{conv_id}/.system_generated/logs/transcript_full.jsonl'
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, 'r') as log_f:
                lines = log_f.readlines()
            for line in reversed(lines):
                entry = json.loads(line)
                if entry.get('source') == 'MODEL' and entry.get('type') == 'PLANNER_RESPONSE':
                    tool_calls = entry.get('tool_calls', [])
                    for tc in tool_calls:
                        if tc.get('name') == 'send_message':
                            completed.add(role)
                            break
                    if role in completed:
                        break
        except Exception:
            pass
    
    if len(completed) == len(targets):
        print("ALL_DONE")
        break
    time.sleep(5)

if len(completed) != len(targets):
    print(f"Missing: {set(targets.keys()) - completed}")

with open("final_combined_audits.md", "w") as f:
    for role, conv_id in targets.items():
        log_path = f'/Users/VJ/.gemini/antigravity-cli/brain/{conv_id}/.system_generated/logs/transcript_full.jsonl'
        report = ""
        try:
            with open(log_path, 'r') as log_f:
                lines = log_f.readlines()
            for line in reversed(lines):
                entry = json.loads(line)
                if entry.get('source') == 'MODEL' and entry.get('type') == 'PLANNER_RESPONSE':
                    tool_calls = entry.get('tool_calls', [])
                    for tc in tool_calls:
                        if tc.get('name') == 'send_message':
                            args = tc.get('args', {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            report = args.get('Message', '')
                            break
                    if report:
                        break
        except Exception:
            pass
        if report:
            f.write(f"\n\n# {role}\n\n")
            f.write(report)
