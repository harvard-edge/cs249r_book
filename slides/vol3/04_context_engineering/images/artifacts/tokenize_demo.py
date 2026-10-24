"""Real tokenizer run behind the 'volatile field' slide of Lecture 4.

Two consecutive requests differ only in a request id. With the id first,
the token sequences diverge early; with the id last, the long shared
prefix survives. Uses the GPT-2 byte-pair tokenizer (tokenizer.json from
huggingface.co/openai-community/gpt2) and the `tokenizers` package.
    usage: python3 tokenize_demo.py path/to/gpt2_tokenizer.json
"""
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
BODY = ("You are a repair agent. Preserve the public API. Stay inside the "
        "approved workspace. Do not deploy to production.\n"
        "Tools: read_file(path), run_test(name), apply_patch(diff).\n")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


for layout, make in [("id first", lambda r: f"request_id: {r}\n" + BODY),
                     ("id last", lambda r: BODY + f"request_id: {r}\n")]:
    a = tok.encode(make("7f3a")).ids
    b = tok.encode(make("7f3b")).ids
    k = first_diff(a, b)
    print(f"[{layout}] call 1: {len(a)} tokens, call 2: {len(b)} tokens")
    print(f"  call 1: {a[:8]}")
    print(f"  call 2: {b[:8]}")
    print(f"  first differing token: position {k}; shared prefix {k} of {len(a)}")
