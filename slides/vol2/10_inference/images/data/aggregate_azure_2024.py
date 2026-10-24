import sys, collections
# stdin: TIMESTAMP,ContextTokens,GeneratedTokens ; aggregate per 10-minute bin
cnt = collections.Counter(); ctx = collections.Counter(); gen = collections.Counter()
first = True
for line in sys.stdin:
    if first:
        first = False
        continue
    p = line.rstrip("\n").split(",")
    if len(p) < 3:
        continue
    ts = p[0]
    key = ts[:15]  # YYYY-MM-DD HH:M (10-minute bin)
    cnt[key] += 1
    try:
        ctx[key] += int(p[1]); gen[key] += int(p[2])
    except ValueError:
        pass
with open(sys.argv[1], "w") as f:
    f.write("bin10min,requests,context_tokens,generated_tokens\n")
    for k in sorted(cnt):
        f.write(f"{k}0,{cnt[k]},{ctx[k]},{gen[k]}\n")
