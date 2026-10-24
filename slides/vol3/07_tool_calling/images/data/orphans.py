import subprocess, time, os, signal
def count(tag):
    out = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    return len(out)
for mode in ("shell only", "process group"):
    tag = f"sleep 31{4 if mode=='shell only' else 5}"
    p = subprocess.Popen(["bash", "-c", f"{tag} & {tag} & {tag} & wait"], process_group=0)
    time.sleep(0.5)
    before = count(tag)
    if mode == "shell only": p.send_signal(signal.SIGTERM)
    else: os.killpg(p.pid, signal.SIGTERM)
    p.wait(); time.sleep(0.5)
    after = count(tag)
    print(f"SIGTERM to {mode}: {before} sleep workers before, {after} still running after")
    os.killpg(p.pid, signal.SIGKILL) if after else None
