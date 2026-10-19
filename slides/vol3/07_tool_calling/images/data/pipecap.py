import os, fcntl
r, w = os.pipe()
fl = fcntl.fcntl(w, fcntl.F_GETFL); fcntl.fcntl(w, fcntl.F_SETFL, fl | os.O_NONBLOCK)
n = 0
try:
    while True:
        n += os.write(w, b"x" * 1024)
except BlockingIOError:
    pass
print("bytes written before the pipe blocked:", n)
