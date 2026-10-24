#!/bin/sh
cd "$(dirname "$0")"
uptime > conditions.txt
./tick 60000 light.bin
pids=""
for i in $(seq 1 18); do ./load & pids="$pids $!"; done
sleep 2
uptime >> conditions.txt
./tick 60000 heavy.bin
for p in $pids; do kill $p; done
uptime >> conditions.txt
echo done
