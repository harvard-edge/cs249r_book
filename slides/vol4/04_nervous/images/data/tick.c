/* A 1 ms periodic task on a general-purpose OS: record how late each release wakes.
   Usage: tick N out.bin   (N periods of 1 ms; writes N doubles, lateness in microseconds) */
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    long n = atol(argv[1]);
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / tb.denom;
    uint64_t period = (uint64_t)(1e6 / ns_per_tick);   /* 1 ms in mach ticks */
    double *late = malloc(sizeof(double) * n);
    uint64_t next = mach_absolute_time() + period;
    for (long i = 0; i < n; i++) {
        mach_wait_until(next);
        uint64_t now = mach_absolute_time();
        late[i] = (double)(now - next) * ns_per_tick / 1000.0;
        next += period;
        if (now > next) next = now + period;   /* a release missed by more than a period restarts the schedule */
    }
    FILE *f = fopen(argv[2], "wb"); fwrite(late, sizeof(double), n, f); fclose(f);
    return 0;
}
