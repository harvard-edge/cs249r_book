/* Measure how long a 1 kHz reader spends getting a proposal from a 20 Hz writer
   that sometimes stalls in the middle of a write (3 ms, every 4th write, injected).
   Three exchanges: a blocking mutex, a one-attempt try-lock (as ros2_control's
   RealtimeBuffer::readFromRT), and a one-attempt seqlock.
   Usage: mailbox VARIANT SECONDS out.csv   (VARIANT: mutex | trylock | seqlock)
   Each CSV row: tick, read time in ns, outcome (0 fresh, 1 kept prior, 2 rejected torn). */
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 288                       /* 16 waypoints x 9 axes x (position, velocity) */
typedef struct { long seq; double w[N]; } chunk_t;

static chunk_t slot, active;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static _Atomic unsigned long counter = 0;
static int variant; static volatile int running = 1;
static double ns_per_tick;

static uint64_t now(void) { return mach_absolute_time(); }
static uint64_t ms(double m) { return (uint64_t)(m * 1e6 / ns_per_tick); }

static void fill(chunk_t *c, long s, int stall) {
    c->seq = s;
    for (int i = 0; i < N / 2; i++) c->w[i] = s + i;
    if (stall) usleep(3000);         /* the writer is descheduled mid-write */
    for (int i = N / 2; i < N; i++) c->w[i] = s + i;
}

static void *writer(void *arg) {
    (void)arg; long s = 0; uint64_t next = now();
    while (running) {
        next += ms(50); mach_wait_until(next); s++;
        int stall = (s % 4 == 0);
        if (variant == 2) {
            atomic_fetch_add_explicit(&counter, 1, memory_order_acq_rel);   /* odd: write open */
            fill(&slot, s, stall);
            atomic_fetch_add_explicit(&counter, 1, memory_order_release);   /* even: published */
        } else {
            pthread_mutex_lock(&mtx); fill(&slot, s, stall); pthread_mutex_unlock(&mtx);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    variant = !strcmp(argv[1], "mutex") ? 0 : !strcmp(argv[1], "trylock") ? 1 : 2;
    int secs = atoi(argv[2]); long ticks = secs * 1000L;
    mach_timebase_info_data_t tb; mach_timebase_info(&tb); ns_per_tick = (double)tb.numer / tb.denom;
    {   /* the reader runs as a macOS time-constraint (real-time) thread: 1 ms period */
        thread_time_constraint_policy_data_t p = { (uint32_t)ms(1), (uint32_t)ms(0.1), (uint32_t)ms(0.5), 1 };
        thread_policy_set(mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY, (thread_policy_t)&p, THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    }
    FILE *f = fopen(argv[3], "w");
    pthread_t th; pthread_create(&th, NULL, writer, NULL);
    uint64_t next = now(); long last = 0;
    for (long k = 0; k < ticks; k++) {
        next += ms(1); mach_wait_until(next);
        uint64_t t0 = now(); int outcome = 1;
        if (variant == 0) {
            pthread_mutex_lock(&mtx); memcpy(&active, &slot, sizeof slot); pthread_mutex_unlock(&mtx);
        } else if (variant == 1) {
            if (pthread_mutex_trylock(&mtx) == 0) { memcpy(&active, &slot, sizeof slot); pthread_mutex_unlock(&mtx); }
        } else {
            static chunk_t snap;
            unsigned long a = atomic_load_explicit(&counter, memory_order_acquire);
            if (a & 1) outcome = 2;
            else {
                memcpy(&snap, &slot, sizeof slot);
                atomic_thread_fence(memory_order_acquire);
                unsigned long b = atomic_load_explicit(&counter, memory_order_relaxed);
                if (a != b) outcome = 2; else active = snap;
            }
        }
        uint64_t t1 = now();
        if (outcome != 2) outcome = (active.seq != last) ? 0 : 1;
        last = active.seq;
        fprintf(f, "%ld,%.0f,%d\n", k, (t1 - t0) * ns_per_tick, outcome);
    }
    running = 0; pthread_join(th, NULL); fclose(f); return 0;
}
