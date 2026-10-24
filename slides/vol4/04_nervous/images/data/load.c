/* Background load: one process per core streaming through 64 MB, so cores and memory are busy. */
#include <stdlib.h>
#include <string.h>
int main(void) {
    size_t n = 64u << 20; char *a = malloc(n), *b = malloc(n);
    memset(a, 1, n);
    for (;;) { memcpy(b, a, n); memcpy(a, b, n); }
}
