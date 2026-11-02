# Original two-phase checkpoint staging

The `original-checkpoint-staging` factory enhances the verified source SVG for
`vol2:fig-storage-checkpoint-staging`. The static artwork is unchanged. Its four
concurrent lanes, time-axis landmarks, stage order, resource colors, and commit
boundary remain recognizable in Explore.

This is the local-NVMe-first schedule in the figure, not the separate
host-DRAM-first asynchronous variant discussed afterward. Training pauses for
every required local write and the completion barrier. It resumes on separate
live state while an immutable local generation is copied to the parallel file
system. The illustrated required set comprises rank shards A and B; symbolic
generation g1 captures step i while g0 is already committed. The first resumed
live step is i+1. No numerical payload, hardware rate, or measured duration is
introduced.

Nine paused states establish the consistent cut, complete the local writes,
resume live training, start copying, complete A's durable write, complete B's
durable write, verify the complete set, publish the manifest atomically, and
release local staging. The snapshot identity stays i while the live state
advances. A durable receipt for only A cannot make the generation complete;
all persisted shards still cannot make g1 selectable before verification and
manifest publication. Local staging release does not imply deletion of g0.

Immutable shard labels are drawn inside the source local and durable blocks.
The source copy block contains pending/durable receipts. Schematic operation
fills illustrate a write in progress while its receipt remains pending until
the endpoint. These fills do not estimate bytes or elapsed I/O time. Verification
marks the required A/B set; a pending manifest explicitly names A@i and B@i
before the restore pointer changes at the publication endpoint.

Compact same-shard status cards keep A and B simultaneously readable on phones.
They supplement the original timeline. Source lane names remain readable as the
wide timeline scrolls locally, and the current operation is centered. The
source future Restorable bracket remains pale until publication. Shared controls
supply keyboard stepping, pause/resume, reduced motion, expansion, and required
build-time pacing. No local timing default is supplied.

## Focused verification

```sh
python3 books/_prototypes/vol1-web-animations/original-checkpoint-staging/check-browser.py
python3 books/_prototypes/vol1-web-animations/original-checkpoint-staging/check-browser.py --engine webkit
```

The fixture mounts the factory inside the actual rendered data-storage chapter
with its theme, figure anchor, caption, and surrounding material. It verifies
all endpoints, immutable payload identity, no early durable receipts, no early
commit pointer, manual stopping, exact menu reconstruction, keyboard controls,
normal/reduced motion, mid-operation pause/resume, expansion, mobile visibility,
local scrolling, and no added chapter overflow. Distinct motion screenshots
cover copying, A/B durable writes, verification, and manifest publication.
Browser print restores the retained original image, and a JavaScript-disabled
page retains its complete static figure. The canonical SVG hash is unchanged.
Fixture timing is explicit accelerated input, not production pacing.

Screenshots are written to `/tmp/original-checkpoint-review`. This isolated
fixture does not replace the packaged Quarto chapter or PDF/EPUB acceptance
build; those remain separate integration checks. Reviews here are agent reviews,
not a claim of human learner testing.
