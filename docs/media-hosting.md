# Audio and other media for the book pages

Audio overviews (and, later, per-chapter audio) are too large for git: pre-commit
rejects files over 5 MB, and every binary committed stays in the history forever.
They are kept out of the repository and wired in through a small manifest.

## How it fits together

| Piece | Where | What it holds |
|---|---|---|
| Files | GitHub Release `audio-overviews` (rolling, not a book version) | The `.m4a` files, uploaded with content type `audio/mp4` |
| Manifest | `books/shared/media/audio-overviews.json` | One entry per file: id, volume, title, file name, duration, bytes, SHA-256, optional note and caption |
| Page block | `{{< audio-overview ID >}}` shortcode, `books/_extensions/mlsysbook-ext/audio-overview/` | Renders the Listen block for an id; renders nothing for an id not in the manifest |
| Deploy copy | `binder/tools/scripts/publish/fetch_audio_overviews.py`, called by `book-publish-live.yml` | Downloads each listed file into `vol{N}-site/assets/media/`, checking bytes and SHA-256 |

The Listen block lists two sources: the same-origin copy at
`/vol{N}/assets/media/<file>` (served as `audio/mp4`), then the release URL. Dev previews do
not run the copy step and play from the release URL. GitHub serves release assets as
`application/octet-stream` with an attachment header, which Chrome and Firefox play but
Safari may not, so the same-origin copy is the one readers should normally get.

A SHA-256 mismatch fails the publish job: it means the release asset is no longer the file
that was reviewed. A download error only warns, because the page still falls back to the
release URL.

## Add or replace an audio file

1. Name the file `mlsysbook-<id>-audio-overview-v<N>.m4a` (a new version gets a new name, so
   caches never serve a stale file).
2. Upload it with an explicit content type:
   ```bash
   RID=$(gh api repos/harvard-edge/cs249r_book/releases/tags/audio-overviews --jq .id)
   gh api -X POST "https://uploads.github.com/repos/harvard-edge/cs249r_book/releases/$RID/assets?name=<file>" \
     -H "Content-Type: audio/mp4" --input <file>
   ```
3. Add or update the manifest entry with `bytes` (`stat -f %z <file>`) and `sha256`
   (`shasum -a 256 <file>`).
4. Put `{{< audio-overview <id> >}}` on the page. A page one folder below the volume root
   passes `local="../assets/media"`.
5. Publish the book as usual. Old assets can be deleted from the release once no manifest
   entry names them.

## Captions

The default caption says the audio was reviewed by the author. An entry that has not been
reviewed by the author sets its own `caption` instead, so the page never claims a review that
did not happen.

## Moving to another host later

If the files move to object storage (for example Cloudflare R2 behind `media.mlsysbook.ai`),
change `base_url` in the manifest and re-upload the files there; the shortcode and pages do
not change.
