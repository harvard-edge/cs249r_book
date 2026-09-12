# Shared

Assets and tooling used by more than one site in the repository (the books, TinyTorch, MLSys·im, labs, kits, slides, instructors, StaffML, and the landing site). Anything used by only one project belongs in that project's folder instead.

| Folder | Contents |
|---|---|
| [`styles/`](styles/) | Brand tokens, shared SCSS partials, and per-project accent themes (see its README) |
| [`config/`](config/) | Common navbar and footer, `<head>` includes, ecosystem cards, redirect maps, and link-checker ignore lists |
| [`release/`](release/) | The release pill and release card shown across sites (see its README) |
| [`scripts/`](scripts/) | Cross-site build helpers: redirects, sitemap, internal link checks, build stamps, stats injection, and shared browser scripts |
| [`assets/`](assets/) | Images shared by several sites |

Some shared files are mirrored into individual projects so each site can build on its own. After editing `scripts/subscribe-modal.js` or another mirrored file, run `bash shared/scripts/sync-mirrors.sh`; the pre-commit check fails if the copies drift.
