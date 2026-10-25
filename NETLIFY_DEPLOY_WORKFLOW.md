# Netlify Deploy Workflow

This project is deployed at:

https://musicva.netlify.app

Netlify site id:

`270538ac-58a7-4e2c-8e36-161c5cb3cdef`

## What Worked Before

The successful deployment was done from the user's macOS Terminal, not from the Codex sandbox.

Working sequence:

```bash
cd "/Users/esther/Desktop/New project"
npx netlify login
npx netlify deploy --prod --dir=netlify_publish
```

During `netlify login`, the browser opened Netlify authorization, the user granted access, then the CLI reported login success.

During deployment, the folder was linked to the Netlify project `musicva`, build ran from `netlify.toml`, and production deploy completed at:

```text
https://musicva.netlify.app
```

## Why Codex Often Fails To Deploy

The common failure is not usually the project configuration.

Observed blockers from Codex:

- `npx netlify ...` often fails with `ENOTFOUND registry.npmjs.org`.
- Netlify API calls can intermittently fail with `Could not resolve host: api.netlify.com`.
- The cached Netlify CLI can start, but the Codex sandbox cannot write:

```text
/Users/esther/Library/Preferences/netlify/config.json.tmp-*
```

This produces:

```text
EPERM: operation not permitted
```

The user's normal Terminal is less restricted, so it is the reliable deployment path when Codex network or filesystem permissions block deployment.

## Required Workflow For Every Platform Change

Every annotation/QC platform change must be synchronized in both places:

1. Local source files under `annotation_tool/`.
2. Netlify publish package under `netlify_publish/`.
3. Live Netlify site at `https://musicva.netlify.app`.

After editing source files, always run:

```bash
cd "/Users/esther/Desktop/New project"
node --check annotation_tool/app.js
git diff --check -- annotation_tool/app.js annotation_tool/index.html annotation_tool/styles.css
python3 annotation_tool/build_netlify_publish.py
```

Then deploy:

```bash
npx netlify deploy --prod --dir=netlify_publish
```

If Codex cannot deploy because of DNS, npm, Netlify API, or sandbox permission errors, explicitly tell the user:

- local files are updated,
- `netlify_publish/` is updated,
- live Netlify is not updated yet,
- and the exact command the user must run.

Do not claim the change is complete until the live site is verified.

## Required Verification Before Reporting Done

Verification must cover the full path from frontend interaction to backend Supabase effect where applicable. Static checks alone are not enough.

Minimum verification:

1. Open or fetch the live URL:

```bash
curl -I https://musicva.netlify.app
```

2. Confirm the deployed assets contain the expected UI/code change:

```bash
curl -sS https://musicva.netlify.app | rg "expected text or element id"
```

3. For UI behavior, verify on the live page, not only `file://`.

4. For database-writing changes, perform a safe end-to-end check:

- Click the frontend control.
- Confirm the page shows the correct submitting/success state.
- Query Supabase or refresh the page to confirm the database-backed state changed.
- If the test modifies real data, restore the original value after verification when appropriate.

Examples of checks for this QC platform:

- `标为未质检` must clear `section_reviews` and `human_annotations`, verify
  `human_annotations_readable` is empty for the song, then update `songs.status`
  to `pending_review`.
- `提交整首歌` must update `songs.status` to `reviewed`.
- Left summary cards must reflect the live database state after refresh.
- Clicking `已质检` must select and display the first matching reviewed song instead of leaving the section area blank.

## Current Known Good Local Preconditions

The project has a linked Netlify state file:

```text
.netlify/state.json
```

with site id:

```text
270538ac-58a7-4e2c-8e36-161c5cb3cdef
```

The Netlify build config is:

```toml
[build]
command = "python3 annotation_tool/build_netlify_publish.py"
publish = "netlify_publish"

[[redirects]]
from = "/*"
to = "/index.html"
status = 200
```
