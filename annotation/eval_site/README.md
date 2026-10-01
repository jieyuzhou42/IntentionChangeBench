# Third-party evaluation site

A static site with a Firestore backend where people outside the team check our annotations:
user utterances, per-turn constraints and priorities, and the reference plans or products.

> **Deploy this to its own Firebase project.** `annotation/firebase` (the WebShop annotator)
> deploys to `intentflow-45722`. Deploying this directory there would replace that site's
> hosting and its Firestore rules.

## How it works

- **Codes.** The admin creates one code per person and domain (TravelPlanner or WebShop).
  Each code gets its own set of trajectories when it is created; least-reviewed trajectories
  are assigned first, so coverage stays balanced. Evaluators open `/?code=TP-XXXX-XXXX`; the
  admin dashboard is at `/?admin`.
- **Evaluator flow.** Consent, then a tutorial with practice questions and a qualification
  quiz (pass mark and attempts are set in `public/config/tutorial.js`), then the assigned
  trajectories. Each reviewed turn has up to three steps:
  1. answer from the conversation alone (naturalness, clarity, the evaluator's own reading);
  2. check the labeled change and the full requirement list;
  3. check the reference plan or product against the requirements.

  The labels stay hidden until step 1 is saved, and each step's answers lock once saved.
- **Records.** One record per evaluator and trajectory (`eval_judgments/<code>__<item_id>`),
  saved after every step, with the pool, rubric, and tutorial versions, timestamps, and the
  time each step was visible. Evaluators can stop and resume with the same link.
- **Admin dashboard.** Create and disable codes, follow coverage and progress, and export
  answers as JSON or as a CSV with one row per answer.

## Files

| Path | Purpose |
|---|---|
| `build_pool.py` | Draws the random pool and writes `public/data/` |
| `set_admin_key.py` | Installs the admin key in a checkout; see [Admin key](#admin-key) |
| `public/config/site.js` | Study settings: consent text, defaults, completion code |
| `public/config/rubric.js` | The questions asked for each turn |
| `public/config/tutorial.js` | Tutorial pages (also shown in the Guide panel), practice, quiz |
| `public/js/` | The app; `core.js` holds the pure logic covered by the tests |
| `firestore.rules`, `firebase.json` | Firebase access rules (without the key) and hosting config |
| `tests/` | `npm test` (Node) and `python3 -m unittest discover -s tests` |

## Try it locally

From the repository root, install the admin key once (the script asks for it), then build a
pool and serve the site:

```sh
python3 annotation/eval_site/set_admin_key.py
python3 annotation/eval_site/build_pool.py \
  --source 'annotation/data/travelplanner_v3_shards/shard_00[2378]_annotated.json' \
  --source 'data/simulation/webshop_v2_350_formal_priority_classified_shards/shard_00[1-5]_human_annotated.json' \
  --size travelplanner=12 --size webshop=12 --demo
python3 -m http.server 8790 --bind 127.0.0.1 --directory annotation/eval_site/public
```

Open <http://localhost:8790/?admin> and enter the admin key (see [Admin key](#admin-key)). On localhost, without
Firebase, the site runs in local test mode: a red banner shows, and codes and answers live
only in that browser. "Reset local test data" on the dashboard clears them.

`public/data/` is generated and not committed: the pool contains gold labels, and this
repository is public. Build it with `build_pool.py` where the source shards exist, or get the
folder privately from whoever built it.

## Building the real pool

```sh
python3 annotation/eval_site/build_pool.py \
  --source 'path/to/final_travelplanner_*.json' --source 'path/to/final_webshop_*.json' \
  --size travelplanner=60 --size webshop=60 --seed 20260930
```

- The draw depends only on the seed and the set of trajectories, not on file order.
  Duplicate instance ids are kept once, from the first file listed.
- Item ids are the domain plus the instance id, so they survive data moving between files.
- `--eval-turns N` reviews N sampled turns per trajectory instead of all of them. Every
  evaluator of a trajectory reviews the same turns.
- `--require-confirmed` skips trajectories with unconfirmed gold actions on reviewed turns.
- `--include-seed-turn` also reviews turn 0, which is shown as context by default.
- Evaluators never see model rollouts or predictions, annotator rationales, or review notes,
  including disclosed violations.

Rebuilding changes `pool_version`. The dashboard flags codes created for an older pool, so
create codes after the final pool is published.

**Sizing.** Codes needed per domain = trajectories × reviewers per trajectory ÷ trajectories
per person. For example, 60 × 3 ÷ 10 = 18 people; 120 × 2 ÷ 10 = 24. The dashboard shows this
for the current pool. A TravelPlanner trajectory has about six reviewed turns of three steps
each, so check pilot timing before fixing 10 per person.

## Deploying to Firebase

1. Create a new Firebase project. Enable **Authentication > Anonymous** and create a
   **Firestore** database.
2. In this directory, run `cp .firebaserc.example .firebaserc` and put in the new project
   id, or run `firebase use --add`.
3. Run `python3 set_admin_key.py` and enter the team key, which you get privately from the
   team. This writes `firestore.deploy.rules`. Without it, the deploy stops with a
   missing-file error.
4. Put the pool in `public/data/` (see [Try it locally](#try-it-locally)); the deploy uploads
   whatever is there. Then run `firebase deploy --only firestore:rules,hosting`. On Firebase
   Hosting the site finds its config at `/__/firebase/init.js`. Anywhere else, set
   `firebaseConfig` in `public/config/site.js`. Then open `/?admin` and enter the key.
5. Smoke test: create one code, finish one turn as that evaluator, check the document in
   `eval_judgments`, and download an export.

## Admin key

There is one fixed team key, the same locally and online. It is never committed, because
this repository is public; share it privately. `set_admin_key.py` writes it into two
git-ignored files, and neither is served with the website:

- `firestore.deploy.rules`: `firestore.rules` with the placeholder replaced. Firebase checks
  the key, and rules never reach the browser. If the template were deployed as it is, nobody
  could become an admin.
- `public/config/.local-admin.json`: read only in local test mode. Firebase Hosting skips
  dotfiles when deploying (`"**/.*"` in `firebase.json`), so this file is never online.

Each browser that enters the key stays signed in as admin. To change the key, rerun the
script, redeploy the rules, and delete the documents in `eval_admins` to sign out every
existing admin browser.

## Firestore collections

| Collection | Document | Written by |
|---|---|---|
| `eval_admins/{uid}` | A browser that entered the admin key | That browser |
| `eval_evaluators/{code}` | Domain, `item_ids`, `pool_version`, `disabled`, current `uid` | Admin; the evaluator may only claim it |
| `eval_progress/{code}` | Consent time and tutorial attempts | The evaluator |
| `eval_judgments/{code}__{item_id}` | Status, versions, timing, `answers_json` | The evaluator |

Answers are stored as a JSON string, because field names such as
`entities.entity_2.constraints.diet` are not safe as Firestore map keys. The admin export
expands them.

## Before launch

- [ ] Consent text, time estimate, pay, and completion code in `public/config/site.js`.
- [ ] Team review of the tutorial content, which is a draft condensed from
      `TRAVELPLANNER_ANNOTATION_PLAYBOOK.md` and `ANNOTATION_GUIDE.md`.
- [ ] Team review of the questions in `public/config/rubric.js`. Bump `version` in the rubric
      or tutorial whenever either changes.
- [ ] A pilot with 2–3 people to check timing and confusing items, then freeze the versions.
- [ ] An ethics board (IRB) exemption or not-human-subjects determination.

## Known limitations (v1)

- The Firestore rules have not been run against the Firebase emulator. Run the smoke test
  above before inviting anyone.
- Gold labels are in the static files, so a determined evaluator could read them in the
  browser's developer tools before answering. This is acceptable for invited evaluators;
  for open crowd recruitment, serve the gold through Firestore instead.
- A code is active in one browser at a time. Entering it elsewhere moves it to the new
  browser, and the old one can no longer save.
- Answers cannot be edited after a step is saved.
- Not built yet: hidden check items and seeded errors, Prolific ID capture and completion
  redirects, and an in-app interface tour.
