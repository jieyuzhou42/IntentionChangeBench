from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from annotation import replay_server


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIREBASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = FIREBASE_DIR / "public"
DATA_DIR = PUBLIC_DIR / "data"
SHARD_DIR = (
    PROJECT_ROOT
    / "data"
    / "simulation"
    / "webshop_v2_350_formal_priority_classified_shards"
)
SHARD_MIN = 6
SHARD_MAX = 20


FIREBASE_BOOTSTRAP = r'''
    const nativeFetch = window.fetch.bind(window);
    const params = new URLSearchParams(window.location.search);
    const requestedShard = Number(params.get("shard") || 6);
    const shardIndex = Number.isInteger(requestedShard) && requestedShard >= 6 && requestedShard <= 20
      ? requestedShard
      : 6;
    const shardName = String(shardIndex).padStart(3, "0");
    const stateResponse = await nativeFetch(`/data/shard_${shardName}.json`, { cache: "no-store" });
    if (!stateResponse.ok) throw new Error(`Unable to load shard ${shardName}`);
    const state = await stateResponse.json();
    state.shard_index = shardIndex;
    state.shard_min = 6;
    state.shard_max = 20;
    state.shard_count = 20;
    state.annotation_output = "Firebase / webshop_annotations";

    await firebase.auth().signInAnonymously();
    const db = firebase.firestore();
    const annotations = db.collection("webshop_annotations");
    const annotationRef = instance => annotations.doc(instance.instance_id);
    const snapshots = await Promise.all(state.instances.map(instance => annotationRef(instance).get()));
    state.instances = state.instances.filter((instance, index) => {
      const snapshot = snapshots[index];
      if (!snapshot.exists) return true;
      const saved = snapshot.data();
      if (saved.deleted === true) return false;
      if (typeof saved.turns_json === "string") {
        const savedTurns = JSON.parse(saved.turns_json);
        if (Array.isArray(savedTurns) && savedTurns.length) instance.turns = savedTurns;
      } else if (Array.isArray(saved.turns) && saved.turns.length) {
        // Backward compatibility for documents created by the first deployment.
        instance.turns = saved.turns;
      }
      return true;
    });

    function cleanTurns(turns) {
      const cleaned = JSON.parse(JSON.stringify(turns));
      for (const turn of cleaned) {
        delete turn.rationales;
        for (const item of turn.env_feedback?.candidate_items || []) delete item.image_url;
      }
      return cleaned;
    }

    async function persistInstance(instance, extra = {}) {
      await annotationRef(instance).set({
        instance_id: instance.instance_id,
        shard_index: shardIndex,
        turns_json: JSON.stringify(cleanTurns(instance.turns || [])),
        deleted: false,
        updated_at: firebase.firestore.FieldValue.serverTimestamp(),
        ...extra
      });
    }

    function jsonResponse(data, status = 200) {
      return new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" }
      });
    }

    let pendingShard = shardIndex;
    window.fetch = async function(input, init = {}) {
      const url = typeof input === "string" ? input : input.url;
      const parsed = new URL(url, window.location.origin);
      const path = parsed.pathname;
      if (path === "/api/update_trajectory") {
        try {
          const payload = JSON.parse(init.body || "{}");
          const instance = state.instances[Number(payload.instance_index)];
          if (!instance || !Array.isArray(payload.turns) || !payload.turns.length) {
            return jsonResponse({ ok: false, error: "Invalid trajectory payload" }, 400);
          }
          instance.turns = payload.turns;
          await persistInstance(instance);
          return jsonResponse({ ok: true, turn_count: payload.turns.length });
        } catch (error) {
          return jsonResponse({ ok: false, error: error.message }, 500);
        }
      }
      const candidateMatch = path.match(/^\/api\/candidates\/(\d+)\/(\d+)$/);
      if (candidateMatch) {
        const instance = state.instances[Number(candidateMatch[1])];
        const turn = instance?.turns?.[Number(candidateMatch[2])];
        return jsonResponse({ ok: true, items: turn?.env_feedback?.candidate_items || [] });
      }
      const trajectoryMatch = path.match(/^\/api\/trajectories\/(\d+)$/);
      if (trajectoryMatch && String(init.method || "GET").toUpperCase() === "DELETE") {
        try {
          const instance = state.instances[Number(trajectoryMatch[1])];
          if (!instance) return jsonResponse({ ok: false, error: "Trajectory not found" }, 404);
          await persistInstance(instance, { deleted: true });
          return jsonResponse({ ok: true });
        } catch (error) {
          return jsonResponse({ ok: false, error: error.message }, 500);
        }
      }
      if (path === "/api/switch_shard") {
        pendingShard = Number(JSON.parse(init.body || "{}").shard);
        return jsonResponse({ ok: true });
      }
      if (path === "/api/shard_status") {
        return jsonResponse({ ok: true, shard_index: pendingShard });
      }
      return nativeFetch(input, init);
    };
'''


def load_instances(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_states() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shards = {
        index: load_instances(SHARD_DIR / f"shard_{index:03d}.json")
        for index in range(SHARD_MIN, SHARD_MAX + 1)
    }
    all_instances = [instance for instances in shards.values() for instance in instances]
    target_asins = replay_server.collect_raw_candidate_asins(all_instances)
    target_asins.update(replay_server.collect_webshop_goal_asins(all_instances))
    catalog_items = replay_server.load_replay_catalog_items(
        target_asins,
        replay_server.DEFAULT_ITEM_CACHE_PATH,
    )
    image_map = replay_server.load_catalog_images(
        target_asins,
        replay_server.DEFAULT_CACHE_PATH,
        scan_full_catalog=True,
    )

    for index, source_instances in shards.items():
        instances = copy.deepcopy(source_instances)
        for instance in instances:
            for turn in instance.get("turns") or []:
                feedback = turn.setdefault("env_feedback", {})
                feedback["candidate_items"] = replay_server.expanded_candidate_items(
                    turn, catalog_items
                )
        state = replay_server.prepare_state(instances, image_map)
        replay_server.attach_webshop_gold_items(state, catalog_items)
        state.update(
            {
                "shard_index": index,
                "shard_min": SHARD_MIN,
                "shard_max": SHARD_MAX,
                "shard_count": SHARD_MAX,
            }
        )
        output = DATA_DIR / f"shard_{index:03d}.json"
        output.write_text(
            json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def build_index() -> None:
    html = replay_server.HTML
    html = html.replace(
        '<button id="reload" class="primary">Reload</button>',
        '<button id="exportShard">Export Shard</button>\n'
        '        <button id="reload" class="primary">Reload</button>',
    )
    html = html.replace(
        "  <script>\n    const state = {{ state_json | safe }};",
        "  <script src=\"https://www.gstatic.com/firebasejs/11.10.0/firebase-app-compat.js\"></script>\n"
        "  <script src=\"https://www.gstatic.com/firebasejs/11.10.0/firebase-auth-compat.js\"></script>\n"
        "  <script src=\"https://www.gstatic.com/firebasejs/11.10.0/firebase-firestore-compat.js\"></script>\n"
        "  <script src=\"/__/firebase/init.js\"></script>\n"
        "  <script type=\"module\">\n"
        + FIREBASE_BOOTSTRAP,
    )
    html = html.replace(
        "Array.from({ length: state.shard_count }, (_, index) => {\n"
        "            const shard = index + 1;",
        "Array.from({ length: state.shard_max - state.shard_min + 1 }, (_, index) => {\n"
        "            const shard = state.shard_min + index;",
    )
    html = html.replace(
        "prevShardButton.disabled = state.shard_index <= 1;",
        "prevShardButton.disabled = state.shard_index <= state.shard_min;",
    ).replace(
        "nextShardButton.disabled = state.shard_index >= state.shard_count;",
        "nextShardButton.disabled = state.shard_index >= state.shard_max;",
    ).replace(
        "target < 1 || target > state.shard_count",
        "target < state.shard_min || target > state.shard_max",
    )
    export_handler = r'''
    document.getElementById("exportShard").addEventListener("click", () => {
      commitCurrentDraft();
      const blob = new Blob([JSON.stringify(state.instances, null, 2)], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `shard_${String(state.shard_index).padStart(3, "0")}_human_annotated.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    });
'''
    html = html.replace(
        '    document.getElementById("reload").addEventListener',
        export_handler + '\n    document.getElementById("reload").addEventListener',
    )
    html = html.replace("  </script>\n</body>", "  </script>\n</body>")
    if "{{ state_json" in html:
        raise RuntimeError("Firebase state bootstrap was not injected")
    if not re.search(r'<script type="module">\s+const nativeFetch', html):
        raise RuntimeError("Firebase module script was not generated")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    build_states()
    build_index()
    print(f"Built shards {SHARD_MIN}-{SHARD_MAX} in {PUBLIC_DIR}")


if __name__ == "__main__":
    main()
