// Admin dashboard: create evaluator codes, follow progress and coverage, export answers.

import {
  CSV_COLUMNS,
  countSummary,
  coverageCounts,
  evaluatorProgress,
  judgmentRows,
  makeCode,
  parseAnswers,
  planAssignments,
  toCsv,
} from "./core.js";
import { copyText, details, download, formatTime, h, mount, toast } from "./ui.js";

const fileStamp = () => new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const linkFor = code => `${location.origin}${location.pathname}?code=${encodeURIComponent(code)}`;

export async function showAdmin(ctx, newCodes = []) {
  mount(ctx.root, h("p", { class: "muted" }, "Loading the dashboard…"));
  try {
    const [evaluators, judgments, progress] = await Promise.all([
      ctx.store.listEvaluators(),
      ctx.store.listJudgments(),
      ctx.store.listProgress(),
    ]);
    evaluators.sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)) || a.code.localeCompare(b.code));
    const byCode = {};
    for (const judgment of judgments) (byCode[judgment.code] ||= {})[judgment.item_id] = judgment;
    const data = { evaluators, judgments, byCode, progress: Object.fromEntries(progress.map(p => [p.code, p])) };
    mount(
      ctx.root,
      h("div", { class: "item-bar" }, h("h1", {}, "Admin"), h("button", { onclick: () => showAdmin(ctx) }, "Refresh")),
      poolCard(ctx, data),
      createCard(ctx, data),
      newCodes.length ? newCodesCard(newCodes) : null,
      coverageCard(ctx, data),
      evaluatorsCard(ctx, data),
      exportCard(ctx, data),
    );
  } catch (error) {
    console.error(error);
    mount(ctx.root, h("section", { class: "card narrow" }, h("h2", {}, "Could not load the dashboard"), h("p", {}, error.message)));
  }
}

function activeCodes(ctx, data, domain) {
  return data.evaluators.filter(e => e.domain === domain && !e.disabled && e.pool_version === ctx.pool.pool_version);
}

function tutorialStatus(progress, tutorial) {
  if (!progress?.consent_at) return "Not started";
  if (progress.tutorial_passed_at) return "Passed";
  const attempts = progress.tutorial_attempts?.length || 0;
  if (attempts >= tutorial.maxAttempts) return "Did not pass";
  return attempts ? `${attempts} failed attempt${attempts > 1 ? "s" : ""}` : "In tutorial";
}

function poolCard(ctx, data) {
  const { pool, site } = ctx;
  const stale = data.evaluators.filter(e => e.pool_version !== pool.pool_version && !e.disabled);
  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Pool"),
    h(
      "p",
      {},
      h("code", {}, pool.pool_version),
      pool.demo ? h("span", { class: "badge warn" }, "demo data") : null,
      h("span", { class: "muted" }, ` built ${formatTime(pool.created_at)} · seed ${pool.seed} · turns reviewed per trajectory: ${pool.eval_turns}`),
    ),
    h(
      "ul",
      {},
      Object.entries(pool.domains).map(([domain, entry]) => {
        const turns = entry.items.reduce((sum, item) => sum + item.eval_turns.length, 0);
        return h("li", {}, `${site.domainLabel[domain] || domain}: ${entry.items.length} trajectories drawn from ${entry.eligible} eligible, ${turns} turns to review`);
      }),
    ),
    stale.length
      ? h(
          "p",
          { class: "warn-text" },
          `${stale.length} active codes were created for a different pool version, so their trajectories may be missing. Disable them and create new codes.`,
        )
      : null,
    details("Source files", h("ul", {}, pool.sources.map(source => h("li", {}, `${source.path} (sha256 ${source.sha256.slice(0, 12)}…)`)))),
    h(
      "p",
      { class: "muted" },
      `Rubric ${ctx.rubric.version} · tutorial ${ctx.tutorial.version} · storage: ${ctx.store.mode === "local" ? "local test mode (this browser only)" : "Firebase"}`,
    ),
  );
}

function createCard(ctx, data) {
  const domains = Object.keys(ctx.pool.domains);
  const domain = h("select", {}, domains.map(d => h("option", { value: d }, ctx.site.domainLabel[d] || d)));
  const count = h("input", { type: "number", min: 1, max: 200, value: 1 });
  const perEvaluator = h("input", { type: "number", min: 1, value: ctx.site.defaultItemsPerEvaluator });
  const label = h("input", { type: "text", placeholder: "Optional, for example: pilot" });
  const plan = h("p", { class: "muted" });
  const button = h("button", { class: "primary" }, "Create codes");

  const updatePlan = () => {
    const poolSize = ctx.pool.domains[domain.value].items.length;
    const per = Number(perEvaluator.value);
    const target = ctx.site.targetEvaluatorsPerItem;
    const active = activeCodes(ctx, data, domain.value).length;
    plan.textContent =
      per > poolSize
        ? `This pool has only ${poolSize} trajectories.`
        : `${poolSize} trajectories at ${per} per person: ${Math.ceil((poolSize * target) / per)} codes give every trajectory ${target} reviewers. Active codes for this domain so far: ${active}.`;
  };
  for (const input of [domain, count, perEvaluator]) input.addEventListener("input", updatePlan);
  updatePlan();

  button.addEventListener("click", async () => {
    const n = Number(count.value);
    const per = Number(perEvaluator.value);
    if (!Number.isInteger(n) || n < 1 || n > 200) {
      toast("Number of codes must be a whole number from 1 to 200.", "error");
      return;
    }
    button.disabled = true;
    try {
      const itemIds = ctx.pool.domains[domain.value].items.map(item => item.item_id);
      const existing = activeCodes(ctx, data, domain.value).map(e => e.item_ids);
      const plans = planAssignments(itemIds, existing, n, per);
      const taken = new Set(data.evaluators.map(e => e.code));
      const createdAt = new Date().toISOString();
      const records = plans.map(item_ids => {
        let code;
        do code = makeCode(ctx.site.codePrefix[domain.value] || domain.value.slice(0, 2).toUpperCase());
        while (taken.has(code));
        taken.add(code);
        return {
          code,
          label: label.value.trim(),
          domain: domain.value,
          item_ids,
          pool_version: ctx.pool.pool_version,
          created_at: createdAt,
          disabled: false,
        };
      });
      await ctx.store.createEvaluators(records);
      toast(`Created ${records.length} code${records.length > 1 ? "s" : ""}.`);
      await showAdmin(ctx, records.map(record => record.code));
    } catch (error) {
      console.error(error);
      toast(error.message, "error");
      button.disabled = false;
    }
  });

  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Create evaluator codes"),
    h(
      "p",
      { class: "muted" },
      "Each code is one person reviewing one domain. Trajectories are assigned when the code is created, least-reviewed first, so coverage stays balanced.",
    ),
    h(
      "div",
      { class: "form-grid" },
      h("label", {}, "Domain", domain),
      h("label", {}, "Number of codes", count),
      h("label", {}, "Trajectories per person", perEvaluator),
      h("label", {}, "Note", label),
    ),
    plan,
    h("div", { class: "actions" }, button),
  );
}

function newCodesCard(codes) {
  const text = codes.map(code => `${code}\t${linkFor(code)}`).join("\n");
  return h(
    "section",
    { class: "card highlight" },
    h("h2", {}, `New codes (${codes.length})`),
    h("p", { class: "muted" }, "Send each person their own link. Anyone with a code can answer as that evaluator, so share codes privately."),
    h("textarea", { class: "mono", readonly: true, rows: Math.min(10, codes.length + 1), value: text }),
    h("div", { class: "actions" }, h("button", { onclick: () => copyText(text, "Codes and links copied.") }, "Copy all")),
  );
}

function coverageCard(ctx, data) {
  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Coverage"),
    Object.entries(ctx.pool.domains).map(([domain, entry]) => {
      const ids = entry.items.map(item => item.item_id);
      const active = activeCodes(ctx, data, domain);
      const assigned = coverageCounts(ids, active.map(e => e.item_ids));
      const completed = coverageCounts(
        ids,
        active.map(e => e.item_ids.filter(id => data.byCode[e.code]?.[id]?.status === "completed")),
      );
      const a = countSummary(assigned);
      const c = countSummary(completed);
      return h(
        "div",
        { class: "coverage" },
        h("h3", {}, ctx.site.domainLabel[domain] || domain),
        h(
          "p",
          {},
          `${ids.length} trajectories · active codes: ${active.length} · each trajectory assigned to ${a.min}–${a.max} people and completed by ${c.min}–${c.max}.`,
        ),
        details(
          "Per trajectory",
          h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              { class: "grid" },
              h("thead", {}, h("tr", {}, ["Trajectory", "Turns to review", "Assigned", "Completed"].map(t => h("th", {}, t)))),
              h(
                "tbody",
                {},
                entry.items.map(item =>
                  h(
                    "tr",
                    {},
                    h("td", { class: "mono" }, item.item_id),
                    h("td", {}, item.eval_turns.length),
                    h("td", {}, assigned.get(item.item_id)),
                    h("td", {}, completed.get(item.item_id)),
                  ),
                ),
              ),
            ),
          ),
        ),
      );
    }),
  );
}

function codeRows(ctx, data) {
  return data.evaluators.map(e => {
    const progress = evaluatorProgress(e, data.byCode[e.code] || {});
    return {
      code: e.code,
      label: e.label || "",
      domain: e.domain,
      link: linkFor(e.code),
      trajectories: e.item_ids.length,
      completed: progress.completed,
      in_progress: progress.started,
      tutorial: tutorialStatus(data.progress[e.code], ctx.tutorial),
      disabled: e.disabled ? "yes" : "",
      pool_version: e.pool_version,
      last_seen_at: e.last_seen_at || "",
    };
  });
}

async function toggleCode(ctx, evaluator) {
  const verb = evaluator.disabled ? "Enable" : "Disable";
  if (!confirm(`${verb} ${evaluator.code}?`)) return;
  try {
    await ctx.store.updateEvaluator(evaluator.code, { disabled: !evaluator.disabled });
    await showAdmin(ctx);
  } catch (error) {
    toast(error.message, "error");
  }
}

function evaluatorsCard(ctx, data) {
  if (!data.evaluators.length) {
    return h("section", { class: "card" }, h("h2", {}, "Evaluators"), h("p", { class: "muted" }, "No codes yet."));
  }
  return h(
    "section",
    { class: "card" },
    h("h2", {}, `Evaluators (${data.evaluators.length})`),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "grid" },
        h(
          "thead",
          {},
          h("tr", {}, ["Code", "Note", "Domain", "Progress", "Tutorial", "Last seen", ""].map(t => h("th", {}, t))),
        ),
        h(
          "tbody",
          {},
          data.evaluators.map(e => {
            const progress = evaluatorProgress(e, data.byCode[e.code] || {});
            return h(
              "tr",
              { class: e.disabled ? "disabled" : null },
              h("td", { class: "mono" }, e.code),
              h("td", {}, e.label || ""),
              h("td", {}, ctx.site.domainLabel[e.domain] || e.domain),
              h("td", {}, `${progress.completed}/${progress.total} done${progress.started ? `, ${progress.started} started` : ""}`),
              h("td", {}, tutorialStatus(data.progress[e.code], ctx.tutorial)),
              h("td", {}, formatTime(e.last_seen_at)),
              h(
                "td",
                { class: "row-actions" },
                h("button", { class: "small", onclick: () => copyText(linkFor(e.code), "Link copied.") }, "Copy link"),
                h("button", { class: "small", onclick: () => toggleCode(ctx, e) }, e.disabled ? "Enable" : "Disable"),
              ),
            );
          }),
        ),
      ),
    ),
  );
}

function exportCard(ctx, data) {
  const json = () =>
    JSON.stringify(
      {
        exported_at: new Date().toISOString(),
        pool_version: ctx.pool.pool_version,
        rubric_version: ctx.rubric.version,
        tutorial_version: ctx.tutorial.version,
        evaluators: data.evaluators,
        progress: Object.values(data.progress),
        judgments: data.judgments.map(({ answers_json, ...rest }) => ({ ...rest, answers: parseAnswers(answers_json) })),
      },
      null,
      1,
    );
  const resetLocal = async () => {
    if (!confirm("Delete every local test code and answer stored in this browser?")) return;
    await ctx.store.reset();
    location.reload();
  };
  const codes = codeRows(ctx, data);
  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Export"),
    h("p", { class: "muted" }, `${data.judgments.length} judgment records, one per evaluator and trajectory.`),
    h(
      "div",
      { class: "actions start" },
      h("button", { onclick: () => download(`eval-answers-${fileStamp()}.json`, json()) }, "Answers (JSON)"),
      h(
        "button",
        { onclick: () => download(`eval-answers-${fileStamp()}.csv`, toCsv(data.judgments.flatMap(judgmentRows), CSV_COLUMNS), "text/csv") },
        "Answers (CSV, one row per answer)",
      ),
      codes.length
        ? h(
            "button",
            { onclick: () => download(`eval-codes-${fileStamp()}.csv`, toCsv(codes, Object.keys(codes[0])), "text/csv") },
            "Codes and progress (CSV)",
          )
        : null,
      ctx.store.mode === "local" ? h("button", { class: "danger", onclick: resetLocal }, "Reset local test data") : null,
    ),
  );
}
