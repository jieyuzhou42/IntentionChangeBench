import assert from "node:assert/strict";
import test from "node:test";

import { RUBRIC } from "../public/config/rubric.js";
import { TUTORIAL } from "../public/config/tutorial.js";
import {
  CODE_ALPHABET,
  CSV_COLUMNS,
  assignItems,
  countSummary,
  coverageCounts,
  judgmentRows,
  makeCode,
  missingAnswers,
  normalizeCode,
  planAssignments,
  seededRandom,
  stagesForTurn,
  toCsv,
  turnFieldLists,
  visibleAnswers,
} from "../public/js/core.js";

const pool = n => Array.from({ length: n }, (_, i) => `item-${String(i).padStart(3, "0")}`);

test("60 trajectories, 10 per person, 18 people: every trajectory gets exactly 3 reviewers", () => {
  const items = pool(60);
  const plans = planAssignments(items, [], 18, 10, seededRandom(1));
  for (const plan of plans) {
    assert.equal(plan.length, 10);
    assert.equal(new Set(plan).size, 10);
  }
  assert.deepEqual(countSummary(coverageCounts(items, plans)), { min: 3, max: 3, histogram: { 3: 60 } });
});

test("uneven pools stay within one reviewer of each other", () => {
  const items = pool(12);
  const { min, max } = countSummary(coverageCounts(items, planAssignments(items, [], 7, 10, seededRandom(7))));
  assert.ok(max - min <= 1, `min ${min}, max ${max}`);
});

test("new codes fill the least-reviewed trajectories first", () => {
  const items = pool(20);
  const next = assignItems(items, [items.slice(0, 10)], 10, seededRandom(3));
  assert.deepEqual([...next].sort(), items.slice(10));
});

test("asking for more trajectories than the pool has is an error", () => {
  assert.throws(() => assignItems(pool(5), [], 10), /5 trajectories/);
  assert.throws(() => assignItems(pool(5), [], 0), /positive whole number/);
});

test("codes use the prefix and the unambiguous alphabet", () => {
  const random = seededRandom(9);
  for (let i = 0; i < 50; i += 1) {
    const code = makeCode("TP", random);
    assert.match(code, /^TP-[0-9A-Z]{4}-[0-9A-Z]{4}$/);
    for (const ch of code.slice(3).replace("-", "")) assert.ok(CODE_ALPHABET.includes(ch), ch);
  }
  assert.equal(normalizeCode("  tp-ab12-cd34 "), "TP-AB12-CD34");
});

const turn = {
  gold: {
    state: {
      constraints: { budget: 2100, schedule: "No sightseeing on day 1" },
      priority: { high: ["budget"], medium: [], low: ["schedule"] },
      entities: {},
    },
    delta: { budget: { op: "override", old: 2000, new: 2100 } },
    action: { kind: "travel_plan", itinerary: [] },
  },
};

test("turns without a reference action skip the action stage", () => {
  assert.deepEqual(stagesForTurn(RUBRIC, "travelplanner", turn).map(s => s.id), ["reading", "change", "plan"]);
  const noAction = { gold: { ...turn.gold, action: null } };
  assert.deepEqual(stagesForTurn(RUBRIC, "webshop", noAction).map(s => s.id), ["reading", "change"]);
});

test("every labeled change needs a verdict, and problems need a comment", () => {
  const stage = RUBRIC.travelplanner.stages.find(s => s.id === "change");
  const answers = { change_verdicts: { budget: { verdict: "wrong_value" } }, missing_change: "no", state_problem: "no" };
  assert.deepEqual(missingAnswers(stage, answers, turnFieldLists(turn)), ["Is each labeled change correct?"]);
  answers.change_verdicts.budget.comment = "The user said $2,200.";
  assert.deepEqual(missingAnswers(stage, answers, turnFieldLists(turn)), []);
});

test("hidden follow-up questions are neither required nor kept", () => {
  const stage = RUBRIC.travelplanner.stages.find(s => s.id === "plan");
  const fields = turnFieldLists(turn);
  const answers = { meets_must: "yes", violated: ["budget"], reasonable: "yes" };
  assert.deepEqual(missingAnswers(stage, answers, fields), []);
  assert.equal("violated" in visibleAnswers(stage, answers), false);
  assert.deepEqual(missingAnswers(stage, { meets_must: "no", reasonable: "yes" }, fields), [
    "Which Must-have requirements does it violate?",
  ]);
});

test("rubric and tutorial configs are well formed", () => {
  for (const [domain, spec] of Object.entries(RUBRIC)) {
    if (domain === "version") continue;
    const stageIds = spec.stages.map(s => s.id);
    assert.equal(new Set(stageIds).size, stageIds.length, `${domain} stage ids`);
    for (const stage of spec.stages) {
      const ids = stage.questions.map(q => q.id);
      assert.equal(new Set(ids).size, ids.length, `${domain}/${stage.id} question ids`);
      for (const q of stage.questions) {
        assert.ok(["scale", "choice", "multi", "text", "per_field"].includes(q.type), `${q.id} type`);
        if (q.showIf) assert.ok(ids.includes(q.showIf.question), `${q.id} showIf target`);
        if (["scale", "choice", "per_field"].includes(q.type)) assert.ok(q.options?.length, `${q.id} options`);
        if (["multi", "per_field"].includes(q.type)) assert.ok(["changed", "must", "active"].includes(q.fieldsFrom), `${q.id} fieldsFrom`);
      }
    }
  }
  const quiz = TUTORIAL.steps.find(s => s.kind === "quiz");
  for (const domain of ["travelplanner", "webshop"]) {
    const questions = quiz.questions.filter(q => !q.domains || q.domains.includes(domain));
    assert.ok(questions.length >= 3, `${domain} quiz has ${questions.length} questions`);
  }
  for (const entry of [...TUTORIAL.steps.filter(s => s.kind === "practice"), ...quiz.questions]) {
    assert.ok(Number.isInteger(entry.answer) && entry.answer >= 0 && entry.answer < entry.options.length, `${entry.id} answer`);
  }
});

test("CSV export writes one row per answer and per field, with quoting", () => {
  const judgment = {
    code: "TP-AAAA-BBBB",
    domain: "travelplanner",
    item_id: "tp-x",
    pool_version: "pool-1",
    rubric_version: "rubric-1",
    answers_json: JSON.stringify({
      turns: {
        1: {
          stages: {
            reading: { answers: { naturalness: 4, own_reading: 'Drop "San Diego", keep the budget' }, submitted_at: "t1", active_ms: 5 },
            change: {
              answers: { change_verdicts: { budget: { verdict: "correct" }, days: { verdict: "wrong_value", comment: "x" } } },
              submitted_at: "t2",
            },
          },
        },
      },
    }),
  };
  const rows = judgmentRows(judgment);
  assert.equal(rows.length, 4);
  const days = rows.find(r => r.field === "days");
  assert.equal(days.value, "wrong_value");
  assert.equal(days.comment, "x");
  const csv = toCsv(rows, CSV_COLUMNS);
  assert.match(csv, /"Drop ""San Diego"", keep the budget"/);
  assert.equal(csv.trim().split("\n").length, 5);
});
