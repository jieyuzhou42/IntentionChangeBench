// Pure logic shared by the pages and the Node tests: no DOM, no storage.

export const TIER_LABEL = { high: "Must-have", medium: "Preferred", low: "Optional" };
export const TIERS = ["high", "medium", "low"];

// No 0/O, 1/I/L: codes are read aloud and typed by hand.
export const CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";

export function cryptoRandom() {
  const buffer = new Uint32Array(1);
  globalThis.crypto.getRandomValues(buffer);
  return buffer[0] / 2 ** 32;
}

/** Deterministic generator (mulberry32) for tests and reproducible plans. */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeCode(prefix, random = cryptoRandom) {
  const group = () =>
    Array.from({ length: 4 }, () => CODE_ALPHABET[Math.floor(random() * CODE_ALPHABET.length)]).join("");
  return `${prefix}-${group()}-${group()}`;
}

export function normalizeCode(text) {
  return String(text || "").trim().toUpperCase().replace(/\s+/g, "");
}

export function shuffled(values, random = cryptoRandom) {
  const copy = [...values];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function coverageCounts(itemIds, assignments) {
  const counts = new Map(itemIds.map(id => [id, 0]));
  for (const assigned of assignments) {
    for (const id of assigned) if (counts.has(id)) counts.set(id, counts.get(id) + 1);
  }
  return counts;
}

/**
 * Pick `perEvaluator` trajectories for one new evaluator: least-reviewed first, ties broken at
 * random so evaluators get different combinations. The result is shuffled again so the order
 * each person sees them in is random too.
 */
export function assignItems(itemIds, assignments, perEvaluator, random = cryptoRandom) {
  if (!Number.isInteger(perEvaluator) || perEvaluator < 1) {
    throw new Error("Trajectories per evaluator must be a positive whole number.");
  }
  if (perEvaluator > itemIds.length) {
    throw new Error(`This pool has ${itemIds.length} trajectories, fewer than the ${perEvaluator} requested per person.`);
  }
  const counts = coverageCounts(itemIds, assignments);
  const picked = shuffled(itemIds, random)
    .sort((a, b) => counts.get(a) - counts.get(b))
    .slice(0, perEvaluator);
  return shuffled(picked, random);
}

/** Assignments for `count` new evaluators, each accounting for the ones before it. */
export function planAssignments(itemIds, assignments, count, perEvaluator, random = cryptoRandom) {
  const all = [...assignments];
  const planned = [];
  for (let i = 0; i < count; i += 1) {
    const next = assignItems(itemIds, all, perEvaluator, random);
    planned.push(next);
    all.push(next);
  }
  return planned;
}

export function countSummary(counts) {
  const values = [...counts.values()];
  const histogram = {};
  for (const value of values) histogram[value] = (histogram[value] || 0) + 1;
  return {
    min: values.length ? Math.min(...values) : 0,
    max: values.length ? Math.max(...values) : 0,
    histogram,
  };
}

export function parseAnswers(answersJson) {
  if (!answersJson) return { turns: {} };
  const parsed = typeof answersJson === "string" ? JSON.parse(answersJson) : answersJson;
  return parsed && typeof parsed === "object" ? { turns: {}, ...parsed } : { turns: {} };
}

// ---- Turn field lists ----------------------------------------------------------------------

export function activeFieldPaths(state) {
  const paths = Object.keys(state?.constraints || {});
  for (const [entityId, entity] of Object.entries(state?.entities || {})) {
    for (const key of Object.keys(entity?.constraints || {})) paths.push(`entities.${entityId}.constraints.${key}`);
  }
  return paths;
}

export function tierOf(path, priority) {
  return TIERS.find(tier => (priority?.[tier] || []).includes(path)) || null;
}

/** Field lists that rubric questions draw their options from. */
export function turnFieldLists(turn) {
  const state = turn?.gold?.state || {};
  return {
    changed: Object.keys(turn?.gold?.delta || {}),
    must: [...(state.priority?.high || [])],
    active: activeFieldPaths(state),
  };
}

// ---- Rubric --------------------------------------------------------------------------------

export function stagesForTurn(rubric, domain, turn) {
  const stages = rubric?.[domain]?.stages || [];
  return stages.filter(stage => stage.requires !== "action" || Boolean(turn?.gold?.action));
}

export function questionVisible(question, answers) {
  if (!question.showIf) return true;
  return answers?.[question.showIf.question] === question.showIf.equals;
}

/** Labels of required questions that are visible but unanswered or missing a required comment. */
export function missingAnswers(stage, answers, fieldLists) {
  const missing = [];
  for (const question of stage.questions) {
    if (!question.required || !questionVisible(question, answers)) continue;
    const value = answers?.[question.id];
    if (question.type === "per_field") {
      const fields = fieldLists[question.fieldsFrom] || [];
      const commentFree = question.commentUnless || [];
      const incomplete = fields.some(field => {
        const entry = value?.[field];
        if (!entry?.verdict) return true;
        return !commentFree.includes(entry.verdict) && !String(entry.comment || "").trim();
      });
      if (incomplete) missing.push(question.label);
    } else if (question.type === "multi") {
      // With nothing to choose from (e.g. no Must-have fields) the question cannot be answered.
      const fields = fieldLists[question.fieldsFrom] || [];
      if (fields.length && (!Array.isArray(value) || value.length === 0)) missing.push(question.label);
    } else if (value === undefined || value === null || String(value).trim() === "") {
      missing.push(question.label);
    }
  }
  return missing;
}

/** Keep only answers to questions that are visible, so hidden follow-ups do not linger. */
export function visibleAnswers(stage, answers) {
  const kept = {};
  for (const question of stage.questions) {
    if (questionVisible(question, answers) && answers?.[question.id] !== undefined) {
      kept[question.id] = answers[question.id];
    }
  }
  return kept;
}

// ---- Progress ------------------------------------------------------------------------------

export function turnComplete(turnRecord, stageIds) {
  return stageIds.every(id => Boolean(turnRecord?.stages?.[id]?.submitted_at));
}

export function judgmentStatus(judgment) {
  if (!judgment) return "not_started";
  return judgment.status === "completed" ? "completed" : "in_progress";
}

export function evaluatorProgress(evaluator, judgmentsByItem) {
  const total = (evaluator.item_ids || []).length;
  let completed = 0;
  let started = 0;
  for (const itemId of evaluator.item_ids || []) {
    const status = judgmentStatus(judgmentsByItem[itemId]);
    if (status === "completed") completed += 1;
    else if (status === "in_progress") started += 1;
  }
  return { completed, started, total };
}

// ---- Export --------------------------------------------------------------------------------

export const CSV_COLUMNS = [
  "code", "domain", "item_id", "turn", "stage", "question", "field",
  "value", "comment", "submitted_at", "active_ms", "pool_version", "rubric_version",
];

/** One row per answer; per-field questions give one row per field. */
export function judgmentRows(judgment) {
  const rows = [];
  const answers = parseAnswers(judgment.answers_json);
  for (const [turn, turnRecord] of Object.entries(answers.turns || {})) {
    for (const [stage, stageRecord] of Object.entries(turnRecord.stages || {})) {
      for (const [question, value] of Object.entries(stageRecord.answers || {})) {
        const base = {
          code: judgment.code,
          domain: judgment.domain,
          item_id: judgment.item_id,
          turn,
          stage,
          question,
          submitted_at: stageRecord.submitted_at || "",
          active_ms: stageRecord.active_ms ?? "",
          pool_version: judgment.pool_version || "",
          rubric_version: judgment.rubric_version || "",
        };
        if (value && typeof value === "object" && !Array.isArray(value)) {
          for (const [field, entry] of Object.entries(value)) {
            rows.push({ ...base, field, value: entry?.verdict ?? "", comment: entry?.comment ?? "" });
          }
        } else {
          rows.push({ ...base, field: "", value: Array.isArray(value) ? value.join(" | ") : value ?? "", comment: "" });
        }
      }
    }
  }
  return rows;
}

export function toCsv(rows, columns) {
  const cell = value => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [columns.join(","), ...rows.map(row => columns.map(column => cell(row[column])).join(","))].join("\n") + "\n";
}
