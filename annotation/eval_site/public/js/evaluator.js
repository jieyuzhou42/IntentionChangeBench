// Evaluator pages: consent, tutorial, task list, and the turn-by-turn review.

import {
  evaluatorProgress,
  judgmentStatus,
  missingAnswers,
  parseAnswers,
  questionVisible,
  stagesForTurn,
  tierOf,
  turnComplete,
  turnFieldLists,
  visibleAnswers,
} from "./core.js";
import { actionView, contextHeader, deltaList, describeChange, evidenceView, fieldLabel, stateTable, tierChip } from "./render.js";
import { details, fetchJson, h, mount, richText, toast } from "./ui.js";

const PROGRESS_KEYS = ["consent_at", "tutorial_passed_at", "tutorial_attempts", "tutorial_version"];
const nowIso = () => new Date().toISOString();

function applies(entry, domain) {
  return !entry.domains || entry.domains.includes(domain);
}

export function tutorialSteps(tutorial, domain) {
  return tutorial.steps
    .filter(step => applies(step, domain))
    .map(step => (step.kind === "quiz" ? { ...step, questions: step.questions.filter(q => applies(q, domain)) } : step));
}

/** Tutorial pages double as the guide that stays available during the task. */
export function guidePages(tutorial, domain) {
  return tutorialSteps(tutorial, domain).filter(step => step.kind === "page");
}

export async function startEvaluator(ctx) {
  ctx.progress = (await ctx.store.getProgress(ctx.evaluator.code)) || {};
  window.addEventListener("hashchange", () => route(ctx));
  await route(ctx);
}

async function route(ctx) {
  stopTimer();
  window.scrollTo(0, 0);
  try {
    if (!ctx.progress.consent_at) return showConsent(ctx);
    if (!ctx.progress.tutorial_passed_at) return showTutorial(ctx);
    const match = /^#\/item\/(.+)$/.exec(location.hash);
    if (match) return await showItem(ctx, decodeURIComponent(match[1]));
    return await showTasks(ctx);
  } catch (error) {
    console.error(error);
    mount(ctx.root, h("section", { class: "card narrow" }, h("h2", {}, "Something went wrong"), h("p", {}, error.message), h("a", { class: "button", href: "#/" }, "Back to the list")));
  }
}

async function saveProgress(ctx, patch) {
  const next = { ...ctx.progress, ...patch };
  const record = Object.fromEntries(PROGRESS_KEYS.filter(key => next[key] !== undefined).map(key => [key, next[key]]));
  await ctx.store.saveProgress(ctx.evaluator.code, record);
  ctx.progress = next;
}

function goHome(ctx) {
  history.replaceState(null, "", `${location.pathname}${location.search}#/`);
  return route(ctx);
}

// ---- Consent ---------------------------------------------------------------------------------

function showConsent(ctx) {
  const { consent } = ctx.site;
  const box = h("input", { type: "checkbox", id: "consent-box" });
  const button = h("button", { class: "primary", disabled: true }, "Continue");
  box.addEventListener("change", () => (button.disabled = !box.checked));
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await saveProgress(ctx, { consent_at: nowIso() });
      await goHome(ctx);
    } catch (error) {
      toast(`Could not save: ${error.message}`, "error");
      button.disabled = false;
    }
  });
  mount(
    ctx.root,
    h(
      "section",
      { class: "card narrow" },
      h("h2", {}, consent.title),
      richText(consent.paragraphs),
      h("label", { class: "check", for: "consent-box" }, box, consent.checkbox),
      h("div", { class: "actions" }, button),
    ),
  );
}

// ---- Tutorial --------------------------------------------------------------------------------

function showNotQualified(ctx) {
  mount(ctx.root, h("section", { class: "card narrow" }, h("h2", {}, "Thank you"), h("p", {}, ctx.site.notQualifiedMessage)));
}

function optionList(name, options, onPick) {
  return h(
    "div",
    { class: "options" },
    options.map((label, index) =>
      h("label", { class: "option" }, h("input", { type: "radio", name, value: index, onchange: () => onPick(index) }), h("span", {}, label)),
    ),
  );
}

function contextBlock(lines) {
  return lines?.length ? h("blockquote", { class: "example" }, lines.map(line => h("p", {}, line))) : null;
}

function showTutorial(ctx) {
  const { tutorial } = ctx;
  const steps = tutorialSteps(tutorial, ctx.evaluator.domain);
  const attempts = ctx.progress.tutorial_attempts || [];
  if (attempts.length >= tutorial.maxAttempts) return showNotQualified(ctx);
  const quizIndex = steps.findIndex(step => step.kind === "quiz");
  // After a failed attempt, return straight to the quiz; the pages stay one click away.
  let index = attempts.length && quizIndex >= 0 ? quizIndex : 0;

  const render = () => {
    const step = steps[index];
    const back = index > 0 ? h("button", { onclick: () => ((index -= 1), render()) }, "Back") : null;
    const next = h("button", { class: "primary", onclick: () => ((index += 1), render()) }, "Next");
    let content;
    if (step.kind === "page") {
      content = [h("h2", {}, step.title), richText(step.body), h("div", { class: "actions" }, back, next)];
    } else if (step.kind === "practice") {
      next.disabled = true;
      let picked = null;
      const feedback = h("div", { class: "feedback", hidden: true });
      const check = h("button", { disabled: true }, "Check answer");
      check.addEventListener("click", () => {
        const right = picked === step.answer;
        feedback.className = `feedback ${right ? "right" : "wrong"}`;
        mount(feedback, h("strong", {}, right ? "Correct. " : `Not quite. The answer is "${step.options[step.answer]}". `), step.explanation);
        feedback.hidden = false;
        next.disabled = false;
      });
      content = [
        h("h2", {}, "Practice"),
        contextBlock(step.context),
        h("p", { class: "prompt" }, step.prompt),
        optionList(step.id, step.options, choice => ((picked = choice), (check.disabled = false))),
        feedback,
        h("div", { class: "actions" }, back, check, next),
      ];
    } else {
      content = quizView(ctx, step, attempts, back);
    }
    mount(
      ctx.root,
      h(
        "section",
        { class: "card narrow tutorial" },
        h("p", { class: "muted" }, `Tutorial · step ${index + 1} of ${steps.length}`),
        h("progress", { max: steps.length, value: index + 1 }),
        content,
      ),
    );
    window.scrollTo(0, 0);
  };
  render();
}

function quizView(ctx, step, attempts, back) {
  const { tutorial } = ctx;
  const questions = step.questions;
  // The epsilon keeps float error from raising the bar (0.7 * 10 is 7.000000000000001).
  const needed = Math.ceil(tutorial.passThreshold * questions.length - 1e-9);
  const chosen = {};
  const result = h("div", { class: "feedback", hidden: true });
  const submit = h("button", { class: "primary" }, "Submit answers");
  submit.addEventListener("click", async () => {
    if (questions.some(q => chosen[q.id] === undefined)) {
      toast("Please answer every question.", "error");
      return;
    }
    submit.disabled = true;
    const score = questions.filter(q => chosen[q.id] === q.answer).length;
    const passed = score >= needed;
    const at = nowIso();
    const allAttempts = [...attempts, { at, score, total: questions.length, passed }];
    try {
      await saveProgress(ctx, {
        tutorial_attempts: allAttempts,
        tutorial_version: tutorial.version,
        ...(passed ? { tutorial_passed_at: at } : {}),
      });
    } catch (error) {
      toast(`Could not save: ${error.message}`, "error");
      submit.disabled = false;
      return;
    }
    if (passed) {
      toast(`Quiz passed (${score} of ${questions.length}).`);
      await goHome(ctx);
    } else if (allAttempts.length >= tutorial.maxAttempts) {
      showNotQualified(ctx);
    } else {
      const wrong = questions.map((q, i) => (chosen[q.id] === q.answer ? null : i + 1)).filter(Boolean);
      result.className = "feedback wrong";
      mount(
        result,
        h("p", {}, `You got ${score} of ${questions.length}; ${needed} are needed. Questions ${wrong.join(", ")} were not right.`),
        h("p", {}, "Review the guide pages (use Back, or the Guide button), then try again."),
        h("button", { class: "primary", onclick: () => showTutorial(ctx) }, "Try again"),
      );
      result.hidden = false;
    }
  });
  return [
    h("h2", {}, "Qualification quiz"),
    h(
      "p",
      {},
      `Answer all ${questions.length} questions; ${needed} correct answers unlock the task. Attempt ${attempts.length + 1} of ${tutorial.maxAttempts}.`,
    ),
    h(
      "ol",
      { class: "quiz" },
      questions.map(question =>
        h(
          "li",
          {},
          contextBlock(question.context),
          h("p", { class: "prompt" }, question.prompt),
          optionList(`quiz-${question.id}`, question.options, choice => (chosen[question.id] = choice)),
        ),
      ),
    ),
    result,
    h("div", { class: "actions" }, back, submit),
  ];
}

// ---- Task list -------------------------------------------------------------------------------

async function showTasks(ctx) {
  const { evaluator, store } = ctx;
  mount(ctx.root, h("p", { class: "muted" }, "Loading your conversations…"));
  const saved = await Promise.all(evaluator.item_ids.map(id => store.getJudgment(evaluator.code, id)));
  const byItem = Object.fromEntries(evaluator.item_ids.map((id, i) => [id, saved[i]]));
  const progress = evaluatorProgress(evaluator, byItem);
  const { completion } = ctx.site;
  const statusText = judgment =>
    ({
      not_started: "Not started",
      in_progress: `In progress: ${judgment?.turns_done ?? 0} of ${judgment?.turns_total ?? "?"} turns`,
      completed: "Done",
    })[judgmentStatus(judgment)];

  mount(
    ctx.root,
    progress.completed === progress.total && progress.total > 0
      ? h(
          "section",
          { class: "card done" },
          h("h2", {}, "All done"),
          h("p", {}, completion.message),
          completion.completionCode ? h("p", {}, "Completion code: ", h("code", {}, completion.completionCode)) : null,
        )
      : null,
    h(
      "section",
      { class: "card" },
      h("h2", {}, "Your conversations"),
      h("p", { class: "muted" }, `${progress.completed} of ${progress.total} complete. Your answers save after every step, so you can stop and come back later.`),
      h("progress", { max: progress.total || 1, value: progress.completed }),
      h(
        "ol",
        { class: "tasks" },
        evaluator.item_ids.map((id, index) => {
          const meta = ctx.poolItems.get(id);
          if (!meta) return h("li", { class: "task" }, `Conversation ${index + 1} is not in the current data. Please contact the research team.`);
          const status = judgmentStatus(byItem[id]);
          return h(
            "li",
            { class: `task ${status}` },
            h(
              "div",
              { class: "task-text" },
              h("strong", {}, `Conversation ${index + 1}`),
              h("span", { class: "muted" }, ` · ${meta.eval_turns.length} turns to review`),
              h("p", { class: "task-title" }, meta.title),
            ),
            h("span", { class: `status ${status}` }, statusText(byItem[id])),
            h(
              "a",
              { class: status === "completed" ? "button" : "button primary", href: `#/item/${encodeURIComponent(id)}` },
              { not_started: "Start", in_progress: "Continue", completed: "View" }[status],
            ),
          );
        }),
      ),
    ),
  );
}

// ---- Turn-by-turn review ---------------------------------------------------------------------

let activeTimer = null;

/** Milliseconds the page was visible, so time away from the tab is not counted. */
function startTimer() {
  stopTimer();
  let total = 0;
  let since = document.hidden ? null : performance.now();
  const onVisibility = () => {
    if (document.hidden && since !== null) {
      total += performance.now() - since;
      since = null;
    } else if (!document.hidden && since === null) {
      since = performance.now();
    }
  };
  document.addEventListener("visibilitychange", onVisibility);
  activeTimer = {
    elapsed: () => Math.round(total + (since === null ? 0 : performance.now() - since)),
    stop: () => document.removeEventListener("visibilitychange", onVisibility),
  };
  return activeTimer;
}

function stopTimer() {
  activeTimer?.stop();
  activeTimer = null;
}

async function showItem(ctx, itemId) {
  const { evaluator } = ctx;
  if (!evaluator.item_ids.includes(itemId)) throw new Error("This conversation is not assigned to your code.");
  const meta = ctx.poolItems.get(itemId);
  if (!meta) throw new Error("This conversation is not in the current data. Please contact the research team.");
  mount(ctx.root, h("p", { class: "muted" }, "Loading the conversation…"));
  const [item, judgment] = await Promise.all([
    fetchJson(`data/items/${encodeURIComponent(itemId)}.json`),
    ctx.store.getJudgment(evaluator.code, itemId),
  ]);
  const session = {
    ctx,
    item,
    meta,
    judgment,
    answers: parseAnswers(judgment?.answers_json),
    number: evaluator.item_ids.indexOf(itemId) + 1,
  };
  const position = nextTurn(session);
  if (position === null) renderItemDone(session);
  else renderTurn(session, position);
}

function stagesAt(session, position) {
  return stagesForTurn(session.ctx.rubric, session.item.domain, session.item.turns[position]);
}

function nextTurn(session) {
  const ids = position => stagesAt(session, position).map(stage => stage.id);
  return session.meta.eval_turns.find(position => !turnComplete(session.answers.turns[position], ids(position))) ?? null;
}

function buildJudgment(session, answers, now) {
  const { ctx, item, meta, judgment } = session;
  const done = meta.eval_turns.filter(position =>
    turnComplete(answers.turns[position], stagesAt(session, position).map(stage => stage.id)),
  ).length;
  const complete = done === meta.eval_turns.length;
  const activeMs = Object.values(answers.turns)
    .flatMap(turn => Object.values(turn.stages || {}))
    .reduce((sum, stage) => sum + (stage.active_ms || 0), 0);
  return {
    domain: item.domain,
    pool_version: ctx.pool.pool_version,
    rubric_version: ctx.rubric.version,
    guide_version: ctx.tutorial.version,
    status: complete ? "completed" : "in_progress",
    turns_done: done,
    turns_total: meta.eval_turns.length,
    answers_json: JSON.stringify(answers),
    active_ms: activeMs,
    started_at: judgment?.started_at || now,
    completed_at: complete ? judgment?.completed_at || now : null,
  };
}

function conversationView(item, position) {
  const label = item.domain === "webshop" ? "Reference product after this turn" : "Reference plan after this turn";
  return h(
    "ol",
    { class: "conversation" },
    item.turns.slice(0, position + 1).map((turn, index) =>
      h(
        "li",
        { class: index === position ? "current" : null },
        h(
          "div",
          { class: "turn-label" },
          index === 0 ? "Turn 0 · initial request" : `Turn ${index}`,
          index === position ? h("span", { class: "badge" }, "Review this message") : null,
        ),
        h("blockquote", {}, turn.utterance || "(no message)"),
        index < position && turn.gold.action
          ? details(index === position - 1 ? `${label} (what the user is reacting to)` : label, actionView(turn.gold.action))
          : null,
      ),
    ),
  );
}

function stepper(stages, current) {
  return h(
    "ol",
    { class: "stepper" },
    stages.map((stage, index) =>
      h("li", { class: index < current ? "done" : index === current ? "current" : null }, `${index + 1}. ${stage.title}`),
    ),
  );
}

function answerText(question, value, state) {
  if (value === undefined || value === null || value === "") return "—";
  if (question.type === "multi") return value.map(path => fieldLabel(path, state)).join(", ");
  if (question.options) return question.options.find(option => option.value === value)?.label ?? String(value);
  return String(value);
}

function submittedSummary(stage, stageRecord, turn, open) {
  const state = turn.gold.state;
  const rows = stage.questions
    .filter(question => stageRecord.answers?.[question.id] !== undefined)
    .flatMap(question => {
      const value = stageRecord.answers[question.id];
      if (question.type !== "per_field") return [h("dt", {}, question.label), h("dd", {}, answerText(question, value, state))];
      return Object.entries(value || {}).flatMap(([path, entry]) => [
        h("dt", {}, fieldLabel(path, state)),
        h("dd", {}, [answerText(question, entry.verdict, state), entry.comment].filter(Boolean).join(": ")),
      ]);
    });
  return details(`Your answers: ${stage.title}`, h("dl", { class: "summary" }, rows), open);
}

function questionInput(question, draft, fieldLists, turn, onChange) {
  const state = turn.gold.state;
  const name = `q-${question.id}`;
  if (question.type === "scale" || question.type === "choice") {
    return h(
      "div",
      { class: question.type === "scale" ? "options scale" : "options" },
      question.options.map(option =>
        h(
          "label",
          { class: "option" },
          h("input", { type: "radio", name, value: option.value, onchange: () => ((draft[question.id] = option.value), onChange()) }),
          h("span", {}, option.label),
        ),
      ),
    );
  }
  if (question.type === "multi") {
    const fields = fieldLists[question.fieldsFrom] || [];
    if (!fields.length) return h("p", { class: "muted" }, "There are no fields to choose from.");
    return h(
      "div",
      { class: "options" },
      fields.map(path =>
        h(
          "label",
          { class: "option" },
          h("input", {
            type: "checkbox",
            onchange: event => {
              const picked = new Set(draft[question.id] || []);
              if (event.target.checked) picked.add(path);
              else picked.delete(path);
              draft[question.id] = [...picked];
              onChange();
            },
          }),
          h("span", {}, fieldLabel(path, state)),
        ),
      ),
    );
  }
  if (question.type === "per_field") {
    const fields = fieldLists[question.fieldsFrom] || [];
    if (!fields.length) return h("p", { class: "muted" }, question.emptyText || "Nothing to check.");
    const verdicts = (draft[question.id] = {});
    const commentFree = question.commentUnless || [];
    return h(
      "div",
      { class: "per-field" },
      fields.map(path => {
        const entry = (verdicts[path] = {});
        const change = turn.gold.delta?.[path];
        const label = fieldLabel(path, state);
        const comment = h("input", {
          type: "text",
          hidden: true,
          placeholder: "What is wrong?",
          "aria-label": `What is wrong with ${label}?`,
          oninput: event => ((entry.comment = event.target.value), onChange()),
        });
        const select = h(
          "select",
          {
            "aria-label": `Verdict for ${label}`,
            onchange: event => {
              entry.verdict = event.target.value || undefined;
              comment.hidden = !entry.verdict || commentFree.includes(entry.verdict);
              onChange();
            },
          },
          h("option", { value: "" }, "Choose…"),
          question.options.map(option => h("option", { value: option.value }, option.label)),
        );
        return h(
          "div",
          { class: "field-row" },
          h(
            "div",
            { class: "field-name" },
            h("strong", {}, label),
            change ? h("span", { class: "delta-change" }, describeChange(change)) : null,
            change?.op === "remove" ? null : tierChip(tierOf(path, state.priority)),
          ),
          select,
          comment,
        );
      }),
    );
  }
  return h("textarea", {
    rows: 3,
    placeholder: question.placeholder || "",
    oninput: event => ((draft[question.id] = event.target.value), onChange()),
  });
}

function questionForm(stage, draft, fieldLists, turn) {
  let blocks = [];
  const refresh = () => {
    for (const { question, block } of blocks) block.hidden = !questionVisible(question, draft);
  };
  blocks = stage.questions.map(question => ({
    question,
    block: h(
      "fieldset",
      { class: "question" },
      h("legend", {}, question.label, question.required ? null : h("span", { class: "muted" }, " (optional)")),
      questionInput(question, draft, fieldLists, turn, refresh),
    ),
  }));
  refresh();
  return {
    element: h("div", { class: "questions" }, blocks.map(entry => entry.block)),
    markMissing(labels) {
      for (const { question, block } of blocks) block.classList.toggle("missing", labels.includes(question.label));
      blocks.find(entry => labels.includes(entry.question.label))?.block.scrollIntoView({ behavior: "smooth", block: "center" });
    },
  };
}

function renderTurn(session, position) {
  const { ctx, item, meta } = session;
  const turn = item.turns[position];
  const stages = stagesAt(session, position);
  const record = session.answers.turns[position] || { stages: {} };
  const current = stages.findIndex(stage => !record.stages?.[stage.id]?.submitted_at);
  const stage = stages[current];
  const fieldLists = turnFieldLists(turn);
  const reveals = new Set(stage.reveals || []);
  const draft = {};
  const form = questionForm(stage, draft, fieldLists, turn);
  const timer = startTimer();
  const last = current === stages.length - 1;
  const submit = h("button", { class: "primary" }, reveals.size === 0 && !last ? "Save and show the labels" : last ? "Save this turn" : "Save and continue");

  submit.addEventListener("click", async () => {
    const answers = visibleAnswers(stage, draft);
    const missing = missingAnswers(stage, answers, fieldLists);
    form.markMissing(missing);
    if (missing.length) {
      toast("Please answer the highlighted questions.", "error");
      return;
    }
    submit.disabled = true;
    const now = nowIso();
    const stageIds = stages.map(entry => entry.id);
    const nextRecord = {
      ...record,
      stages: { ...record.stages, [stage.id]: { answers, submitted_at: now, active_ms: timer.elapsed() } },
    };
    if (turnComplete(nextRecord, stageIds)) nextRecord.completed_at = now;
    const nextAnswers = { ...session.answers, turns: { ...session.answers.turns, [position]: nextRecord } };
    const judgment = buildJudgment(session, nextAnswers, now);
    try {
      await ctx.store.saveJudgment(ctx.evaluator.code, item.item_id, judgment);
    } catch (error) {
      console.error(error);
      toast(`Could not save (${error.message}). Your answers are still on the page; please try again.`, "error");
      submit.disabled = false;
      return;
    }
    session.answers = nextAnswers;
    session.judgment = judgment;
    if (nextRecord.completed_at) toast("Turn saved.");
    const next = nextTurn(session);
    if (next === null) renderItemDone(session);
    else renderTurn(session, next);
  });

  const actionTitle = turn.gold.action?.kind === "product" ? "Reference product for this turn" : "Reference plan for this turn";
  const evidence = reveals.has("evidence") ? evidenceView(item.evidence) : null;
  const review = h(
    "section",
    { class: "pane review-pane" },
    stepper(stages, current),
    stages.slice(0, current).map((done, index) => submittedSummary(done, record.stages[done.id], turn, index === 0 && current === 1)),
    h(
      "div",
      { class: "card stage" },
      h("h2", {}, stage.title),
      h("p", { class: "muted" }, stage.intro),
      reveals.has("delta") ? h("div", { class: "reveal" }, h("h3", {}, "Labeled change in this turn"), deltaList(turn.gold.delta, turn.gold.state)) : null,
      reveals.has("state")
        ? h("div", { class: "reveal" }, details("All active requirements after this turn", stateTable(turn.gold.state, { highlight: fieldLists.changed }), true))
        : null,
      reveals.has("action") ? h("div", { class: "reveal" }, h("h3", {}, actionTitle), actionView(turn.gold.action)) : null,
      evidence ? h("div", { class: "reveal" }, details("Candidate records (for checking prices, dates, and capacity)", evidence)) : null,
      form.element,
      h("div", { class: "actions" }, submit),
    ),
  );
  const context = h(
    "section",
    { class: "pane context-pane" },
    contextHeader(item, ctx.site.domainLabel[item.domain] || item.domain),
    conversationView(item, position),
    position > 0 ? details("Requirements before this turn", stateTable(item.turns[position - 1].gold.state)) : null,
  );

  mount(
    ctx.root,
    h(
      "div",
      { class: "item-bar" },
      h("a", { class: "button", href: "#/" }, "← All conversations"),
      h(
        "span",
        { class: "muted" },
        `Conversation ${session.number} of ${ctx.evaluator.item_ids.length} · reviewing turn ${meta.eval_turns.indexOf(position) + 1} of ${meta.eval_turns.length}`,
      ),
    ),
    h("div", { class: "item-layout" }, context, review),
  );
  window.scrollTo(0, 0);
}

function renderItemDone(session) {
  stopTimer();
  mount(
    session.ctx.root,
    h(
      "section",
      { class: "card narrow done" },
      h("h2", {}, `Conversation ${session.number} is complete`),
      h("p", {}, `You reviewed ${session.meta.eval_turns.length} turns. Thank you.`),
      h("a", { class: "button primary", href: "#/" }, "Back to your conversations"),
    ),
  );
  window.scrollTo(0, 0);
}
