// Questions asked for every reviewed turn. Stages run in order; each stage reveals more of the
// reference labels, and answers lock once the next stage is shown. Bump `version` whenever a
// question changes: it is saved with every judgment.
//
// Question types:
//   scale      one value from `options`, shown in a row
//   choice     one value from `options`
//   multi      any number of fields from `fieldsFrom`
//   text       free text
//   per_field  one verdict from `options` per field in `fieldsFrom`, with a comment that is
//              required unless the verdict is listed in `commentUnless`
// `showIf: { question, equals }` shows a question only after that answer.
// `fieldsFrom`: "changed" (fields in this turn's labeled change), "must" (active Must-have
// fields), "active" (every active field).
// A stage with `requires: "action"` is skipped on turns without a reference plan or product.
// `reveals` controls which reference labels the stage shows: "delta", "state", "action", "evidence".

const NATURALNESS = {
  id: "naturalness",
  type: "scale",
  required: true,
  label: "How natural is this message as something a real person would write?",
  options: [
    { value: 1, label: "1 Clearly unnatural" },
    { value: 2, label: "2" },
    { value: 3, label: "3" },
    { value: 4, label: "4" },
    { value: 5, label: "5 Completely natural" },
  ],
};

const CLARITY = {
  id: "clarity",
  type: "choice",
  required: true,
  label: "Given the conversation so far, how many reasonable readings does the message have?",
  options: [
    { value: "one", label: "One reasonable reading" },
    { value: "minor", label: "One main reading, with minor ambiguity" },
    { value: "multiple", label: "Two or more reasonable readings" },
  ],
};

const readingStage = example => ({
  id: "reading",
  title: "Your reading of the message",
  intro:
    "Answer from the conversation alone. The reference labels for this turn stay hidden until you save, and these answers lock once they appear.",
  reveals: [],
  questions: [
    NATURALNESS,
    CLARITY,
    {
      id: "own_reading",
      type: "text",
      required: true,
      label: "In your own words, what did the user change or ask for?",
      placeholder: example,
    },
  ],
});

const CHANGE_STAGE = {
  id: "change",
  title: "Check the labeled change",
  intro:
    "These are the changes the reference labels record for this turn, followed by every requirement active after it.",
  reveals: ["delta", "state"],
  questions: [
    {
      id: "change_verdicts",
      type: "per_field",
      fieldsFrom: "changed",
      required: true,
      label: "Is each labeled change correct?",
      emptyText: "The reference labels record no change in this turn.",
      options: [
        { value: "correct", label: "Correct" },
        { value: "wrong_value", label: "Wrong value" },
        { value: "wrong_priority", label: "Wrong priority tier" },
        { value: "wrong_scope", label: "Wrong scope (who, when, or where it applies)" },
        { value: "unsupported", label: "Not supported by the conversation" },
        { value: "cant_tell", label: "Can't tell" },
      ],
      commentUnless: ["correct", "cant_tell"],
    },
    {
      id: "missing_change",
      type: "choice",
      required: true,
      label: "Did the user change anything that is missing from this list?",
      options: [
        { value: "no", label: "No" },
        { value: "yes", label: "Yes" },
      ],
    },
    {
      id: "missing_change_detail",
      type: "text",
      required: true,
      showIf: { question: "missing_change", equals: "yes" },
      label: "What is missing?",
    },
    {
      id: "state_problem",
      type: "choice",
      required: true,
      label:
        "Is anything wrong in the full list of active requirements? For example, something that should have been dropped, or something that was lost.",
      options: [
        { value: "no", label: "No" },
        { value: "yes", label: "Yes" },
        { value: "cant_tell", label: "Can't tell" },
      ],
    },
    {
      id: "state_problem_detail",
      type: "text",
      required: true,
      showIf: { question: "state_problem", equals: "yes" },
      label: "What is wrong?",
    },
  ],
};

const actionStage = ({ id, noun, intro }) => ({
  id,
  title: `Check the reference ${noun}`,
  intro,
  reveals: ["state", "action", "evidence"],
  requires: "action",
  questions: [
    {
      id: "meets_must",
      type: "choice",
      required: true,
      label: `Does the ${noun} satisfy every Must-have requirement?`,
      options: [
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" },
        { value: "cant_verify", label: "Can't verify from the information shown" },
      ],
    },
    {
      id: "violated",
      type: "multi",
      fieldsFrom: "must",
      required: true,
      showIf: { question: "meets_must", equals: "no" },
      label: "Which Must-have requirements does it violate?",
    },
    {
      id: "reasonable",
      type: "choice",
      required: true,
      label: `Taking the whole conversation into account, is this a reasonable ${noun} to give the user?`,
      options: [
        { value: "yes", label: "Yes" },
        { value: "minor_issues", label: "Acceptable, with minor issues" },
        { value: "no", label: "No" },
      ],
    },
    { id: "comment", type: "text", required: false, label: "Comments" },
  ],
});

export const RUBRIC = {
  version: "rubric-v0.1-draft",
  travelplanner: {
    stages: [
      readingStage("For example: drop San Diego, stay four nights in Los Angeles, keep the same budget."),
      CHANGE_STAGE,
      actionStage({
        id: "plan",
        noun: "plan",
        intro:
          "Check the plan against the active requirements. Use the candidate records for prices, dates, minimum nights, and capacity.",
      }),
    ],
  },
  webshop: {
    stages: [
      readingStage("For example: a smaller size is fine now; everything else stays the same."),
      CHANGE_STAGE,
      actionStage({
        id: "product",
        noun: "product",
        intro: "Check the product against the active requirements, using only what the product listing shows.",
      }),
    ],
  },
};
