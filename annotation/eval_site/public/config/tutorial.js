// Tutorial and evaluator guide.
//
// DRAFT: the rules and examples below are condensed from
// annotation/TRAVELPLANNER_ANNOTATION_PLAYBOOK.md and annotation/ANNOTATION_GUIDE.md and need a
// team review before launch. Bump `version` whenever content changes; it is saved with every
// judgment.
//
// Step kinds:
//   page      title + body; pages also make up the Guide panel available during the task
//   practice  one question with immediate feedback; not scored
//   quiz      scored questions; passing unlocks the real task
// `domains` limits a step or quiz question to some domains (default: all).
// Body strings that start with "- " render as bullet points.

const TP = ["travelplanner"];
const WS = ["webshop"];

export const TUTORIAL = {
  version: "tutorial-v0.1-draft",
  passThreshold: 0.8,
  maxAttempts: 2,
  steps: [
    {
      kind: "page",
      id: "task",
      title: "What you will do",
      body: [
        "Each conversation shows a user making a request and then changing it over several turns. Our team labeled what the user wants after every turn. You will check those labels.",
        "For each turn you review there are up to three steps:",
        "- Read the new message and answer a few questions before seeing our labels.",
        "- Check the change our labels record, and the full list of requirements after it.",
        "- Check the reference plan or product against those requirements.",
        "We are testing our labels, not you. If the information shown is not enough to decide, answer \"Can't tell\" instead of guessing.",
        "You can reopen these pages at any time with the Guide button.",
      ],
    },
    {
      kind: "page",
      id: "tiers",
      title: "Requirements and priority tiers",
      body: [
        "Every active requirement has exactly one priority tier:",
        "- Must-have: a requirement the user is currently protecting.",
        "- Preferred: still active, but the user would give it up before a Must-have.",
        "- Optional: pursued only if circumstances allow.",
        "The tier and the value are separate. Lowering a rating threshold from 4.0 to 3.7 changes the value. It does not by itself make the rating Preferred, and it never allows 3.6.",
      ],
    },
    {
      kind: "page",
      id: "cumulative",
      title: "Requirements carry forward",
      body: [
        "The labels after each turn list every requirement that is still active, not just the new ones. A requirement stays in force, with the same tier, until the user changes or withdraws it.",
        "A new requirement applies from the turn where it was stated, never to earlier turns.",
        "Mentioning something again, or recently, is not by itself a reason to raise its tier.",
      ],
    },
    {
      kind: "page",
      id: "tp-implicit",
      domains: TP,
      title: "Implicit is fine; unsupported is not",
      body: [
        "Users often hint instead of spelling things out. A label may follow from a hint, but it must not add precision the conversation does not give.",
        "\"Let's take it slower\" can support fewer activities. Without an earlier number to anchor it, it does not support a label like \"at most two attractions per day\".",
      ],
    },
    {
      kind: "page",
      id: "tp-conditional",
      domains: TP,
      title: "A withdrawn condition undoes only its own change",
      body: [
        "When an earlier change depended on a condition, withdrawing the condition undoes that change and nothing else.",
        "- Turn 1: \"For the driving plan, I can spend up to $1,500.\"",
        "- Turn 2: \"If we fly instead, I can stretch that to $2,000.\"",
        "- Turn 3: \"Let's go back to driving.\" The budget returns to $1,500.",
        "If turn 2 had simply said \"I have $2,000 now\", going back to driving would not restore $1,500.",
        "Independent changes made in between also stay. With a $3,300 base, a $1,500 room upgrade, and a separate $100 flight allowance, cancelling the upgrade leaves $3,400, not $3,300.",
      ],
    },
    {
      kind: "page",
      id: "tp-scope",
      domains: TP,
      title: "Scope: who, when, and what exactly",
      body: [
        "\"One attraction that day\" applies to that day only, not to the whole trip.",
        "\"Don't schedule A\", \"A is no longer required\", and \"cancel that visit to A\" mean different things: a ban, the removal of a requirement, and the cancellation of one arrangement. Only the first forbids A.",
        "When a travel companion drops out, requirements that belonged only to them go away. The user's own requirements stay.",
      ],
    },
    {
      kind: "page",
      id: "tp-basics",
      domains: TP,
      title: "Trip basics",
      body: [
        "From turn 1 on, trip basics that never changed (dates, party size, origin, destination, trip length, number of cities) are labeled Optional so they do not dominate the scores.",
        "That is a scoring convention, not permission to ignore them: a plan for the wrong dates or the wrong number of travelers is still invalid. Once one of these basics changes, it is labeled Must-have from then on.",
      ],
    },
    {
      kind: "page",
      id: "tp-plan",
      domains: TP,
      title: "Checking a plan",
      body: [
        "Compare each day of the plan with the active requirements and the candidate records: prices, dates, flight times, minimum nights, room capacity, and whether a cost is per person or per night.",
        "Entries marked UNRESOLVED, and costs listed as not included in the subtotal, are things the reference could not confirm. If a Must-have depends on them, answer \"Can't verify\".",
        "A cuisine tag does not prove a meal is allergy-safe, and a room type does not prove accessibility.",
      ],
    },
    {
      kind: "page",
      id: "ws-priority",
      domains: WS,
      title: "Priority tiers in shopping requests",
      body: [
        "Several requirements can be Must-have at once. Users often name two or three in one sentence.",
        "A relaxed requirement that still filters products (\"dark wood or neutral\") is not Optional. One that no longer filters anything (\"any color\") is Optional.",
        "\"X can change, as long as Y\" makes Y a Must-have. X, the relaxed one, does not move up.",
        "Raising the budget to reach a feature makes the feature the Must-have, not the budget. Lowering the budget makes the budget a Must-have.",
      ],
    },
    {
      kind: "page",
      id: "ws-product",
      domains: WS,
      title: "Checking a product",
      body: [
        "Every Must-have should be supported by the product listing. A requirement the listing does not mention is not established: answer \"No\" or \"Can't verify\", not \"Yes\".",
        "The price should be within the active budget.",
      ],
    },
    {
      kind: "practice",
      id: "tp-practice-condition",
      domains: TP,
      context: [
        "Turn 1: \"For the driving plan, I can spend up to $1,800.\"",
        "Turn 2: \"If we take the train instead, I could go to $2,200.\"",
        "Turn 3: \"Actually, let's drive after all.\"",
      ],
      prompt: "What budget is active after turn 3?",
      options: ["$1,800", "$2,200", "Can't tell"],
      answer: 0,
      explanation: "The $2,200 depended on taking the train. Withdrawing that condition restores $1,800.",
    },
    {
      kind: "practice",
      id: "tp-practice-independent",
      domains: TP,
      context: [
        "Start: budget $2,000.",
        "Turn 1: \"I want nicer dinners, so let's raise the budget to $2,300.\"",
        "Turn 2: \"My partner is joining, so we can go up to $4,000.\"",
        "Turn 3: \"My partner can't come after all.\"",
      ],
      prompt: "What budget is active after turn 3?",
      options: ["$2,000", "$2,300", "$4,000"],
      answer: 1,
      explanation:
        "Only the $4,000 depended on the partner. The dinner increase was independent, so the budget returns to $2,300, not $2,000.",
    },
    {
      kind: "practice",
      id: "tp-practice-scope",
      domains: TP,
      context: ["Turn 4: \"Just one attraction on the 19th, please.\""],
      prompt: "What does this message require?",
      options: ["At most one attraction on every day", "No attractions after March 19", "At most one attraction on March 19 only"],
      answer: 2,
      explanation: "The limit is scoped to one day. Other days keep whatever requirements they already had.",
    },
    {
      kind: "practice",
      id: "ws-practice-any",
      domains: WS,
      context: ["Earlier: the pillow cover must be black.", "Now: \"Any color is fine.\""],
      prompt: "What tier should the color requirement have now?",
      options: ["Optional, because it no longer filters products", "Must-have, because the user just mentioned it", "Preferred, because it was relaxed"],
      answer: 0,
      explanation: "\"Any color\" no longer narrows the choice, so color is Optional. Mentioning it does not raise its tier.",
    },
    {
      kind: "practice",
      id: "ws-practice-as-long-as",
      domains: WS,
      context: ["\"The cushion can be a different shape, as long as it is machine washable.\""],
      prompt: "Which requirement should be Must-have?",
      options: ["Shape", "Machine washable", "Neither"],
      answer: 1,
      explanation: "\"As long as\" pins down the condition the user insists on. The shape is the part being relaxed.",
    },
    {
      kind: "quiz",
      id: "quiz",
      questions: [
        {
          id: "tp-q-withdraw",
          domains: TP,
          context: ["Earlier: \"Make sure we visit the Getty.\"", "Now: \"The Getty is no longer a must.\""],
          prompt: "Which label is right?",
          options: [
            "The Getty must not appear in the plan",
            "The requirement to visit the Getty is removed; the plan may still include it",
            "Every attraction requirement is removed",
          ],
          answer: 1,
        },
        {
          id: "tp-q-revert",
          domains: TP,
          context: [
            "Base budget: $3,000.",
            "Turn 1: \"If we get the suite, I'll add $1,000.\"",
            "Turn 2: \"Also add $200 so we can take the earlier flight.\"",
            "Turn 3: \"Skip the suite.\"",
          ],
          prompt: "What budget is active after turn 3?",
          options: ["$3,000", "$4,200", "$3,200"],
          answer: 2,
        },
        {
          id: "tp-q-implicit",
          domains: TP,
          context: [
            "\"Let's take things slower this trip.\"",
            "Label: \"at most two attractions per day\" (Must-have). Nothing earlier mentions a number.",
          ],
          prompt: "Is this label supported?",
          options: [
            "No, the message supports fewer activities but not that specific limit",
            "Yes, slower clearly means two",
            "Yes, because Must-have requirements should be specific",
          ],
          answer: 0,
        },
        {
          id: "tp-q-floor",
          domains: TP,
          context: ["\"Hotels rated at least 4, ideally 5.\""],
          prompt: "What is the best labeling?",
          options: ["One Must-have: rating 5", "One Preferred: rating 4 or 5", "A Must-have floor of 4 and a Preferred target of 5"],
          answer: 2,
        },
        {
          id: "tp-q-basics",
          domains: TP,
          context: [
            "The party size (2 people) never changed, so from turn 1 on it is labeled Optional.",
            "The reference plan books a room that holds one person.",
          ],
          prompt: "Is the plan acceptable?",
          options: ["No, the plan still has to fit the actual party", "Yes, party size is Optional", "Can't tell"],
          answer: 0,
        },
        {
          id: "ws-q-budget-up",
          domains: WS,
          context: ["\"Raise my budget from $530 to $670 so I can look at real hardwood.\""],
          prompt: "Which should be Must-have?",
          options: ["Solid hardwood", "The budget", "Both, equally"],
          answer: 0,
        },
        {
          id: "ws-q-several",
          domains: WS,
          context: ["\"I care more about speed and space than about paying the least.\""],
          prompt: "Which labeling is right?",
          options: [
            "Only speed is Must-have, since each turn has one Must-have",
            "Price stays the top priority",
            "Speed and space rank above price, and both can be Must-have",
          ],
          answer: 2,
        },
        {
          id: "ws-q-budget-down",
          domains: WS,
          context: ["\"Lower my limit to $40.\""],
          prompt: "What should happen to the budget?",
          options: ["Optional, because it changed", "Must-have at $40", "Preferred at $40"],
          answer: 1,
        },
        {
          id: "ws-q-still-filters",
          domains: WS,
          context: ["Earlier: walnut finish (Must-have).", "Now: \"Dark wood or a neutral color is fine.\""],
          prompt: "What about the color requirement?",
          options: [
            "It becomes Optional because it was relaxed",
            "It still filters products, so it should not be Optional",
            "It should be deleted",
          ],
          answer: 1,
        },
        {
          id: "ws-q-evidence",
          domains: WS,
          context: ["Must-have: machine washable.", "The product listing says nothing about washing."],
          prompt: "Does the product satisfy every Must-have?",
          options: ["No, or can't verify: the listing does not establish it", "Yes, most pillow covers are washable"],
          answer: 0,
        },
      ],
    },
  ],
};
