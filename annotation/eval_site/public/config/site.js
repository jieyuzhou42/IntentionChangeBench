// Study settings. Anything marked TEAM needs a decision or approved text before launch.

export const SITE = {
  title: "IntentionChangeBench label review",

  // "auto": Firebase when served by Firebase Hosting (or when firebaseConfig is set), local
  // test mode on localhost otherwise. "firebase" never falls back to local storage.
  backend: "auto",
  // Only needed when hosting somewhere other than Firebase Hosting.
  firebaseConfig: null,
  // The admin key is not set here, because this file is public. See "Admin key" in the README.

  // Defaults for the admin form that creates evaluator codes.
  defaultItemsPerEvaluator: 10,
  targetEvaluatorsPerItem: 3,
  codePrefix: { travelplanner: "TP", webshop: "WS" },
  domainLabel: { travelplanner: "TravelPlanner", webshop: "WebShop" },

  consent: {
    title: "Before you start",
    // TEAM: replace with the IRB-approved consent text.
    paragraphs: [
      "You will review conversations from a research dataset about how people change their requests while planning a trip or shopping online. For each conversation, you will answer questions about labels our team wrote.",
      "The task takes about [TEAM: time estimate] and pays [TEAM: amount]. You can stop at any time; your answers are saved after every step, and you can come back later with the same link.",
      "We record your answers, the time spent on each step, and your evaluator code. We do not ask for your name or email. [TEAM: data retention, contact, and ethics-board details.]",
    ],
    checkbox: "I have read the information above and agree to take part.",
  },

  completion: {
    message: "Thank you. All of your answers are saved, and there is nothing else to do.",
    // TEAM: completion code to show (for example, for Prolific), or null.
    completionCode: null,
  },

  // Shown when someone does not pass the qualification quiz.
  notQualifiedMessage:
    "Thank you for your time. Based on the quiz, this task is not a good fit, so we will not ask you to continue. [TEAM: what happens next, for example how payment for the tutorial works.]",

  contact: "[TEAM: contact email]",
};
