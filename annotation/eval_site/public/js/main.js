// Entry point: picks the storage backend, loads the pool, and routes to the evaluator or admin pages.
//   ?code=XXXX  evaluator link (the code is also remembered in this browser)
//   ?admin      admin dashboard

import { RUBRIC } from "../config/rubric.js";
import { SITE } from "../config/site.js";
import { TUTORIAL } from "../config/tutorial.js";
import { showAdmin } from "./admin.js";
import { normalizeCode } from "./core.js";
import { guidePages, startEvaluator } from "./evaluator.js";
import { createStore } from "./store.js";
import { fetchJson, h, mount, richText } from "./ui.js";

const CODE_KEY = "evalsite:code";
const root = document.getElementById("app");

function failure(title, error) {
  console.error(error);
  mount(root, h("section", { class: "card narrow" }, h("h2", {}, title), h("p", {}, error.message)));
}

function showBanners(ctx) {
  const banners = document.getElementById("banners");
  if (ctx.store.mode === "local") {
    banners.append(h("div", { class: "banner local" }, "Local test mode: codes and answers are saved only in this browser."));
  }
  if (ctx.pool.demo) {
    banners.append(h("div", { class: "banner demo" }, `Demo pool (${ctx.pool.pool_version}): placeholder trajectories until the final data is published.`));
  }
}

function setHeader(ctx, who, { guide = false } = {}) {
  document.getElementById("who").textContent = who;
  const signOut = document.getElementById("signout-button");
  signOut.hidden = false;
  signOut.onclick = async () => {
    localStorage.removeItem(CODE_KEY);
    await ctx.store.signOut().catch(() => {});
    location.href = location.pathname;
  };
  const guideButton = document.getElementById("guide-button");
  guideButton.hidden = !guide;
  guideButton.onclick = () => openGuide(ctx);
}

function openGuide(ctx) {
  const drawer = document.getElementById("guide");
  const close = () => {
    drawer.hidden = true;
    document.getElementById("guide-button").focus();
  };
  mount(
    drawer,
    h("div", { class: "guide-head" }, h("h2", {}, "Guide"), h("button", { onclick: close }, "Close")),
    guidePages(ctx.tutorial, ctx.evaluator.domain).map(page => h("section", {}, h("h3", {}, page.title), richText(page.body))),
    h("p", { class: "muted" }, `Questions? ${ctx.site.contact}`),
  );
  drawer.onkeydown = event => event.key === "Escape" && close();
  drawer.hidden = false;
  drawer.focus();
}

function showLanding(ctx, message = "") {
  const input = h("input", {
    type: "text",
    id: "code-input",
    autocomplete: "off",
    autocapitalize: "characters",
    spellcheck: "false",
    placeholder: "TP-XXXX-XXXX",
  });
  const form = h(
    "form",
    {
      class: "card narrow",
      onsubmit: event => {
        event.preventDefault();
        const code = normalizeCode(input.value);
        if (code) signIn(ctx, code);
      },
    },
    h("h2", {}, "Welcome"),
    h("p", {}, "Enter the evaluator code you received, or open the link that came with it."),
    h("label", { for: "code-input" }, "Evaluator code"),
    h("div", { class: "inline" }, input, h("button", { class: "primary", type: "submit" }, "Continue")),
    message ? h("p", { class: "error-text", role: "alert" }, message) : null,
    h("p", { class: "muted small-print" }, h("a", { href: "?admin" }, "Research team sign-in")),
  );
  mount(root, form);
  input.focus();
}

async function signIn(ctx, code) {
  mount(root, h("p", { class: "muted" }, "Checking your code…"));
  let result;
  try {
    result = await ctx.store.claimEvaluator(code);
  } catch (error) {
    return failure("Could not check your code", error);
  }
  if (result.error) {
    localStorage.removeItem(CODE_KEY);
    return showLanding(
      ctx,
      result.error === "disabled"
        ? "This code has been deactivated. Please contact the research team."
        : "We could not find that code. Check it and try again.",
    );
  }
  localStorage.setItem(CODE_KEY, code);
  ctx.evaluator = result.evaluator;
  setHeader(ctx, `${code} · ${SITE.domainLabel[result.evaluator.domain] || result.evaluator.domain}`, { guide: true });
  try {
    await startEvaluator(ctx);
  } catch (error) {
    failure("Could not load your work", error);
  }
}

function showAdminLogin(ctx, message = "") {
  const input = h("input", { type: "password", id: "admin-key", autocomplete: "off" });
  const form = h(
    "form",
    {
      class: "card narrow",
      onsubmit: async event => {
        event.preventDefault();
        try {
          if (await ctx.store.claimAdmin(input.value.trim())) return openAdmin(ctx);
          showAdminLogin(ctx, "That key was not accepted.");
        } catch (error) {
          failure("Could not check the key", error);
        }
      },
    },
    h("h2", {}, "Research team sign-in"),
    h("label", { for: "admin-key" }, "Admin key"),
    h("div", { class: "inline" }, input, h("button", { class: "primary", type: "submit" }, "Sign in")),
    message ? h("p", { class: "error-text", role: "alert" }, message) : null,
    ctx.store.mode === "local" ? h("p", { class: "muted" }, "Local test mode uses the same admin key as the deployed site, installed with set_admin_key.py.") : null,
  );
  mount(root, form);
  input.focus();
}

function openAdmin(ctx) {
  setHeader(ctx, "Admin");
  return showAdmin(ctx);
}

async function boot() {
  document.title = SITE.title;
  document.getElementById("site-title").textContent = SITE.title;
  let store;
  let pool;
  try {
    [store, pool] = await Promise.all([createStore(SITE), fetchJson("data/pool.json")]);
  } catch (error) {
    return failure("The site could not start", error);
  }
  const poolItems = new Map(
    Object.entries(pool.domains).flatMap(([domain, entry]) => entry.items.map(item => [item.item_id, { ...item, domain }])),
  );
  const ctx = { store, pool, poolItems, site: SITE, rubric: RUBRIC, tutorial: TUTORIAL, root };
  showBanners(ctx);

  const params = new URLSearchParams(location.search);
  if (params.has("admin")) {
    try {
      return (await store.isAdmin()) ? openAdmin(ctx) : showAdminLogin(ctx);
    } catch (error) {
      return failure("Could not check admin access", error);
    }
  }
  const code = normalizeCode(params.get("code") || localStorage.getItem(CODE_KEY));
  return code ? signIn(ctx, code) : showLanding(ctx);
}

boot();
