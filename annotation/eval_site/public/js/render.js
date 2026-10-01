// Renders reference labels, plans, products, and candidate records for both domains.

import { TIERS, TIER_LABEL, activeFieldPaths, tierOf } from "./core.js";
import { details, h } from "./ui.js";

const ENTITY_PATH = /^entities\.([^.]+)\.constraints\.(.+)$/;
const OP_LABEL = {
  add: "Added",
  override: "Changed",
  relax: "Relaxed",
  tighten: "Tightened",
  remove: "Removed",
  reprioritize: "Priority changed",
  set: "Set",
};
const PLAN_FIELDS = [
  ["transportation", "Transportation"],
  ["breakfast", "Breakfast"],
  ["lunch", "Lunch"],
  ["dinner", "Dinner"],
  ["attraction", "Attractions"],
  ["accommodation", "Accommodation"],
];
// Map and contact columns add noise without helping anyone check a plan.
const HIDDEN_RECORD_KEYS = new Set(["Latitude", "Longitude", "Phone", "Website", "result_index"]);

export function prettyKey(key) {
  const text = String(key).replace(/_/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function fieldLabel(path, state) {
  const match = ENTITY_PATH.exec(path);
  if (!match) return prettyKey(path);
  const reference = state?.entities?.[match[1]]?.reference || match[1];
  return `${prettyKey(match[2])} (for ${reference})`;
}

function fieldValue(path, state) {
  const match = ENTITY_PATH.exec(path);
  return match ? state?.entities?.[match[1]]?.constraints?.[match[2]] : state?.constraints?.[path];
}

export function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, inner]) => `${prettyKey(key)}: ${formatValue(inner)}`)
      .join("; ");
  }
  return String(value);
}

function money(value) {
  const number = Number(value);
  return value === null || value === undefined || value === "" || Number.isNaN(number) ? formatValue(value) : `$${number.toFixed(2)}`;
}

export function tierChip(tier) {
  return h("span", { class: `tier tier-${tier || "none"}` }, tier ? TIER_LABEL[tier] : "No tier");
}

export function stateTable(state, { highlight = [] } = {}) {
  const paths = activeFieldPaths(state);
  if (!paths.length) return h("p", { class: "muted" }, "No active requirements.");
  const rank = path => {
    const tier = tierOf(path, state.priority);
    return tier ? TIERS.indexOf(tier) : TIERS.length;
  };
  return h(
    "table",
    { class: "grid" },
    h("thead", {}, h("tr", {}, h("th", {}, "Requirement"), h("th", {}, "Value"), h("th", {}, "Priority"))),
    h(
      "tbody",
      {},
      [...paths]
        .sort((a, b) => rank(a) - rank(b))
        .map(path =>
          h(
            "tr",
            { class: highlight.includes(path) ? "changed" : null },
            h("td", {}, fieldLabel(path, state)),
            h("td", {}, formatValue(fieldValue(path, state))),
            h("td", {}, tierChip(tierOf(path, state.priority))),
          ),
        ),
    ),
  );
}

export function describeChange(change) {
  const show = value => (change.op === "reprioritize" && TIER_LABEL[value] ? TIER_LABEL[value] : formatValue(value));
  const op = OP_LABEL[change.op] || prettyKey(change.op || "changed");
  if (change.op === "add" || change.op === "set") return `${op}: ${show(change.new)}`;
  if (change.op === "remove") return `${op} (was ${show(change.old)})`;
  return `${op}: ${show(change.old)} → ${show(change.new)}`;
}

export function deltaList(delta, state) {
  const entries = Object.entries(delta || {});
  if (!entries.length) return h("p", { class: "muted" }, "No change recorded for this turn.");
  return h(
    "ul",
    { class: "delta" },
    entries.map(([path, change]) =>
      h(
        "li",
        {},
        h("strong", {}, fieldLabel(path, state)),
        h("span", { class: "delta-change" }, describeChange(change)),
        change.op === "remove" ? null : tierChip(tierOf(path, state?.priority)),
      ),
    ),
  );
}

export function contextHeader(item, domainLabel) {
  const trip = item.context?.trip;
  if (!trip) return h("p", { class: "context" }, domainLabel);
  const dates = Array.isArray(trip.date) && trip.date.length ? `${trip.date[0]} to ${trip.date[trip.date.length - 1]}` : null;
  const facts = [
    trip.org && trip.dest ? `${trip.org} → ${trip.dest}` : null,
    trip.days ? `${trip.days} days` : null,
    dates,
    trip.people_number ? `${trip.people_number} traveler${trip.people_number === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return h("p", { class: "context" }, `${domainLabel}: ${facts.join(" · ")} (initial request)`);
}

export function actionView(action) {
  if (!action) return h("p", { class: "muted" }, "No reference action for this turn.");
  return action.kind === "product" ? productView(action) : planView(action);
}

function planView(action) {
  const itinerary = action.itinerary || [];
  const ledger = action.cost_ledger || [];
  // One card per day, like the annotation tool; a wide table is unreadable in the review column.
  return h(
    "div",
    { class: "plan" },
    h(
      "ol",
      { class: "plan-days" },
      itinerary.map(day =>
        h(
          "li",
          { class: "plan-day" },
          h("div", { class: "plan-day-head" }, h("strong", {}, formatValue(day.day)), ` · ${formatValue(day.current_city)}`),
          h(
            "dl",
            {},
            PLAN_FIELDS.flatMap(([key, label]) => [
              h("dt", {}, label),
              h("dd", { class: /UNRESOLVED/i.test(String(day[key] || "")) ? "unresolved" : null }, formatValue(day[key])),
            ]),
          ),
        ),
      ),
    ),
    ledger.length
      ? details(
          `Cost breakdown: known subtotal ${money(action.known_cost_subtotal)}`,
          h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              { class: "grid" },
              h("thead", {}, h("tr", {}, ["Item", "Date", "Unit price", "Qty", "Total", "Basis"].map(label => h("th", {}, label)))),
              h(
                "tbody",
                {},
                ledger.map(entry =>
                  h(
                    "tr",
                    {},
                    h("td", {}, [prettyKey(entry.kind || ""), entry.name].filter(Boolean).join(": ")),
                    h("td", {}, formatValue(entry.date)),
                    h("td", {}, money(entry.unit_price)),
                    h("td", {}, formatValue(entry.quantity)),
                    h("td", {}, money(entry.total)),
                    h("td", { class: "muted" }, formatValue(entry.evidence)),
                  ),
                ),
              ),
            ),
          ),
        )
      : null,
    action.not_in_subtotal?.length
      ? h("div", { class: "note" }, h("strong", {}, "Not included in the subtotal: "), h("ul", {}, action.not_in_subtotal.map(text => h("li", {}, text))))
      : null,
  );
}

function productView(action) {
  const product = action.product;
  const chosen = Object.entries(action.selected_options || {});
  if (!product) return h("p", {}, `Product ${action.asin}. Details are not available for this listing.`);
  const facts = [
    money(product.price),
    product.category,
    product.rating ? `rated ${product.rating} (${product.reviews ?? 0} reviews)` : null,
  ].filter(Boolean);
  return h(
    "div",
    { class: "product" },
    product.image_url ? h("img", { src: product.image_url, alt: "", loading: "lazy", referrerpolicy: "no-referrer" }) : null,
    h(
      "div",
      {},
      h("h4", {}, product.title || action.asin),
      h("p", { class: "muted" }, `${facts.join(" · ")} · ASIN ${action.asin}`),
      chosen.length ? h("p", {}, h("strong", {}, "Selected options: "), chosen.map(([k, v]) => `${k}: ${v}`).join(", ")) : null,
      Object.keys(product.options || {}).length
        ? h("p", {}, h("strong", {}, "Available options: "), formatValue(product.options))
        : null,
      product.bullet_points?.length ? h("ul", {}, product.bullet_points.map(point => h("li", {}, point))) : null,
      Object.keys(product.information || {}).length
        ? details(
            "Product information",
            h("dl", {}, Object.entries(product.information).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
          )
        : null,
      product.description ? details("Description", h("p", {}, product.description)) : null,
    ),
  );
}

/** Candidate records (TravelPlanner reference information) with a text filter. */
export function evidenceView(evidence) {
  const categories = Object.entries(evidence?.reference_information || {});
  if (!categories.length) return null;
  const sections = categories.map(([name, records]) => recordSection(name, records));
  const filter = h("input", {
    type: "search",
    placeholder: "Filter records, for example a hotel name or flight number",
    "aria-label": "Filter candidate records",
  });
  filter.addEventListener("input", () => {
    const query = filter.value.trim().toLowerCase();
    for (const section of sections) section.filter(query);
  });
  return h("div", { class: "evidence" }, filter, sections.map(section => section.element));
}

function recordSection(name, records) {
  const rows = Array.isArray(records) ? records : [records];
  const objects = rows.every(row => row && typeof row === "object" && !Array.isArray(row));
  if (!objects) {
    const element = details(`${name}`, h("p", {}, formatValue(records)));
    const text = `${name} ${formatValue(records)}`.toLowerCase();
    return { element, filter: query => (element.hidden = Boolean(query) && !text.includes(query)) };
  }
  const columns = [...new Set(rows.flatMap(row => Object.keys(row)))].filter(key => !HIDDEN_RECORD_KEYS.has(key));
  const bodyRows = rows.map(row => {
    const tr = h("tr", {}, columns.map(key => h("td", {}, formatValue(row[key]))));
    return { tr, text: columns.map(key => formatValue(row[key])).join(" ").toLowerCase() };
  });
  const summary = h("summary", {}, `${name} (${rows.length})`);
  const element = h(
    "details",
    {},
    summary,
    h(
      "div",
      { class: "table-wrap" },
      h("table", { class: "grid records" }, h("thead", {}, h("tr", {}, columns.map(key => h("th", {}, key)))), h("tbody", {}, bodyRows.map(row => row.tr))),
    ),
  );
  return {
    element,
    filter(query) {
      let shown = 0;
      for (const row of bodyRows) {
        row.tr.hidden = Boolean(query) && !row.text.includes(query) && !name.toLowerCase().includes(query);
        if (!row.tr.hidden) shown += 1;
      }
      element.hidden = Boolean(query) && shown === 0;
      element.open = Boolean(query) && shown > 0;
      summary.textContent = query ? `${name} (${shown} of ${rows.length})` : `${name} (${rows.length})`;
    },
  };
}
