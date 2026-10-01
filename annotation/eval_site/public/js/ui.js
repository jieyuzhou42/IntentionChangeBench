// Small DOM helpers. Data is always inserted as text, never as HTML.

export function h(tag, attrs, ...children) {
  const element = document.createElement(tag);
  let value;
  for (const [key, attr] of Object.entries(attrs || {})) {
    if (attr === null || attr === undefined || attr === false) continue;
    if (key === "class") element.className = attr;
    else if (key === "value") value = attr;
    else if (key.startsWith("on") && typeof attr === "function") element.addEventListener(key.slice(2), attr);
    else element.setAttribute(key, attr === true ? "" : String(attr));
  }
  append(element, children);
  // Set after the children exist, so a <select> can pick one of its options.
  if (value !== undefined) element.value = value;
  return element;
}

function append(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function mount(root, ...children) {
  root.replaceChildren();
  append(root, children);
}

/** Paragraphs, with consecutive "- " lines grouped into one bullet list. */
export function richText(lines) {
  const out = [];
  let list = null;
  for (const line of lines || []) {
    if (line.startsWith("- ")) {
      if (!list) out.push((list = h("ul")));
      list.append(h("li", {}, line.slice(2)));
    } else {
      list = null;
      out.push(h("p", {}, line));
    }
  }
  return out;
}

export function details(summary, content, open = false) {
  return h("details", { open }, h("summary", {}, summary), content);
}

export function toast(message, kind = "info") {
  const note = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, message);
  document.getElementById("toasts").append(note);
  setTimeout(() => note.remove(), kind === "error" ? 8000 : 3500);
}

export function download(filename, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text, done = "Copied.") {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch {
    toast("Copying failed. Select the text and copy it by hand.", "error");
  }
}

export function formatTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? String(iso) : date.toLocaleString();
}

export async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`);
  return response.json();
}
