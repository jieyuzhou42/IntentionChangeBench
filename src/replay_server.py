from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from flask import Flask, jsonify, render_template_string, request, send_from_directory


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "IntentionChangeBench" / "data" / "simulation" / "simulated_dataset.json"
DEFAULT_CACHE_PATH = REPO_ROOT / "IntentionChangeBench" / "data" / "replay_image_cache.json"
SMALL_CATALOG_PATH = REPO_ROOT / "WebShop" / "data" / "items_shuffle_1000.json"
FULL_CATALOG_PATH = REPO_ROOT / "WebShop" / "data" / "items_shuffle.json"
NO_IMAGE_PATH = REPO_ROOT / "WebShop" / "web_agent_site" / "static" / "images"


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WebShop Replay</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #1f2933;
      --muted: #5f6b7a;
      --blue: #1f6feb;
      --green: #1f7a4d;
      --red: #b42318;
      --amber: #8a6100;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
    }
    .wrap {
      width: min(1240px, calc(100vw - 32px));
      margin: 0 auto;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto auto;
      gap: 10px;
      align-items: end;
      padding: 12px 0;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    select, button {
      height: 34px;
      border: 1px solid #b8c0ca;
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    select {
      width: 100%;
      padding: 0 8px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid #b8c0ca;
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    textarea {
      min-height: 76px;
      padding: 9px 10px;
      resize: vertical;
    }
    input {
      height: 34px;
      padding: 0 8px;
    }
    button {
      min-width: 38px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      border-color: #1f6feb;
      background: #1f6feb;
      color: #fff;
      font-weight: 700;
    }
    button.danger {
      border-color: #f1aaa4;
      color: var(--red);
      background: #fff4f2;
    }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    main {
      padding: 18px 0 32px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.9fr;
      gap: 14px;
      align-items: start;
      margin-bottom: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .panel-pad { padding: 14px; }
    .section-title {
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .utterance {
      margin: 0;
      font-size: 19px;
      font-weight: 700;
    }
    .editor-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .editor-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .status {
      min-width: 88px;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .status.error { color: var(--red); }
    .status.ok { color: var(--green); }
    .constraints {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .constraint-row {
      display: grid;
      grid-template-columns: minmax(120px, 0.7fr) minmax(160px, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      border: 1px solid #e0e4e8;
      border-radius: 5px;
      background: #fbfcfd;
      cursor: grab;
    }
    .constraint-row.dragging {
      opacity: 0.45;
    }
    .priority-board {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .priority-column {
      min-height: 92px;
      border: 1px dashed #b8c0ca;
      border-radius: 6px;
      background: #fbfcfd;
      padding: 8px;
    }
    .priority-column.over {
      border-color: var(--blue);
      background: #eef5ff;
    }
    .priority-title {
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .priority-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-height: 36px;
    }
    .priority-chip {
      border: 1px solid #cbd3dc;
      border-radius: 999px;
      padding: 4px 8px;
      background: #fff;
      color: #344054;
      font-size: 12px;
      cursor: grab;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: Consolas, Monaco, monospace;
      font-size: 12px;
    }
    .rationale-list {
      display: grid;
      gap: 8px;
    }
    .rationale {
      border-left: 3px solid #8a6100;
      padding-left: 9px;
      color: #2f3337;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border: 1px solid #cbd3dc;
      border-radius: 999px;
      padding: 2px 8px;
      color: #344054;
      background: #fff;
      font-size: 12px;
    }
    .pill.good { border-color: #9bd0b5; color: var(--green); background: #f0faf5; }
    .pill.bad { border-color: #f1aaa4; color: var(--red); background: #fff4f2; }
    .pill.warn { border-color: #e3c36a; color: var(--amber); background: #fff9e8; }
    .results-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      border-radius: 6px 6px 0 0;
    }
    .results-title {
      margin: 0;
      font-size: 20px;
    }
    .query {
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .item {
      display: grid;
      grid-template-columns: 210px minmax(0, 1fr);
      gap: 18px;
      padding: 16px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .item:last-child {
      border-bottom: 0;
      border-radius: 0 0 6px 6px;
    }
    .image-box {
      width: 210px;
      aspect-ratio: 1 / 1;
      border: 1px solid #e0e4e8;
      background: #fafafa;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .image-box img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .asin {
      margin: 0 0 4px;
      color: var(--blue);
      font-size: 18px;
      font-weight: 700;
    }
    .title {
      margin: 0 0 8px;
      font-size: 17px;
      font-weight: 700;
    }
    .price {
      margin: 0 0 10px;
      font-size: 17px;
      font-weight: 700;
    }
    .details {
      color: var(--muted);
      margin-bottom: 9px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }
    .small-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-top: 10px;
    }
    .empty {
      padding: 28px 14px;
      color: var(--muted);
      background: #fff;
      border-radius: 0 0 6px 6px;
    }
    @media (max-width: 820px) {
      .toolbar {
        grid-template-columns: 1fr 1fr;
      }
      .top-grid {
        grid-template-columns: 1fr;
      }
      .priority-board {
        grid-template-columns: 1fr;
      }
      .constraint-row {
        grid-template-columns: 1fr;
      }
      .item {
        grid-template-columns: 120px minmax(0, 1fr);
        gap: 12px;
      }
      .image-box {
        width: 120px;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap toolbar">
      <div>
        <label for="instanceSelect">Instance</label>
        <select id="instanceSelect"></select>
      </div>
      <div>
        <label for="turnSelect">Turn</label>
        <select id="turnSelect"></select>
      </div>
      <button id="prevTurn" title="Previous turn">&lt;</button>
      <button id="nextTurn" title="Next turn">&gt;</button>
      <button id="reload" class="primary">Reload</button>
    </div>
  </header>
  <main>
    <div class="wrap">
      <div class="top-grid">
        <section class="panel panel-pad">
          <div class="editor-head">
            <h2 class="section-title">User Utterance</h2>
            <div class="editor-actions">
              <span id="saveStatus" class="status"></span>
              <button id="saveEdit" class="primary">Save</button>
            </div>
          </div>
          <textarea id="utterance" class="utterance"></textarea>
          <div id="meta" class="meta-row"></div>
        </section>
        <section class="panel panel-pad">
          <div class="editor-head">
            <h2 class="section-title">Constraints</h2>
            <button id="addConstraint">Add</button>
          </div>
          <div id="constraints" class="constraints"></div>
          <div id="priorityBoard" class="priority-board"></div>
        </section>
      </div>
      <section class="panel panel-pad" style="margin-bottom:16px;">
        <h2 class="section-title">Rationale</h2>
        <div id="rationale" class="rationale-list"></div>
      </section>
      <section class="panel">
        <div class="results-head">
          <div>
            <h2 id="resultsTitle" class="results-title">Search Results</h2>
            <div id="query" class="query"></div>
          </div>
          <div id="resultCount" class="pill"></div>
        </div>
        <div id="results"></div>
      </section>
    </div>
  </main>
  <script>
    const state = {{ state_json | safe }};
    const instanceSelect = document.getElementById("instanceSelect");
    const turnSelect = document.getElementById("turnSelect");
    const utterance = document.getElementById("utterance");
    const constraints = document.getElementById("constraints");
    const priorityBoard = document.getElementById("priorityBoard");
    const saveEdit = document.getElementById("saveEdit");
    const saveStatus = document.getElementById("saveStatus");
    const rationale = document.getElementById("rationale");
    const meta = document.getElementById("meta");
    const results = document.getElementById("results");
    const resultsTitle = document.getElementById("resultsTitle");
    const resultCount = document.getElementById("resultCount");
    const query = document.getElementById("query");
    const prevTurn = document.getElementById("prevTurn");
    const nextTurn = document.getElementById("nextTurn");
    const priorityLevels = [
      ["high", "High"],
      ["medium", "Medium"],
      ["low", "Low"]
    ];
    let constraintDraft = [];
    let priorityDraft = { high: [], medium: [], low: [] };
    let draggedKey = "";

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function money(value) {
      if (value === null || value === undefined || value === "") return "";
      const num = Number(value);
      return Number.isFinite(num) ? `$${num.toFixed(2)}` : String(value);
    }
    function pill(text, cls = "") {
      return `<span class="pill ${cls}">${esc(text)}</span>`;
    }
    function firstNonEmpty(...values) {
      for (const value of values) {
        if (value !== undefined && value !== null && String(value).trim() !== "") return value;
      }
      return "";
    }
    function valueToText(value) {
      if (value === null || typeof value === "object") return JSON.stringify(value);
      return String(value);
    }
    function textToValue(text) {
      const trimmed = String(text ?? "").trim();
      if (trimmed === "") return "";
      try {
        return JSON.parse(trimmed);
      } catch {
        return text;
      }
    }
    function currentTurn() {
      const instIndex = Number(instanceSelect.value || 0);
      const turnIndex = Number(turnSelect.value || 0);
      return state.instances[instIndex].turns[turnIndex];
    }
    function normalizePriority(priority, keys) {
      const keySet = new Set(keys);
      const next = { high: [], medium: [], low: [] };
      if (priority && !Array.isArray(priority) && typeof priority === "object") {
        for (const level of Object.keys(next)) {
          for (const key of priority[level] || []) {
            if (keySet.has(key) && !next[level].includes(key)) next[level].push(key);
          }
        }
      } else if (Array.isArray(priority)) {
        next.medium = priority.filter(key => keySet.has(key));
      }
      const assigned = new Set([...next.high, ...next.medium, ...next.low]);
      for (const key of keys) {
        if (!assigned.has(key)) next.medium.push(key);
      }
      return next;
    }
    function renderEditor(turn) {
      const gold = turn.gold_current_intention || {};
      const rawConstraints = gold.constraints || {};
      constraintDraft = Object.entries(rawConstraints).map(([key, value]) => ({
        key,
        valueText: valueToText(value)
      }));
      priorityDraft = normalizePriority(gold.priority, constraintDraft.map(row => row.key));
      utterance.value = turn.user_utterance || "";
      saveStatus.textContent = "";
      saveStatus.className = "status";
      renderConstraints();
      renderPriorityBoard();
    }
    function renderConstraints() {
      constraints.innerHTML = constraintDraft.map((row, index) => `
        <div class="constraint-row" draggable="true" data-key="${esc(row.key)}">
          <input data-role="key" data-index="${index}" value="${esc(row.key)}" placeholder="constraint">
          <input data-role="value" data-index="${index}" value="${esc(row.valueText)}" placeholder="value, JSON allowed">
          <button class="danger" data-role="remove" data-index="${index}" title="Remove">Remove</button>
        </div>
      `).join("");
    }
    function removeFromPriority(key) {
      for (const level of Object.keys(priorityDraft)) {
        priorityDraft[level] = priorityDraft[level].filter(item => item !== key);
      }
    }
    function renderPriorityBoard() {
      priorityDraft = normalizePriority(priorityDraft, constraintDraft.map(row => row.key).filter(Boolean));
      priorityBoard.innerHTML = priorityLevels.map(([level, label]) => `
        <div class="priority-column" data-level="${level}">
          <p class="priority-title">${esc(label)}</p>
          <div class="priority-list">
            ${(priorityDraft[level] || []).map(key => `
              <span class="priority-chip" draggable="true" data-key="${esc(key)}">${esc(key)}</span>
            `).join("")}
          </div>
        </div>
      `).join("");
    }
    function collectEditPayload() {
      const cleanRows = constraintDraft
        .map(row => ({ key: row.key.trim(), valueText: row.valueText }))
        .filter(row => row.key);
      const constraintsPayload = {};
      for (const row of cleanRows) constraintsPayload[row.key] = textToValue(row.valueText);
      const keys = cleanRows.map(row => row.key);
      const priorityPayload = normalizePriority(priorityDraft, keys);
      return {
        user_utterance: utterance.value,
        constraints: constraintsPayload,
        priority: priorityPayload
      };
    }
    async function saveCurrentEdit() {
      const instIndex = Number(instanceSelect.value || 0);
      const turnIndex = Number(turnSelect.value || 0);
      saveEdit.disabled = true;
      saveStatus.textContent = "Saving...";
      saveStatus.className = "status";
      try {
        const response = await fetch("/api/update_turn", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ instance_index: instIndex, turn_index: turnIndex, ...collectEditPayload() })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Save failed");
        const turn = currentTurn();
        turn.user_utterance = utterance.value;
        turn.gold_current_intention = turn.gold_current_intention || {};
        turn.gold_current_intention.constraints = data.turn.gold_current_intention.constraints;
        turn.gold_current_intention.priority = data.turn.gold_current_intention.priority;
        saveStatus.textContent = "Saved";
        saveStatus.className = "status ok";
        render();
      } catch (error) {
        saveStatus.textContent = error.message;
        saveStatus.className = "status error";
      } finally {
        saveEdit.disabled = false;
      }
    }
    function renderSelectors() {
      instanceSelect.innerHTML = state.instances.map((inst, index) => {
        const label = `${index + 1}. ${inst.instance_id}`;
        return `<option value="${index}">${esc(label)}</option>`;
      }).join("");
      renderTurnOptions();
    }
    function renderTurnOptions() {
      const inst = state.instances[Number(instanceSelect.value || 0)];
      turnSelect.innerHTML = inst.turns.map((turn, index) => {
        const label = `Turn ${turn.turn_id ?? index}`;
        return `<option value="${index}">${esc(label)}</option>`;
      }).join("");
    }
    function render() {
      const instIndex = Number(instanceSelect.value || 0);
      const turnIndex = Number(turnSelect.value || 0);
      const inst = state.instances[instIndex];
      const turn = inst.turns[turnIndex];
      const feedback = turn.env_feedback || {};
      const items = feedback.candidate_items || [];
      const gold = turn.gold_current_intention || {};
      const action = turn.agent_action || {};
      const actionPayload = action.action_payload || {};

      renderEditor(turn);
      meta.innerHTML = [
        pill(inst.instance_id),
        pill(`turn ${turn.turn_id ?? turnIndex}`),
        turn.shift_condition?.type ? pill(turn.shift_condition.type, "warn") : "",
        turn.action_implication ? pill(turn.action_implication) : "",
        turn.stop_reason ? pill(`stop: ${turn.stop_reason}`) : ""
      ].join("");

      const rationales = turn.rationales || [];
      rationale.innerHTML = rationales.length
        ? rationales.map(text => `<div class="rationale">${esc(text)}</div>`).join("")
        : `<div class="rationale">No rationale recorded for this turn.</div>`;

      const queryText = firstNonEmpty(
        feedback.gold_search_query,
        gold.gold_search_query,
        actionPayload.query,
        (turn.rollout_search_queries || [])[0]
      );
      resultsTitle.textContent = `Page 1 (Total results: ${items.length})`;
      resultCount.textContent = `${items.length} items`;
      query.textContent = queryText ? `Search query: ${queryText}` : "";

      if (!items.length) {
        results.innerHTML = `<div class="empty">No candidate_items found in env_feedback for this turn.</div>`;
      } else {
        results.innerHTML = items.map(renderItem).join("");
      }
      prevTurn.disabled = turnIndex <= 0;
      nextTurn.disabled = turnIndex >= inst.turns.length - 1;
    }
    function renderItem(item) {
      const imageUrl = item.image_url || state.no_image_url;
      const rankBits = [
        item.rerank_rank ? `rerank #${item.rerank_rank}` : "",
        item.original_rank ? `original #${item.original_rank}` : "",
        item.rank ? `rank #${item.rank}` : ""
      ].filter(Boolean);
      const matched = item.rerank_matched_constraints || [];
      const missing = item.rerank_missing_or_uncertain_constraints || [];
      const mismatches = item.rerank_mismatch_reasons || [];
      const attrs = item.attributes || [];
      return `
        <article class="item">
          <div class="image-box"><img src="${esc(imageUrl)}" alt=""></div>
          <div>
            <h3 class="asin">${esc(item.asin || "")}</h3>
            <h4 class="title">${esc(item.title || item.Title || "")}</h4>
            <p class="price">${esc(money(firstNonEmpty(item.price, item.Price)))}</p>
            <div class="details">
              ${rankBits.map(x => esc(x)).join(" | ")}
              ${item.category ? ` | ${esc(item.category)}` : ""}
              ${item.product_category ? `<br>${esc(item.product_category)}` : ""}
            </div>
            <div class="chips">
              ${item.rerank_decision ? pill(item.rerank_decision, item.rerank_decision === "keep" ? "good" : "warn") : ""}
              ${item.rerank_constraint_match_level ? pill(item.rerank_constraint_match_level) : ""}
              ${item.rerank_product_family_match ? pill(`family: ${item.rerank_product_family_match}`) : ""}
            </div>
            ${matched.length ? `<div class="small-title">Matched constraints</div><div class="chips">${matched.map(x => pill(x, "good")).join("")}</div>` : ""}
            ${missing.length ? `<div class="small-title">Missing or uncertain</div><div class="chips">${missing.map(x => pill(x, "warn")).join("")}</div>` : ""}
            ${mismatches.length ? `<div class="small-title">Mismatch reasons</div><div class="chips">${mismatches.map(x => pill(x, "bad")).join("")}</div>` : ""}
            ${attrs.length ? `<div class="small-title">Attributes</div><div class="chips">${attrs.slice(0, 12).map(x => pill(x)).join("")}</div>` : ""}
          </div>
        </article>`;
    }

    instanceSelect.addEventListener("change", () => {
      renderTurnOptions();
      turnSelect.value = "0";
      render();
    });
    turnSelect.addEventListener("change", render);
    prevTurn.addEventListener("click", () => {
      turnSelect.value = String(Math.max(0, Number(turnSelect.value) - 1));
      render();
    });
    nextTurn.addEventListener("click", () => {
      const inst = state.instances[Number(instanceSelect.value || 0)];
      turnSelect.value = String(Math.min(inst.turns.length - 1, Number(turnSelect.value) + 1));
      render();
    });
    document.getElementById("addConstraint").addEventListener("click", () => {
      let index = constraintDraft.length + 1;
      let key = `constraint_${index}`;
      const existing = new Set(constraintDraft.map(row => row.key));
      while (existing.has(key)) {
        index += 1;
        key = `constraint_${index}`;
      }
      constraintDraft.push({ key, valueText: "" });
      priorityDraft.medium.push(key);
      renderConstraints();
      renderPriorityBoard();
    });
    constraints.addEventListener("input", event => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement)) return;
      const index = Number(target.dataset.index);
      if (!Number.isInteger(index) || !constraintDraft[index]) return;
      if (target.dataset.role === "key") {
        const oldKey = constraintDraft[index].key;
        const newKey = target.value;
        constraintDraft[index].key = newKey;
        const rowEl = target.closest(".constraint-row");
        if (rowEl) rowEl.dataset.key = newKey;
        for (const level of Object.keys(priorityDraft)) {
          priorityDraft[level] = priorityDraft[level].map(key => key === oldKey ? newKey : key);
        }
        renderPriorityBoard();
      } else if (target.dataset.role === "value") {
        constraintDraft[index].valueText = target.value;
      }
    });
    constraints.addEventListener("click", event => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement) || target.dataset.role !== "remove") return;
      const index = Number(target.dataset.index);
      const row = constraintDraft[index];
      if (!row) return;
      removeFromPriority(row.key);
      constraintDraft.splice(index, 1);
      renderConstraints();
      renderPriorityBoard();
    });
    document.addEventListener("dragstart", event => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const key = target.dataset.key;
      if (!key) return;
      draggedKey = key;
      event.dataTransfer.setData("text/plain", key);
      target.classList.add("dragging");
    });
    document.addEventListener("dragend", event => {
      const target = event.target;
      if (target instanceof HTMLElement) target.classList.remove("dragging");
      draggedKey = "";
      document.querySelectorAll(".priority-column.over").forEach(el => el.classList.remove("over"));
    });
    priorityBoard.addEventListener("dragover", event => {
      const target = event.target;
      const column = target instanceof Element ? target.closest(".priority-column") : null;
      if (!column) return;
      event.preventDefault();
      column.classList.add("over");
    });
    priorityBoard.addEventListener("dragleave", event => {
      const target = event.target;
      const column = target instanceof Element ? target.closest(".priority-column") : null;
      if (column) column.classList.remove("over");
    });
    priorityBoard.addEventListener("drop", event => {
      const target = event.target;
      const column = target instanceof Element ? target.closest(".priority-column") : null;
      if (!column) return;
      event.preventDefault();
      const key = event.dataTransfer.getData("text/plain") || draggedKey;
      const level = column.dataset.level;
      if (!key || !priorityDraft[level]) return;
      removeFromPriority(key);
      priorityDraft[level].push(key);
      renderPriorityBoard();
    });
    saveEdit.addEventListener("click", saveCurrentEdit);
    document.getElementById("reload").addEventListener("click", () => location.reload());

    renderSelectors();
    render();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    started = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            if pos >= len(buffer):
                more = f.read(chunk_size)
                if not more:
                    break
                buffer = more
                pos = 0

            while True:
                while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                    pos += 1
                if not started and pos < len(buffer) and buffer[pos] == "[":
                    started = True
                    pos += 1
                    continue
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                if pos >= len(buffer):
                    break

                try:
                    item, next_pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    more = f.read(chunk_size)
                    if not more:
                        raise
                    buffer = buffer[pos:] + more
                    pos = 0
                    break

                pos = next_pos
                if isinstance(item, dict):
                    yield item

                if pos > chunk_size * 2:
                    buffer = buffer[pos:]
                    pos = 0
                    break


def collect_asins(instances: List[Dict[str, Any]]) -> set[str]:
    asins: set[str] = set()
    for instance in instances:
        for turn in instance.get("turns") or []:
            feedback = turn.get("env_feedback") or {}
            for item in feedback.get("candidate_items") or []:
                asin = str(item.get("asin") or "").strip()
                if asin:
                    asins.add(asin)
    return asins


def normalize_product_image(product: Dict[str, Any]) -> str:
    images = product.get("images")
    if isinstance(images, list):
        for image in images:
            image = str(image or "").strip()
            if image and "transparent-pixel" not in image:
                return image
    image = str(product.get("MainImage") or "").strip()
    return image


def load_catalog_images(
    asins: set[str],
    cache_path: Path,
    scan_full_catalog: bool,
) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    if cache_path.is_file():
        try:
            cached = load_json(cache_path)
            if isinstance(cached, dict):
                image_map.update({str(k): str(v) for k, v in cached.items() if v})
        except Exception:
            image_map = {}

    missing = {asin for asin in asins if asin not in image_map}
    for path, streaming in ((SMALL_CATALOG_PATH, False), (FULL_CATALOG_PATH, True)):
        if not missing or not path.is_file():
            continue
        if path == FULL_CATALOG_PATH and not scan_full_catalog:
            continue

        products = iter_json_array(path) if streaming else load_json(path)
        for product in products:
            asin = str(product.get("asin") or "").strip()
            if asin not in missing:
                continue
            image = normalize_product_image(product)
            if image:
                image_map[asin] = image
            missing.discard(asin)
            if not missing:
                break

    if image_map:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(image_map, f, indent=2, ensure_ascii=False)

    return image_map


def rationales_for_turn(turn: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    rationales: List[str] = []

    def add(value: Any) -> None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and text not in seen:
            seen.add(text)
            rationales.append(text)

    trigger = turn.get("trigger_evidence") or {}
    add(((trigger.get("details") or {}) if isinstance(trigger, dict) else {}).get("rationale"))
    condition = turn.get("shift_condition") or {}
    if isinstance(condition, dict):
        add(condition.get("reason"))
    delta = turn.get("gold_delta") or {}
    if isinstance(delta, dict):
        for change in delta.values():
            if isinstance(change, dict):
                add(change.get("rationale"))

    return rationales


def prepare_state(instances: List[Dict[str, Any]], image_map: Dict[str, str]) -> Dict[str, Any]:
    prepared = []
    for instance in instances:
        instance_copy = dict(instance)
        turns = []
        for turn in instance.get("turns") or []:
            turn_copy = dict(turn)
            feedback = dict(turn_copy.get("env_feedback") or {})
            items = []
            for item in feedback.get("candidate_items") or []:
                item_copy = dict(item)
                asin = str(item_copy.get("asin") or "").strip()
                item_copy["image_url"] = image_map.get(asin, "")
                items.append(item_copy)
            feedback["candidate_items"] = items
            turn_copy["env_feedback"] = feedback
            turn_copy["rationales"] = rationales_for_turn(turn)
            turns.append(turn_copy)
        instance_copy["turns"] = turns
        prepared.append(instance_copy)

    return {
        "instances": prepared,
        "no_image_url": "/static-webshop/images/no-image-available.png",
    }


def normalize_priority_payload(priority: Any, constraint_keys: List[str]) -> Dict[str, List[str]]:
    levels = ("high", "medium", "low")
    key_set = set(constraint_keys)
    normalized: Dict[str, List[str]] = {level: [] for level in levels}

    if isinstance(priority, dict):
        for level in levels:
            values = priority.get(level) or []
            if not isinstance(values, list):
                continue
            for key in values:
                key = str(key).strip()
                if key in key_set and key not in normalized[level]:
                    normalized[level].append(key)
    elif isinstance(priority, list):
        for key in priority:
            key = str(key).strip()
            if key in key_set and key not in normalized["medium"]:
                normalized["medium"].append(key)

    assigned = {key for values in normalized.values() for key in values}
    for key in constraint_keys:
        if key not in assigned:
            normalized["medium"].append(key)
    return normalized


def create_app(state: Dict[str, Any], instances: List[Dict[str, Any]], dataset_path: Path) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template_string(HTML, state_json=json.dumps(state, ensure_ascii=False))

    @app.route("/api/state")
    def api_state():
        return jsonify(state)

    @app.route("/api/update_turn", methods=["POST"])
    def api_update_turn():
        payload = request.get_json(silent=True) or {}
        try:
            instance_index = int(payload.get("instance_index"))
            turn_index = int(payload.get("turn_index"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid instance_index or turn_index"}), 400

        if instance_index < 0 or instance_index >= len(instances):
            return jsonify({"ok": False, "error": "Instance index out of range"}), 404
        turns = instances[instance_index].get("turns") or []
        if turn_index < 0 or turn_index >= len(turns):
            return jsonify({"ok": False, "error": "Turn index out of range"}), 404

        constraints_payload = payload.get("constraints")
        if not isinstance(constraints_payload, dict):
            return jsonify({"ok": False, "error": "constraints must be an object"}), 400

        constraints_clean = {
            str(key).strip(): value
            for key, value in constraints_payload.items()
            if str(key).strip()
        }
        priority_clean = normalize_priority_payload(payload.get("priority"), list(constraints_clean.keys()))
        turn = turns[turn_index]
        turn["user_utterance"] = str(payload.get("user_utterance") or "")
        gold = turn.setdefault("gold_current_intention", {})
        gold["constraints"] = constraints_clean
        gold["priority"] = priority_clean

        state_turn = state["instances"][instance_index]["turns"][turn_index]
        state_turn["user_utterance"] = turn["user_utterance"]
        state_gold = state_turn.setdefault("gold_current_intention", {})
        state_gold["constraints"] = constraints_clean
        state_gold["priority"] = priority_clean

        save_json(dataset_path, instances)
        return jsonify({"ok": True, "turn": state_turn})

    @app.route("/static-webshop/images/<path:filename>")
    def webshop_image(filename: str):
        return send_from_directory(NO_IMAGE_PATH, filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay WebShop simulated dataset with images.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--skip_full_catalog",
        action="store_true",
        help="Only use cached images and the 1k catalog. Startup is faster, but many images may be missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances = load_json(args.dataset)
    if not isinstance(instances, list):
        raise ValueError(f"Expected dataset JSON list, got {type(instances).__name__}")
    asins = collect_asins(instances)
    image_map = load_catalog_images(
        asins,
        cache_path=args.cache,
        scan_full_catalog=not args.skip_full_catalog,
    )
    state = prepare_state(instances, image_map)
    app = create_app(state, instances, args.dataset)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
