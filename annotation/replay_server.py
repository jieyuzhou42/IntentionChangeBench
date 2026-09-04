from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DATA_DIR = PROJECT_ROOT / "annotation" / "data"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "simulation" / "simulated_dataset.json"
DEFAULT_CACHE_PATH = ANNOTATION_DATA_DIR / "replay_image_cache.json"
DEFAULT_ITEM_CACHE_PATH = ANNOTATION_DATA_DIR / "replay_item_cache.json"
ITEM_CACHE_VERSION = 2
WEBSHOP_ROOT = PROJECT_ROOT / "WebShop"
if not (WEBSHOP_ROOT / "data" / "items_shuffle_1000.json").is_file():
    WEBSHOP_ROOT = PROJECT_ROOT.parent / "WebShop"
SMALL_CATALOG_PATH = WEBSHOP_ROOT / "data" / "items_shuffle_1000.json"
FULL_CATALOG_PATH = WEBSHOP_ROOT / "data" / "items_shuffle.json"
NO_IMAGE_PATH = WEBSHOP_ROOT / "web_agent_site" / "static" / "images"
SAVE_LOCK = threading.RLock()


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Replay</title>
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
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 12px 0;
    }
    .toolbar-primary-row, .toolbar-secondary-row, .toolbar-group, .shard-controls, .turn-controls {
      display: flex;
      align-items: flex-end;
      gap: 10px;
    }
    .toolbar-primary-row, .toolbar-secondary-row { flex-wrap: wrap; }
    .toolbar-group {
      flex-wrap: nowrap;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }
    .trajectory-group { flex: 1 1 560px; }
    .trajectory-control { flex: 1 1 260px; min-width: 220px; }
    .shard-controls, .turn-controls { flex: 0 0 auto; flex-wrap: nowrap; }
    .shard-controls .pill { height: 34px; min-width: auto; }
    .shard-picker { width: 96px; }
    .turn-picker { width: 80px; }
    .toolbar-secondary-row { align-items: center; }
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
      grid-template-columns: minmax(300px, 0.75fr) minmax(0, 1.45fr);
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
      min-height: 180px;
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
    .toolbar-status { min-width: 110px; text-align: left; }
    .toolbar-save { display: flex; align-items: center; gap: 8px; }
    .status.unsaved { color: var(--amber); }
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
    .item.gold-selected { box-shadow: inset 4px 0 0 var(--green); background: #f7fcf9; }
    .item-action { margin-top: 10px; }
    .option-summary { margin-top: 7px; color: var(--muted); font-size: 11px; line-height: 1.5; }
    .option-summary strong { color: #344054; }
    .item-evidence { margin-top: 10px; border-top: 1px solid #e0e4e8; padding-top: 8px; }
    .item-evidence summary { color: var(--blue); cursor: pointer; font-weight: 700; }
    .bullet-list { margin: 7px 0 0 18px; padding: 0; color: #344054; font-size: 12px; }
    .bullet-list li { margin-bottom: 4px; }
    .product-facts { display: grid; grid-template-columns: minmax(120px, 0.35fr) 1fr; gap: 4px 10px; margin-top: 8px; font-size: 12px; }
    .fact-key { color: var(--muted); }
    .fact-value { overflow-wrap: anywhere; }
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
    .travel-category {
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 14px;
    }
    .travel-category:last-child { border-bottom: 0; border-radius: 0 0 6px 6px; }
    .travel-category-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .travel-category-title { margin: 0; font-size: 17px; text-transform: capitalize; }
    .search-page {
      border: 1px solid #e0e4e8;
      border-radius: 6px;
      margin-top: 10px;
      overflow: hidden;
    }
    .search-page-head {
      padding: 8px 10px;
      background: #f7f9fb;
      border-bottom: 1px solid #e0e4e8;
      color: var(--muted);
    }
    .travel-items {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 8px;
      padding: 9px;
    }
    .travel-card {
      border: 1px solid #e0e4e8;
      border-radius: 5px;
      padding: 10px;
      background: #fff;
    }
    .travel-card-title { margin: 0 0 7px; color: var(--blue); font-size: 15px; }
    .kv { display: grid; grid-template-columns: minmax(90px, 0.35fr) 1fr; gap: 5px 9px; }
    .kv-key { color: var(--muted); font-size: 12px; }
    .kv-value { overflow-wrap: anywhere; }
    .day-list { display: grid; gap: 12px; }
    .day-card { border: 1px solid #dfe4ea; border-radius: 6px; overflow: hidden; background: #fff; }
    .day-head { padding: 9px 12px; background: #f1f6fc; border-bottom: 1px solid #dfe4ea; font-size: 16px; }
    .day-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 12px; }
    .day-field { min-width: 0; }
    .day-field label { margin-bottom: 3px; }
    .day-value { overflow-wrap: anywhere; white-space: pre-wrap; }
    .travel-cost-summary { margin-top: 14px; border: 1px solid #cbd3dc; border-radius: 6px; overflow: hidden; }
    .travel-cost-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 11px 12px; background: #f1f6fc; border-bottom: 1px solid #cbd3dc; }
    .travel-cost-head strong { font-size: 17px; }
    .travel-cost-day { padding: 10px 12px; border-bottom: 1px solid #e0e4e8; }
    .cost-row { display: grid; grid-template-columns: minmax(110px, .7fr) minmax(170px, 1fr) auto; gap: 10px; padding: 4px 0; align-items: baseline; }
    .cost-row .cost-formula { color: var(--muted); font-size: 12px; }
    .cost-row.unavailable { color: var(--amber); }
    .cost-subtotal { margin-top: 5px; padding-top: 7px; border-top: 1px dashed #cbd3dc; font-weight: 700; }
    .travel-cost-total { padding: 12px; background: #f7fcf9; }
    .travel-cost-total.over-budget { background: #fff4f2; }
    .travel-cost-note { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .gold-action-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .gold-confirm { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-weight: 700; }
    .gold-confirm input { width: auto; height: auto; }
    .gold-options { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 10px; }
    .gold-options select { width: 100%; }
    .substitution {
      border-left: 3px solid var(--amber);
      padding: 8px 10px;
      margin-top: 8px;
      background: #fff9e8;
    }
    @media (max-width: 820px) {
      .toolbar-primary-row, .toolbar-secondary-row { align-items: flex-end; }
      .toolbar-group { max-width: 100%; overflow-x: auto; }
      .trajectory-group { flex-basis: 100%; }
      .top-grid {
        grid-template-columns: 1fr;
      }
      .priority-board {
        grid-template-columns: 1fr;
      }
      .constraint-row {
        grid-template-columns: 1fr;
      }
      .day-grid, .gold-action-grid, .cost-row { grid-template-columns: 1fr; }
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
      <div class="toolbar-primary-row">
        <div class="toolbar-group trajectory-group">
          <div class="trajectory-control">
            <label id="instanceLabel" for="instanceSelect">Trajectories</label>
            <select id="instanceSelect"></select>
          </div>
          <div class="toolbar-save">
            <button id="saveEdit" class="primary" title="Save every turn in the current instance">Save Trajectory</button>
            <span id="saveStatus" class="status toolbar-status"></span>
          </div>
          <button id="deleteTrajectory" class="danger" title="Permanently delete this entire trajectory from the annotation output">Delete Trajectory</button>
        </div>
        <div class="toolbar-group shard-controls">
          <span id="shardIndicator" class="pill" style="display:none;"></span>
          <button id="prevShard" title="Switch to previous shard" style="display:none;">&lt; Shard</button>
          <div id="shardPicker" class="shard-picker" style="display:none;">
            <label for="shardSelect">Go to shard</label>
            <select id="shardSelect"></select>
          </div>
          <button id="switchShard" class="primary" title="Save current trajectory and switch shard" style="display:none;">Switch Shard</button>
          <button id="nextShard" title="Switch to next shard" style="display:none;">Shard &gt;</button>
        </div>
      </div>
      <div class="toolbar-secondary-row">
        <div class="toolbar-group turn-controls">
          <div class="turn-picker">
            <label for="turnSelect">Turn</label>
            <select id="turnSelect"></select>
          </div>
          <button id="prevTurn" title="Previous turn">&lt;</button>
          <button id="nextTurn" title="Next turn">&gt;</button>
          <button id="addTurn" title="Add a blank turn after this one">+ Turn</button>
          <button id="deleteTurn" class="danger" title="Delete this turn">Delete Turn</button>
        </div>
        <button id="reload" class="primary">Reload</button>
      </div>
    </div>
  </header>
  <main>
    <div class="wrap">
      <div class="top-grid">
        <section class="panel panel-pad utterance-panel">
          <div class="editor-head"><h2 class="section-title">User Utterance</h2></div>
          <textarea id="utterance" class="utterance"></textarea>
          <div id="meta" class="meta-row"></div>
        </section>
        <section class="panel panel-pad constraints-panel">
          <div class="editor-head">
            <div><h2 class="section-title">Constraints</h2><div class="details">Changes inherit forward to later turns unless a later turn explicitly overrides the value.</div></div>
            <button id="addConstraint">Add</button>
          </div>
          <div id="constraints" class="constraints"></div>
          <div id="priorityBoard" class="priority-board"></div>
        </section>
      </div>
      <section id="goldActionPanel" class="panel panel-pad" style="margin-bottom:16px;">
        <div class="editor-head">
          <div>
            <h2 class="section-title">Gold Action Confirmation</h2>
            <div class="details">Confirm the exact product/options or the exact day-by-day travel selections.</div>
          </div>
          <div id="goldActionStatus" class="pill warn">Unconfirmed</div>
        </div>
        <div id="goldActionEditor"></div>
        <div id="entityPanel" style="margin-top:16px; padding-top:16px; border-top:1px solid var(--line); display:none;">
          <h2 class="section-title">Entity Intentions &amp; Gold Changes</h2>
          <div id="entityState"></div>
        </div>
      </section>
      <section class="panel panel-pad" style="margin-bottom:16px;">
        <h2 class="section-title">Rationale</h2>
        <div id="rationale" class="rationale-list"></div>
      </section>
      <section id="webshopGoldPanel" class="panel" style="margin-bottom:16px; display:none;">
        <div class="results-head">
          <div>
            <h2 class="results-title">WebShop Original Gold Item</h2>
            <div class="query">The source product used by WebShop to create the original goal instruction.</div>
          </div>
          <div class="pill good">Turn 0 reference</div>
        </div>
        <div id="webshopGoldItem"></div>
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
    document.title = state.domain === "travelplanner" ? "TravelPlanner Replay" : "WebShop Replay";
    const instanceSelect = document.getElementById("instanceSelect");
    const instanceLabel = document.getElementById("instanceLabel");
    const shardIndicator = document.getElementById("shardIndicator");
    const shardPicker = document.getElementById("shardPicker");
    const shardSelect = document.getElementById("shardSelect");
    const switchShardButton = document.getElementById("switchShard");
    const prevShardButton = document.getElementById("prevShard");
    const nextShardButton = document.getElementById("nextShard");
    const turnSelect = document.getElementById("turnSelect");
    const utterance = document.getElementById("utterance");
    const constraints = document.getElementById("constraints");
    const priorityBoard = document.getElementById("priorityBoard");
    const entityPanel = document.getElementById("entityPanel");
    const entityState = document.getElementById("entityState");
    const saveEdit = document.getElementById("saveEdit");
    const saveStatus = document.getElementById("saveStatus");
    const deleteTrajectory = document.getElementById("deleteTrajectory");
    const rationale = document.getElementById("rationale");
    const meta = document.getElementById("meta");
    const results = document.getElementById("results");
    const resultsTitle = document.getElementById("resultsTitle");
    const resultCount = document.getElementById("resultCount");
    const query = document.getElementById("query");
    const prevTurn = document.getElementById("prevTurn");
    const nextTurn = document.getElementById("nextTurn");
    const addTurn = document.getElementById("addTurn");
    const deleteTurn = document.getElementById("deleteTurn");
    const goldActionEditor = document.getElementById("goldActionEditor");
    const goldActionStatus = document.getElementById("goldActionStatus");
    const webshopGoldPanel = document.getElementById("webshopGoldPanel");
    const webshopGoldItem = document.getElementById("webshopGoldItem");
    const priorityLevels = [
      ["high", "Must-have"],
      ["medium", "Preferred"],
      ["low", "Optional"]
    ];
    let constraintDraft = [];
    let priorityDraft = { high: [], medium: [], low: [] };
    let originalConstraints = {};
    let originalPriority = { high: [], medium: [], low: [] };
    let goldActionDraft = {};
    let draggedKey = "";
    let activeInstanceIndex = 0;
    let activeTurnIndex = 0;
    let editDirty = false;
    let navigationBusy = false;
    const dirtyInstances = new Set();
    let displayedItems = [];
    let candidateRequestToken = 0;

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
      return state.instances[activeInstanceIndex].turns[activeTurnIndex];
    }
    function markDirty() {
      editDirty = true;
      dirtyInstances.add(activeInstanceIndex);
      saveStatus.textContent = "Unsaved trajectory";
      saveStatus.className = "status toolbar-status unsaved";
    }
    function currentItems() {
      return displayedItems;
    }
    function actionItinerary(action) {
      return ((((action || {}).action_payload || {}).plan || {}).itinerary || []);
    }
    function compactSelection(value) {
      if (value && typeof value === "object") {
        return firstNonEmpty(value.name, value.Name, value.flight_number, value["Flight Number"], value.type, valueToText(value));
      }
      return valueToText(value ?? "");
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
      originalConstraints = JSON.parse(JSON.stringify(rawConstraints));
      constraintDraft = Object.entries(rawConstraints).map(([key, value]) => ({
        key,
        valueText: valueToText(value)
      }));
      priorityDraft = normalizePriority(gold.priority, constraintDraft.map(row => row.key));
      originalPriority = JSON.parse(JSON.stringify(priorityDraft));
      utterance.value = turn.user_utterance || "";
      editDirty = dirtyInstances.has(activeInstanceIndex);
      saveStatus.textContent = editDirty ? "Unsaved trajectory" : "";
      saveStatus.className = editDirty ? "status toolbar-status unsaved" : "status toolbar-status";
      renderConstraints();
      renderPriorityBoard();
    }
    function renderConstraints() {
      constraints.innerHTML = constraintDraft.map((row, index) => `
        <div class="constraint-row" data-key="${esc(row.key)}">
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
    function initialGoldAction(turn) {
      if (turn.gold_action && Object.keys(turn.gold_action).length) {
        return JSON.parse(JSON.stringify(turn.gold_action));
      }
      if (state.domain === "travelplanner") {
        const proposedDays = actionItinerary(turn.agent_action).map(day => {
          const next = {};
          for (const field of ["day", "current_city", "transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"]) {
            next[field] = compactSelection(day[field]);
          }
          return next;
        });
        return { action_type: "Planner", confirmed: false, action_payload: { plan: { itinerary: proposedDays } } };
      }
      return { action_type: "Buy", confirmed: false, action_payload: { selected_asin: "", selected_options: {} } };
    }
    function setGoldStatus() {
      const confirmed = Boolean(goldActionDraft.confirmed);
      goldActionStatus.textContent = confirmed ? "Confirmed" : "Unconfirmed";
      goldActionStatus.className = `pill ${confirmed ? "good" : "warn"}`;
    }
    function renderGoldAction(turn) {
      goldActionDraft = initialGoldAction(turn);
      if (state.domain === "travelplanner") renderTravelGoldAction();
      else renderWebshopGoldAction();
      setGoldStatus();
    }
    function renderWebshopGoldAction() {
      const payload = goldActionDraft.action_payload || (goldActionDraft.action_payload = {});
      const selectedOptions = payload.selected_options || (payload.selected_options = {});
      const selectedItem = currentItems().find(item => String(item.asin || "") === String(payload.selected_asin || ""));
      const availableOptions = (selectedItem && selectedItem.options) || {};
      const optionKeys = [...new Set([...Object.keys(availableOptions), ...Object.keys(selectedOptions)])];
      goldActionEditor.innerHTML = `
        <div class="gold-action-grid">
          <div><label>Selected product ASIN</label><input data-gold-role="asin" value="${esc(payload.selected_asin || "")}" placeholder="Choose an item below or enter its ASIN"></div>
          <div><label>Action</label><input value="Buy / select product" disabled></div>
        </div>
        <div class="gold-options">${optionKeys.map(key => {
          const values = Array.isArray(availableOptions[key]) ? availableOptions[key] : [];
          const current = selectedOptions[key] ?? "";
          return `<div><label>${esc(prettyKey(key))}</label>${values.length
            ? `<select data-gold-role="option" data-option-key="${esc(key)}"><option value="">Not selected</option>${values.map(value => `<option value="${esc(value)}" ${String(value) === String(current) ? "selected" : ""}>${esc(value)}</option>`).join("")}</select>`
            : `<input data-gold-role="option" data-option-key="${esc(key)}" value="${esc(current)}" placeholder="Selected value">`}</div>`;
        }).join("")}</div>
        <label class="gold-confirm"><input type="checkbox" data-gold-role="confirmed" ${goldActionDraft.confirmed ? "checked" : ""}> I confirm this is the gold product and these are the exact options.</label>`;
    }
    function renderTravelGoldAction() {
      const payload = goldActionDraft.action_payload || (goldActionDraft.action_payload = {});
      const plan = payload.plan || (payload.plan = {});
      const days = plan.itinerary || (plan.itinerary = []);
      const fields = ["day", "current_city", "transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"];
      goldActionEditor.innerHTML = `
        <div class="day-list">${days.map((day, dayIndex) => `
          <article class="day-card">
            <div class="day-head editor-head"><strong>${esc(day.day || `Day ${dayIndex + 1}`)}</strong><button class="danger" data-gold-role="remove-day" data-day-index="${dayIndex}">Remove day</button></div>
            <div class="day-grid">${fields.map(field => `<div class="day-field"><label>${esc(prettyKey(field))}</label><input data-gold-role="travel-field" data-day-index="${dayIndex}" data-field="${field}" value="${esc(day[field] ?? "")}" placeholder="Exact ${esc(prettyKey(field).toLowerCase())}"></div>`).join("")}</div>
          </article>`).join("")}</div>
        ${days.length ? "" : `<div class="empty">No proposed days. Add a day to define the gold itinerary.</div>`}
        <div id="travelCostSummary"></div>
        <div class="editor-actions" style="margin-top:10px;"><button data-gold-role="add-day">+ Day</button></div>
        <label class="gold-confirm"><input type="checkbox" data-gold-role="confirmed" ${goldActionDraft.confirmed ? "checked" : ""}> I confirm every day's exact transportation, restaurants, attraction, and hotel.</label>`;
      renderTravelCostSummary();
    }
    function draftConstraintNumber(keys, fallback) {
      for (const key of keys) {
        const row = constraintDraft.find(item => item.key.trim() === key);
        if (!row) continue;
        const parsed = textToValue(row.valueText);
        const number = Number(String(parsed).replace(/[$,]/g, ""));
        if (Number.isFinite(number)) return number;
      }
      return fallback;
    }
    function extractTravelUnitCost(value) {
      if (value && typeof value === "object") {
        for (const key of ["cost", "price", "Average Cost", "Price"]) {
          const number = Number(String(value[key] ?? "").replace(/[$,]/g, ""));
          if (Number.isFinite(number)) return number;
        }
      }
      const text = valueToText(value ?? "");
      const match = text.match(/(?:average cost|listed price|cost|price)\s*[:=]?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)/i)
        || text.match(/\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)/);
      return match ? Number(match[1].replaceAll(",", "")) : null;
    }
    function extractMaximumOccupancy(value) {
      if (value && typeof value === "object") {
        for (const key of ["maximum occupancy", "maximum_occupancy", "max_occupancy"]) {
          const number = Number(value[key]);
          if (Number.isFinite(number) && number > 0) return number;
        }
      }
      const match = valueToText(value ?? "").match(/maximum occupancy\s*:?\s*([0-9]+)/i);
      return match ? Number(match[1]) : null;
    }
    function travelCostItem(field, value, people) {
      const text = valueToText(value ?? "").trim();
      const label = prettyKey(field);
      if (field === "attraction") {
        return { label, subtotal: 0, formula: "$0.00 (not costed by the benchmark)", unavailable: false };
      }
      if (!text || text === "-") {
        return { label, subtotal: 0, formula: "$0.00 (not selected)", unavailable: false };
      }
      const unit = extractTravelUnitCost(value);
      if (unit === null) {
        return { label, subtotal: 0, formula: "Price unavailable — excluded from total", unavailable: true };
      }
      let multiplier = people;
      let unitLabel = `${people} traveler${people === 1 ? "" : "s"}`;
      const lower = text.toLowerCase();
      if (field === "transportation" && lower.includes("taxi")) {
        multiplier = Math.ceil(people / 4);
        unitLabel = `${multiplier} taxi${multiplier === 1 ? "" : "s"} (4 travelers each)`;
      } else if (field === "transportation" && (lower.includes("self-driving") || lower.includes("self driving"))) {
        multiplier = Math.ceil(people / 5);
        unitLabel = `${multiplier} car${multiplier === 1 ? "" : "s"} (5 travelers each)`;
      } else if (field === "accommodation") {
        const occupancy = extractMaximumOccupancy(value);
        multiplier = occupancy ? Math.ceil(people / occupancy) : 1;
        unitLabel = occupancy
          ? `${multiplier} room${multiplier === 1 ? "" : "s"} (max ${occupancy} each)`
          : "1 room (occupancy unavailable)";
      }
      return { label, subtotal: unit * multiplier, formula: `${money(unit)} × ${unitLabel}`, unavailable: false };
    }
    function renderTravelCostSummary() {
      const container = document.getElementById("travelCostSummary");
      if (!container) return;
      const days = (((goldActionDraft.action_payload || {}).plan || {}).itinerary || []);
      if (!days.length) {
        container.innerHTML = "";
        return;
      }
      const people = Math.max(1, Math.floor(draftConstraintNumber(["people_number", "party_size"], 1)));
      const budget = draftConstraintNumber(["budget", "budget_max"], null);
      const costFields = ["transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation"];
      let total = 0;
      let unavailableCount = 0;
      const daySections = days.map((day, dayIndex) => {
        const items = costFields.map(field => travelCostItem(field, day[field], people));
        const subtotal = items.reduce((sum, item) => sum + item.subtotal, 0);
        total += subtotal;
        unavailableCount += items.filter(item => item.unavailable).length;
        return `<div class="travel-cost-day">
          <strong>${esc(day.day || `Day ${dayIndex + 1}`)}</strong>
          ${items.map(item => `<div class="cost-row ${item.unavailable ? "unavailable" : ""}">
            <span>${esc(item.label)}</span><span class="cost-formula">${esc(item.formula)}</span><strong>${item.unavailable ? "—" : esc(money(item.subtotal))}</strong>
          </div>`).join("")}
          <div class="cost-row cost-subtotal"><span>Day subtotal</span><span></span><strong>${esc(money(subtotal))}</strong></div>
        </div>`;
      }).join("");
      const hasBudget = Number.isFinite(budget);
      const difference = hasBudget ? budget - total : null;
      container.innerHTML = `<section class="travel-cost-summary">
        <div class="travel-cost-head"><div><div class="small-title" style="margin-top:0;">Proposed itinerary cost breakdown</div><strong>${people} traveler${people === 1 ? "" : "s"}</strong></div><strong>${esc(money(total))} total</strong></div>
        ${daySections}
        <div class="travel-cost-total ${hasBudget && difference < 0 ? "over-budget" : ""}">
          <div class="cost-row"><strong>Estimated total</strong><span></span><strong>${esc(money(total))}</strong></div>
          ${hasBudget ? `<div class="cost-row"><span>Budget</span><span>${difference >= 0 ? "Remaining" : "Over budget"}</span><strong>${esc(money(Math.abs(difference)))}</strong></div>` : ""}
          <div class="travel-cost-note">Flights and meals are per traveler; taxis use 4 travelers per vehicle, self-driving uses 5, and lodging uses maximum occupancy. Attractions are not included in the benchmark cost.${unavailableCount ? ` ${unavailableCount} selected item${unavailableCount === 1 ? " has" : "s have"} no extractable price and ${unavailableCount === 1 ? "is" : "are"} excluded.` : ""}</div>
        </div>
      </section>`;
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
        priority: priorityPayload,
        gold_action: goldActionDraft
      };
    }
    function sameValue(left, right) {
      return JSON.stringify(left) === JSON.stringify(right);
    }
    function cloneValue(value) {
      return JSON.parse(JSON.stringify(value));
    }
    function priorityLevel(priority, key) {
      for (const level of ["high", "medium", "low"]) {
        if ((priority[level] || []).includes(key)) return level;
      }
      return "";
    }
    function movePriority(priority, key, level) {
      for (const candidate of ["high", "medium", "low"]) {
        priority[candidate] = (priority[candidate] || []).filter(item => item !== key);
      }
      if (level) priority[level].push(key);
    }
    function propagateConstraintChanges(nextConstraints, nextPriority) {
      const keys = new Set([...Object.keys(originalConstraints), ...Object.keys(nextConstraints)]);
      const currentIndex = activeTurnIndex;
      const laterTurns = state.instances[activeInstanceIndex].turns.slice(currentIndex + 1);
      for (const key of keys) {
        const existedBefore = Object.prototype.hasOwnProperty.call(originalConstraints, key);
        const existsNow = Object.prototype.hasOwnProperty.call(nextConstraints, key);
        const valueChanged = existedBefore && existsNow && !sameValue(originalConstraints[key], nextConstraints[key]);
        const added = !existedBefore && existsNow;
        const removed = existedBefore && !existsNow;
        const oldLevel = priorityLevel(originalPriority, key);
        const newLevel = priorityLevel(nextPriority, key);
        const priorityChanged = oldLevel !== newLevel;

        for (const laterTurn of laterTurns) {
          const gold = laterTurn.gold_current_intention || (laterTurn.gold_current_intention = {});
          const constraints = gold.constraints || (gold.constraints = {});
          const laterHadKey = Object.prototype.hasOwnProperty.call(constraints, key);
          let inherited = false;

          if (added && !laterHadKey) {
            constraints[key] = cloneValue(nextConstraints[key]);
            inherited = true;
          } else if (valueChanged && laterHadKey && sameValue(constraints[key], originalConstraints[key])) {
            constraints[key] = cloneValue(nextConstraints[key]);
            inherited = true;
          } else if (removed && laterHadKey && sameValue(constraints[key], originalConstraints[key])) {
            delete constraints[key];
            inherited = true;
          }

          const laterPriority = normalizePriority(gold.priority, Object.keys(constraints));
          const laterLevel = priorityLevel(laterPriority, key);
          if (existsNow && ((added && inherited) || (priorityChanged && laterLevel === oldLevel))) {
            movePriority(laterPriority, key, newLevel || "medium");
          }
          gold.priority = laterPriority;
        }
      }
    }
    function commitCurrentDraft() {
      const turn = currentTurn();
      const payload = collectEditPayload();
      propagateConstraintChanges(payload.constraints, payload.priority);
      turn.user_utterance = payload.user_utterance;
      turn.gold_current_intention = turn.gold_current_intention || {};
      turn.gold_current_intention.constraints = payload.constraints;
      turn.gold_current_intention.priority = payload.priority;
      turn.gold_action = payload.gold_action;
      originalConstraints = cloneValue(payload.constraints);
      originalPriority = cloneValue(payload.priority);
    }
    async function saveTrajectory(force = false) {
      commitCurrentDraft();
      if (!dirtyInstances.has(activeInstanceIndex) && !force) return true;
      const instIndex = activeInstanceIndex;
      saveEdit.disabled = true;
      saveStatus.textContent = "Saving trajectory...";
      saveStatus.className = "status toolbar-status";
      try {
        const response = await fetch("/api/update_trajectory", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ instance_index: instIndex, turns: state.instances[instIndex].turns })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Trajectory save failed");
        dirtyInstances.delete(instIndex);
        editDirty = false;
        saveStatus.textContent = `Trajectory saved (${data.turn_count} turns)`;
        saveStatus.className = "status toolbar-status ok";
        return true;
      } catch (error) {
        saveStatus.textContent = error.message;
        saveStatus.className = "status toolbar-status error";
        return false;
      } finally {
        saveEdit.disabled = false;
      }
    }
    function renderSelectors() {
      instanceLabel.textContent = `Trajectories (${state.instances.length})`;
      if (state.shard_index) {
        shardIndicator.textContent = `Shard ${state.shard_index} / ${state.shard_count || "?"}`;
        shardIndicator.style.display = "inline-flex";
        if (state.shard_count) {
          shardPicker.style.display = "block";
          switchShardButton.style.display = "inline-block";
          prevShardButton.style.display = "inline-block";
          nextShardButton.style.display = "inline-block";
          shardSelect.innerHTML = Array.from({ length: state.shard_count }, (_, index) => {
            const shard = index + 1;
            return `<option value="${shard}">Shard ${shard}</option>`;
          }).join("");
          shardSelect.value = String(state.shard_index);
          prevShardButton.disabled = state.shard_index <= 1;
          nextShardButton.disabled = state.shard_index >= state.shard_count;
        }
      }
      instanceSelect.innerHTML = state.instances.map((inst, index) => {
        const label = inst.instance_id || `trajectory_${index}`;
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
      activeInstanceIndex = instIndex;
      activeTurnIndex = turnIndex;
      const inst = state.instances[instIndex];
      const turn = inst.turns[turnIndex];
      const feedback = turn.env_feedback || {};
      const items = feedback.candidate_items || [];
      const gold = turn.gold_current_intention || {};
      const action = turn.agent_action || {};
      const actionPayload = action.action_payload || {};
      displayedItems = items;

      renderEditor(turn);
      renderGoldAction(turn);
      renderEntityIntentions(turn);
      const originalGoldItem = inst.webshop_gold_item;
      if (state.domain === "webshop" && turnIndex === 0 && originalGoldItem) {
        webshopGoldPanel.style.display = "block";
        webshopGoldItem.innerHTML = renderItem(originalGoldItem, { originalGoldReference: true });
      } else {
        webshopGoldPanel.style.display = "none";
        webshopGoldItem.innerHTML = "";
      }
      meta.innerHTML = [
        pill(state.domain),
        pill(inst.instance_id),
        state.annotation_output ? pill(`save → ${state.annotation_output}`, "good") : "",
        pill(`turn ${turn.turn_id ?? turnIndex}`),
        turn.shift_condition?.type ? pill(turn.shift_condition.type, "warn") : "",
        turn.action_implication ? pill(turn.action_implication) : "",
        turn.stop_reason ? pill(`stop: ${turn.stop_reason}`) : ""
      ].join("");

      const rationales = turn.rationales || [];
      rationale.innerHTML = rationales.length
        ? rationales.map(text => `<div class="rationale">${esc(text)}</div>`).join("")
        : `<div class="rationale">No rationale recorded for this turn.</div>`;
      deleteTurn.disabled = inst.turns.length <= 1;
      deleteTrajectory.disabled = state.instances.length <= 1;

      if (state.domain === "travelplanner") {
        renderTravelPlanner(turn, feedback, actionPayload);
        prevTurn.disabled = turnIndex <= 0;
        nextTurn.disabled = turnIndex >= inst.turns.length - 1;
        return;
      }

      const queryText = firstNonEmpty(
        feedback.gold_search_query,
        gold.gold_search_query,
        actionPayload.query,
        (turn.rollout_search_queries || [])[0]
      );
      resultsTitle.textContent = "Loading 15 candidates...";
      resultCount.textContent = "";
      query.textContent = queryText ? `Search query: ${queryText}` : "";
      results.innerHTML = `<div class="empty">Loading full candidate set...</div>`;
      loadWebshopCandidates(instIndex, turnIndex, items);
      prevTurn.disabled = turnIndex <= 0;
      nextTurn.disabled = turnIndex >= inst.turns.length - 1;
    }
    async function loadWebshopCandidates(instanceIndex, turnIndex, fallbackItems) {
      const requestToken = ++candidateRequestToken;
      try {
        const response = await fetch(`/api/candidates/${instanceIndex}/${turnIndex}`, { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Candidate load failed");
        if (requestToken !== candidateRequestToken || instanceIndex !== activeInstanceIndex || turnIndex !== activeTurnIndex) return;
        displayedItems = data.items || [];
        resultsTitle.textContent = `Candidates (Total results: ${displayedItems.length})`;
        resultCount.textContent = `${displayedItems.length} items`;
        results.innerHTML = displayedItems.length
          ? displayedItems.map(item => renderItem(item)).join("")
          : `<div class="empty">No candidates found for this turn.</div>`;
        renderWebshopGoldAction();
      } catch (error) {
        if (requestToken !== candidateRequestToken) return;
        displayedItems = fallbackItems;
        resultsTitle.textContent = `Candidates (Total results: ${fallbackItems.length})`;
        resultCount.textContent = `${fallbackItems.length} items`;
        results.innerHTML = fallbackItems.length
          ? fallbackItems.map(item => renderItem(item)).join("")
          : `<div class="empty">${esc(error.message)}</div>`;
      }
    }
    function prettyKey(value) {
      return String(value || "").replaceAll("_", " ").replace(/\b\w/g, ch => ch.toUpperCase());
    }
    function travelItemTitle(item, category) {
      return firstNonEmpty(
        item.name, item.Name, item.flight_number, item.FlightNumber,
        item.description, item.Description, `${prettyKey(category)} result`
      );
    }
    function renderTravelCard(item, category) {
      const ignored = new Set(["name", "Name", "flight_number", "FlightNumber", "result_index"]);
      const rows = Object.entries(item || {}).filter(([key, value]) => !ignored.has(key) && value !== null && value !== "");
      return `<article class="travel-card">
        <h4 class="travel-card-title">${esc(travelItemTitle(item, category))}</h4>
        <div class="kv">${rows.map(([key, value]) => `
          <div class="kv-key">${esc(prettyKey(key))}</div>
          <div class="kv-value">${esc(valueToText(value))}</div>
        `).join("")}</div>
      </article>`;
    }
    function renderSearchCategory(category, pages) {
      const pageList = Array.isArray(pages) ? pages : [];
      const total = pageList.reduce((sum, page) => sum + ((page && page.items) || []).length, 0);
      return `<section class="travel-category">
        <div class="travel-category-head">
          <h3 class="travel-category-title">${esc(prettyKey(category))}</h3>
          ${pill(`${total} items`)}
        </div>
        ${pageList.length ? pageList.map(page => {
          const items = (page && page.items) || [];
          const statusClass = page.status === "no_results" ? "bad" : "good";
          return `<div class="search-page">
            <div class="search-page-head">
              ${pill(page.source_action || "Search")} ${pill(page.status || "observed", statusClass)}
              <strong>${esc(page.query || "")}</strong>
              ${page.message ? `<div>${esc(page.message)}</div>` : ""}
            </div>
            ${items.length ? `<div class="travel-items">${items.map(item => renderTravelCard(item, category)).join("")}</div>`
              : `<div class="empty">No results returned for this search page.</div>`}
          </div>`;
        }).join("") : `<div class="empty">No ${esc(category)} search was recorded for this turn.</div>`}
      </section>`;
    }
    function renderEntityIntentions(turn) {
      const gold = turn.gold_current_intention || {};
      const entities = Object.entries(gold.entities || {});
      if (state.domain !== "travelplanner" || entities.length <= 1) {
        entityPanel.style.display = "none";
        entityState.innerHTML = "";
        return;
      }
      entityPanel.style.display = "block";
      const changes = Object.entries(turn.gold_delta || {});
      const entityCards = entities.map(([entityId, entity]) => `<article class="travel-card">
            <h3 class="travel-card-title">${esc(entity.reference || "Traveler")} <span class="details">(${esc(entityId)})</span></h3>
            <pre>${esc(JSON.stringify(entity.constraints || {}, null, 2))}</pre>
          </article>`).join("");
      const changeCards = changes.length
        ? changes.map(([path, change]) => `<article class="travel-card">
            <h3 class="travel-card-title">${esc(path)}</h3>
            <div class="chips">${pill(change.category || change.op || "change", change.category === "entity" ? "warn" : "")}</div>
            <pre>${esc(JSON.stringify({old: change.old, new: change.new, rationale: change.rationale}, null, 2))}</pre>
          </article>`).join("")
        : `<div class="empty">Initial turn: no gold changes.</div>`;
      entityState.innerHTML = `
        <div class="small-title">Travelers and person-specific constraints</div>
        <div class="travel-items">${entityCards}</div>
        <div class="small-title">This turn's gold changes (${changes.length})</div>
        <div class="travel-items">${changeCards}</div>`;
    }
    function renderTravelPlanner(turn, feedback, actionPayload) {
      const searchResults = feedback.search_results || {};
      const order = ["attractions", "accommodations", "restaurants", "transportation", "cities"];
      const categories = [...order.filter(key => key in searchResults), ...Object.keys(searchResults).filter(key => !order.includes(key))];
      const total = categories.reduce((sum, key) => sum + (searchResults[key] || []).reduce((n, page) => n + ((page && page.items) || []).length, 0), 0);
      resultsTitle.textContent = "Travel Search Evidence";
      resultCount.textContent = `${total} items`;
      query.textContent = actionPayload.query ? `Planner query: ${actionPayload.query}` : "";
      results.innerHTML = categories.length
        ? categories.map(category => renderSearchCategory(category, searchResults[category])).join("")
        : `<div class="empty">No structured search_results found in env_feedback for this turn.</div>`;
    }
    function renderItem(item, renderOptions = {}) {
      const originalGoldReference = renderOptions?.originalGoldReference === true;
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
      const options = item.options || {};
      const goldOptions = item.gold_selected_options || {};
      const bullets = item.bullet_points || [];
      const facts = Object.entries(item.product_information || {}).filter(([, value]) => valueToText(value).trim());
      const detailBits = [
        item.brand ? `Brand: ${valueToText(item.brand)}` : "",
        item.color ? `Color: ${valueToText(item.color)}` : "",
        item.average_rating ? `Rating: ${valueToText(item.average_rating)}` : "",
        item.total_reviews ? `Reviews: ${valueToText(item.total_reviews)}` : "",
        item.availability_status ? valueToText(item.availability_status) : "",
        item.seller_name ? `Seller: ${valueToText(item.seller_name)}` : ""
      ].filter(Boolean);
      const selectedAsin = String(((goldActionDraft.action_payload || {}).selected_asin) || "");
      const isGoldSelected = selectedAsin && selectedAsin === String(item.asin || "");
      return `
        <article class="item ${isGoldSelected || originalGoldReference ? "gold-selected" : ""}">
          <div class="image-box"><img src="${esc(imageUrl)}" alt=""></div>
          <div>
            <h3 class="asin">${originalGoldReference ? "Original gold · " : ""}${esc(item.asin || "")}</h3>
            <h4 class="title">${esc(item.title || item.Title || "")}</h4>
            <p class="price">${esc(money(firstNonEmpty(item.price, item.Price)))}</p>
            <div class="details">
              ${rankBits.map(x => esc(x)).join(" | ")}
              ${item.category ? ` | ${esc(item.category)}` : ""}
              ${item.product_category ? `<br>${esc(item.product_category)}` : ""}
              ${detailBits.length ? `<br>${detailBits.map(esc).join(" &middot; ")}` : ""}
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
            ${Object.keys(options).length ? `<div class="option-summary"><strong>Options:</strong> ${Object.entries(options).map(([key, values]) => `${esc(prettyKey(key))}: ${esc(Array.isArray(values) ? values.join(", ") : valueToText(values))}`).join(" · ")}</div>` : ""}
            ${Object.keys(goldOptions).length ? `<div class="option-summary"><strong>Gold options:</strong> ${Object.entries(goldOptions).map(([key, value]) => `${esc(prettyKey(key))}: ${esc(valueToText(value))}`).join(" · ")}</div>` : ""}
            ${(item.description || bullets.length || facts.length) ? `<details class="item-evidence" open>
              <summary>Product evidence</summary>
              ${item.description ? `<div class="small-title">Description</div><div>${esc(item.description)}</div>` : ""}
              ${bullets.length ? `<div class="small-title">Product highlights</div><ul class="bullet-list">${bullets.map(value => `<li>${esc(valueToText(value))}</li>`).join("")}</ul>` : ""}
              ${facts.length ? `<div class="small-title">Product information</div><div class="product-facts">${facts.map(([key, value]) => `<div class="fact-key">${esc(prettyKey(key))}</div><div class="fact-value">${esc(valueToText(value))}</div>`).join("")}</div>` : ""}
            </details>` : ""}
            ${originalGoldReference
              ? `<div class="item-action">${pill("WebShop source gold product", "good")}</div>`
              : `<div class="item-action"><button data-select-gold-asin="${esc(item.asin || "")}" class="${isGoldSelected ? "primary" : ""}">${isGoldSelected ? "Gold product selected" : "Select as gold product"}</button></div>`}
          </div>
        </article>`;
    }

    async function navigateTo(instanceIndex, turnIndex) {
      if (navigationBusy) return;
      navigationBusy = true;
      try {
        instanceSelect.value = String(activeInstanceIndex);
        renderTurnOptions();
        turnSelect.value = String(activeTurnIndex);
        commitCurrentDraft();
        if (instanceIndex !== activeInstanceIndex && dirtyInstances.has(activeInstanceIndex)) {
          if (!(await saveTrajectory())) return;
        }
        instanceSelect.value = String(instanceIndex);
        renderTurnOptions();
        const turns = state.instances[instanceIndex].turns;
        turnSelect.value = String(Math.min(Math.max(0, turnIndex), turns.length - 1));
        render();
      } finally {
        navigationBusy = false;
      }
    }
    instanceSelect.addEventListener("change", async () => {
      const requestedInstance = Number(instanceSelect.value || 0);
      await navigateTo(requestedInstance, 0);
    });
    turnSelect.addEventListener("change", async () => {
      const requestedTurn = Number(turnSelect.value || 0);
      await navigateTo(activeInstanceIndex, requestedTurn);
    });
    prevTurn.addEventListener("click", async () => {
      await navigateTo(activeInstanceIndex, Math.max(0, activeTurnIndex - 1));
    });
    nextTurn.addEventListener("click", async () => {
      const inst = state.instances[activeInstanceIndex];
      await navigateTo(activeInstanceIndex, Math.min(inst.turns.length - 1, activeTurnIndex + 1));
    });
    function makeClientAnnotationTurn(previousTurn) {
      return {
        turn_id: 0,
        user_utterance: "",
        gold_current_intention: JSON.parse(JSON.stringify(previousTurn.gold_current_intention || {})),
        gold_delta: {},
        gold_action: {},
        agent_action: {},
        env_feedback: {},
        rollout_trace: [],
        rationales: []
      };
    }
    addTurn.addEventListener("click", () => {
      commitCurrentDraft();
      const instanceIndex = activeInstanceIndex;
      const afterTurnIndex = activeTurnIndex;
      const turns = state.instances[instanceIndex].turns;
      const insertAt = afterTurnIndex + 1;
      turns.splice(insertAt, 0, makeClientAnnotationTurn(turns[afterTurnIndex]));
      turns.forEach((turn, index) => { turn.turn_id = index; });
      renderTurnOptions();
      turnSelect.value = String(insertAt);
      render();
      markDirty();
    });
    deleteTurn.addEventListener("click", () => {
      const instanceIndex = activeInstanceIndex;
      const turnIndex = activeTurnIndex;
      const turns = state.instances[instanceIndex].turns;
      if (turns.length <= 1) return;
      if (!window.confirm(`Remove Turn ${turnIndex} from this trajectory draft? Click Save Trajectory to persist it.`)) return;
      turns.splice(turnIndex, 1);
      turns.forEach((turn, index) => { turn.turn_id = index; });
      renderTurnOptions();
      turnSelect.value = String(Math.min(turnIndex, turns.length - 1));
      render();
      markDirty();
    });
    deleteTrajectory.addEventListener("click", async () => {
      if (state.instances.length <= 1) return;
      const instanceIndex = activeInstanceIndex;
      const instanceId = state.instances[instanceIndex].instance_id || `#${instanceIndex + 1}`;

      deleteTrajectory.disabled = true;
      saveEdit.disabled = true;
      saveStatus.textContent = "Deleting trajectory...";
      saveStatus.className = "status toolbar-status";
      try {
        const response = await fetch(`/api/trajectories/${instanceIndex}`, { method: "DELETE" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Trajectory deletion failed");

        state.instances.splice(instanceIndex, 1);
        const shiftedDirty = [...dirtyInstances]
          .filter(index => index !== instanceIndex)
          .map(index => index > instanceIndex ? index - 1 : index);
        dirtyInstances.clear();
        shiftedDirty.forEach(index => dirtyInstances.add(index));
        candidateRequestToken += 1;
        activeInstanceIndex = Math.max(0, instanceIndex - 1);
        activeTurnIndex = 0;
        renderSelectors();
        instanceSelect.value = String(activeInstanceIndex);
        renderTurnOptions();
        turnSelect.value = "0";
        render();
        saveStatus.textContent = `Deleted trajectory ${instanceId}`;
        saveStatus.className = "status toolbar-status ok";
      } catch (error) {
        saveStatus.textContent = error.message;
        saveStatus.className = "status toolbar-status error";
      } finally {
        saveEdit.disabled = false;
        deleteTrajectory.disabled = state.instances.length <= 1;
      }
    });
    function updateGoldDraft(target) {
      const role = target.dataset.goldRole;
      if (!role) return;
      if (role === "confirmed") {
        goldActionDraft.confirmed = target.checked;
        setGoldStatus();
        markDirty();
        return;
      }
      const payload = goldActionDraft.action_payload || (goldActionDraft.action_payload = {});
      if (role === "asin") payload.selected_asin = target.value.trim();
      if (role === "option") {
        const options = payload.selected_options || (payload.selected_options = {});
        const key = target.dataset.optionKey;
        if (target.value === "") delete options[key];
        else options[key] = target.value;
      }
      if (role === "travel-field") {
        const plan = payload.plan || (payload.plan = {});
        const days = plan.itinerary || (plan.itinerary = []);
        const day = days[Number(target.dataset.dayIndex)];
        if (day) day[target.dataset.field] = target.value;
        renderTravelCostSummary();
      }
      goldActionDraft.confirmed = false;
      const confirmBox = goldActionEditor.querySelector('[data-gold-role="confirmed"]');
      if (confirmBox) confirmBox.checked = false;
      setGoldStatus();
      markDirty();
    }
    goldActionEditor.addEventListener("input", event => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) updateGoldDraft(target);
    });
    goldActionEditor.addEventListener("change", event => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
      updateGoldDraft(target);
      if (target.dataset.goldRole === "asin") renderWebshopGoldAction();
    });
    goldActionEditor.addEventListener("click", event => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) return;
      const role = target.dataset.goldRole;
      const plan = (goldActionDraft.action_payload || {}).plan || {};
      const days = plan.itinerary || (plan.itinerary = []);
      if (role === "add-day") {
        days.push({ day: `Day ${days.length + 1}`, current_city: "", transportation: "", breakfast: "", lunch: "", dinner: "", attraction: "", accommodation: "" });
        goldActionDraft.confirmed = false;
        renderTravelGoldAction();
      } else if (role === "remove-day") {
        days.splice(Number(target.dataset.dayIndex), 1);
        goldActionDraft.confirmed = false;
        renderTravelGoldAction();
      }
      setGoldStatus();
      if (role === "add-day" || role === "remove-day") markDirty();
    });
    results.addEventListener("click", event => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement) || !target.dataset.selectGoldAsin) return;
      const payload = goldActionDraft.action_payload || (goldActionDraft.action_payload = {});
      if (payload.selected_asin !== target.dataset.selectGoldAsin) payload.selected_options = {};
      payload.selected_asin = target.dataset.selectGoldAsin;
      goldActionDraft.confirmed = false;
      renderWebshopGoldAction();
      results.innerHTML = currentItems().map(item => renderItem(item)).join("");
      setGoldStatus();
      markDirty();
    });
    utterance.addEventListener("input", markDirty);
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
      markDirty();
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
      if (state.domain === "travelplanner") renderTravelCostSummary();
      markDirty();
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
      markDirty();
    });
    document.addEventListener("dragstart", event => {
      const target = event.target;
      const chip = target instanceof Element ? target.closest(".priority-chip") : null;
      if (!(chip instanceof HTMLElement)) return;
      const key = chip.dataset.key;
      if (!key) return;
      draggedKey = key;
      event.dataTransfer.setData("text/plain", key);
      chip.classList.add("dragging");
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
      const key = draggedKey;
      const level = column.dataset.level;
      const validKeys = new Set(constraintDraft.map(row => row.key));
      if (!key || !validKeys.has(key) || !priorityDraft[level]) return;
      removeFromPriority(key);
      priorityDraft[level].push(key);
      renderPriorityBoard();
      markDirty();
    });
    saveEdit.addEventListener("click", () => saveTrajectory(true));
    async function switchToShard(targetShard) {
      const target = Number(targetShard);
      if (!Number.isInteger(target) || target < 1 || target > state.shard_count || target === state.shard_index) return;
      if (!(await saveTrajectory())) return;

      switchShardButton.disabled = true;
      prevShardButton.disabled = true;
      nextShardButton.disabled = true;
      saveStatus.textContent = `Switching to Shard ${target}...`;
      saveStatus.className = "status toolbar-status";
      try {
        const response = await fetch("/api/switch_shard", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ shard: target })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Shard switch failed");

        await new Promise(resolve => setTimeout(resolve, 1800));
        for (let attempt = 0; attempt < 40; attempt += 1) {
          try {
            const statusResponse = await fetch(`/api/shard_status?t=${Date.now()}`, { cache: "no-store" });
            const status = await statusResponse.json();
            if (status.ok && status.shard_index === target) {
              window.location.href = `/?shard=${String(target).padStart(3, "0")}`;
              return;
            }
          } catch (_) {
            // The old server is expected to disappear briefly during the switch.
          }
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        throw new Error(`Shard ${target} did not become ready`);
      } catch (error) {
        saveStatus.textContent = error.message;
        saveStatus.className = "status toolbar-status error";
        switchShardButton.disabled = false;
        prevShardButton.disabled = state.shard_index <= 1;
        nextShardButton.disabled = state.shard_index >= state.shard_count;
      }
    }
    switchShardButton.addEventListener("click", () => switchToShard(shardSelect.value));
    prevShardButton.addEventListener("click", () => switchToShard(state.shard_index - 1));
    nextShardButton.addEventListener("click", () => switchToShard(state.shard_index + 1));
    document.getElementById("reload").addEventListener("click", async () => {
      if (await saveTrajectory()) location.reload();
    });
    window.addEventListener("beforeunload", event => {
      if (!dirtyInstances.size) return;
      event.preventDefault();
      event.returnValue = "";
    });

    const initialInstanceId = new URLSearchParams(window.location.search).get("instance_id");
    const initialInstanceIndex = initialInstanceId
      ? state.instances.findIndex(instance => instance.instance_id === initialInstanceId)
      : 0;
    renderSelectors();
    if (initialInstanceIndex >= 0) {
      instanceSelect.value = String(initialInstanceIndex);
      renderTurnOptions();
      turnSelect.value = "0";
    }
    render();
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with SAVE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            for attempt in range(6):
                try:
                    os.replace(str(tmp_path), str(path))
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


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


def collect_raw_candidate_asins(instances: List[Dict[str, Any]]) -> set[str]:
    asins = collect_asins(instances)
    for instance in instances:
        for turn in instance.get("turns") or []:
            feedback = turn.get("env_feedback") or {}
            rerank_info = feedback.get("rerank_info") or {}
            for item in rerank_info.get("raw_top_candidates") or []:
                asin = str((item or {}).get("asin") or "").strip()
                if asin:
                    asins.add(asin)
    return asins


def collect_webshop_goal_asins(instances: List[Dict[str, Any]]) -> set[str]:
    asins: set[str] = set()
    for instance in instances:
        metadata = (instance.get("world_state") or {}).get("webshop_selection_metadata") or {}
        asin = str(metadata.get("asin") or "").strip()
        if asin:
            asins.add(asin)
    return asins


def attach_webshop_gold_items(
    state: Dict[str, Any],
    catalog_items: Dict[str, Dict[str, Any]],
) -> int:
    attached = 0
    for instance in state.get("instances") or []:
        metadata = (instance.get("world_state") or {}).get("webshop_selection_metadata") or {}
        asin = str(metadata.get("asin") or "").strip()
        if not asin:
            continue
        item = copy.deepcopy(catalog_items.get(asin) or {})
        item.setdefault("asin", asin)
        item.setdefault("title", metadata.get("name") or "Catalog details unavailable")
        item.setdefault("query", metadata.get("query"))
        item.setdefault("product_category", metadata.get("product_category"))
        item.setdefault("attributes", list(metadata.get("attributes") or []))
        item.setdefault("options", {})
        item.setdefault("image_url", "")
        item["gold_selected_options"] = copy.deepcopy(
            metadata.get("options") or metadata.get("goal_options") or {}
        )
        instance["webshop_gold_item"] = item
        attached += 1
    return attached


def _catalog_options(product: Dict[str, Any]) -> Dict[str, List[str]]:
    raw_options = product.get("options") or product.get("customization_options") or {}
    if not isinstance(raw_options, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, raw_values in raw_options.items():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        clean: List[str] = []
        for raw_value in values:
            value = raw_value.get("value") if isinstance(raw_value, dict) else raw_value
            text = str(value or "").strip()
            if text and text not in clean:
                clean.append(text)
        if clean:
            normalized[str(key)] = clean
    return normalized


def _catalog_prices(product: Dict[str, Any]) -> List[float]:
    raw_pricing = product.get("pricing") or product.get("price") or product.get("Price") or []
    values = raw_pricing if isinstance(raw_pricing, list) else [raw_pricing]
    prices: List[float] = []
    for value in values:
        for match in re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", "")):
            price = float(match)
            if price not in prices:
                prices.append(price)
    return prices[:2]


def candidate_item_from_catalog(product: Dict[str, Any]) -> Dict[str, Any]:
    bullet_points = product.get("BulletPoints") or product.get("small_description") or []
    if not isinstance(bullet_points, list):
        bullet_points = [bullet_points]
    images = product.get("images") or []
    if isinstance(images, str):
        images = [images]
    pricing = _catalog_prices(product)
    attributes = product.get("Attributes") or product.get("attributes") or []
    if not isinstance(attributes, list):
        attributes = [attributes]
    product_information = product.get("product_information") or {}
    if not isinstance(product_information, dict):
        product_information = {"details": product_information}
    return {
        "asin": str(product.get("asin") or product.get("ASIN") or "").strip(),
        "title": product.get("Title") or product.get("name") or "",
        "price": pricing[0] if pricing else None,
        "pricing": pricing,
        "query": product.get("query"),
        "category": product.get("category"),
        "product_category": product.get("product_category"),
        "description": str(product.get("Description") or product.get("full_description") or "")[:2000],
        "bullet_points": [str(value)[:750] for value in bullet_points[:10]],
        "attributes": [str(value)[:500] for value in attributes[:20]],
        "options": _catalog_options(product),
        "brand": product.get("brand"),
        "color": product.get("color"),
        "average_rating": product.get("average_rating") or product.get("Rating"),
        "total_reviews": product.get("total_reviews"),
        "availability_status": product.get("availability_status"),
        "seller_name": product.get("seller_name"),
        "list_price": product.get("list_price"),
        "product_information": product_information,
        "image_url": str(images[0]) if images else "",
    }


def load_replay_catalog_items(
    target_asins: set[str],
    cache_path: Path,
    scan_full_catalog: bool = True,
) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    known_missing: set[str] = set()
    cache_is_current = False
    if cache_path.is_file():
        loaded = load_json(cache_path)
        if isinstance(loaded, dict) and loaded.get("_version") == ITEM_CACHE_VERSION:
            cached_items = loaded.get("items") or {}
            if isinstance(cached_items, dict):
                cached = {str(key): value for key, value in cached_items.items() if isinstance(value, dict)}
                known_missing = {str(value) for value in (loaded.get("missing_asins") or [])}
                cache_is_current = True

    found = {asin: copy.deepcopy(cached[asin]) for asin in target_asins if asin in cached}
    remaining = target_asins - set(found) - known_missing
    catalog_paths = [SMALL_CATALOG_PATH]
    if scan_full_catalog:
        catalog_paths.append(FULL_CATALOG_PATH)
    for catalog_path in catalog_paths:
        if not remaining or not catalog_path.is_file():
            continue
        for product in iter_json_array(catalog_path):
            asin = str(product.get("asin") or product.get("ASIN") or "").strip()
            if asin not in remaining:
                continue
            found[asin] = candidate_item_from_catalog(product)
            remaining.remove(asin)
            if not remaining:
                break

    newly_missing = remaining - known_missing
    if not cache_is_current or any(asin not in cached for asin in found) or newly_missing:
        cached.update(found)
        known_missing.update(remaining)
        save_json(
            cache_path,
            {"_version": ITEM_CACHE_VERSION, "items": cached, "missing_asins": sorted(known_missing)},
        )
    return found


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


def _metadata_constraint_key(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", key).strip("_")


def enrich_webshop_constraints_from_metadata(instances: List[Dict[str, Any]]) -> int:
    """Restore authoritative WebShop attributes/options omitted by the old LLM schema."""
    added_count = 0
    removal_ops = {"remove", "delete", "drop"}
    for instance in instances:
        world_state = instance.get("world_state") or {}
        if str(world_state.get("domain") or "").lower() != "webshop":
            continue
        metadata = world_state.get("webshop_selection_metadata") or {}
        if not isinstance(metadata, dict):
            continue

        base_constraints: Dict[str, Any] = {}
        query = str(metadata.get("query") or "").strip()
        if query:
            base_constraints["category"] = query
        price_upper = metadata.get("price_upper")
        if isinstance(price_upper, (int, float)) and float(price_upper) < 1_000_000:
            base_constraints["budget_max"] = float(price_upper)
        for raw_key, value in (metadata.get("options") or {}).items():
            key = _metadata_constraint_key(raw_key)
            if key and value not in (None, ""):
                base_constraints[key] = copy.deepcopy(value)
        attribute_constraints = {
            _metadata_constraint_key(attribute): True
            for attribute in metadata.get("attributes") or []
            if _metadata_constraint_key(attribute)
        }
        base_constraints.update(attribute_constraints)

        active = {key: True for key in base_constraints}
        for turn in instance.get("turns") or []:
            delta = turn.get("gold_delta") or {}
            for key in base_constraints:
                change = delta.get(key) if isinstance(delta, dict) else None
                if isinstance(change, dict):
                    op = str(change.get("op") or "").lower()
                    if op in removal_ops:
                        active[key] = False
                    elif op:
                        active[key] = True

            gold = turn.setdefault("gold_current_intention", {})
            constraints = gold.setdefault("constraints", {})
            existing_value_texts = {
                re.sub(r"\s+", " ", str(value).strip().lower())
                for value in constraints.values()
                if not isinstance(value, (dict, list))
            }
            for key, value in base_constraints.items():
                if not active[key] or key in constraints:
                    continue
                # Preserve human-added generic rows such as constraint_4=machine washable.
                attribute_text = key.replace("_", " ")
                if key in attribute_constraints and attribute_text in existing_value_texts:
                    continue
                constraints[key] = copy.deepcopy(value)
                priority = gold.get("priority")
                if isinstance(priority, dict):
                    for level in ("high", "medium", "low"):
                        priority.setdefault(level, [])
                    if not any(key in priority[level] for level in ("high", "medium", "low")):
                        priority["low"].append(key)
                elif isinstance(priority, list) and key not in priority:
                    priority.append(key)
                added_count += 1
    return added_count


def set_initial_constraints_must_have(instances: List[Dict[str, Any]]) -> int:
    changed_count = 0
    for instance in instances:
        turns = instance.get("turns") or []
        if not turns:
            continue
        gold = turns[0].setdefault("gold_current_intention", {})
        constraints = gold.get("constraints") or {}
        desired = {"high": list(constraints.keys()), "medium": [], "low": []}
        if gold.get("priority") != desired:
            gold["priority"] = desired
            changed_count += 1
    return changed_count


def expanded_candidate_items(
    turn: Dict[str, Any],
    catalog_items: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    feedback = turn.get("env_feedback") or {}
    current_items = feedback.get("candidate_items") or []
    rerank_info = feedback.get("rerank_info") or {}
    raw_items = rerank_info.get("raw_top_candidates") or []
    raw_ranks = {
        str((item or {}).get("asin") or "").strip(): (item or {}).get("original_rank")
        for item in raw_items
        if str((item or {}).get("asin") or "").strip()
    }

    expanded: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for current in current_items:
        asin = str((current or {}).get("asin") or "").strip()
        if not asin or asin in seen:
            continue
        merged = copy.deepcopy(catalog_items.get(asin) or {})
        merged.update(copy.deepcopy(current))
        merged["original_rank"] = merged.get("original_rank") or raw_ranks.get(asin)
        expanded.append(merged)
        seen.add(asin)

    for raw in raw_items:
        asin = str((raw or {}).get("asin") or "").strip()
        if not asin or asin in seen:
            continue
        item = copy.deepcopy(catalog_items.get(asin) or {})
        item["asin"] = asin
        item["original_rank"] = (raw or {}).get("original_rank")
        item.setdefault("title", "Catalog details unavailable")
        item.setdefault("options", {})
        item.setdefault("attributes", [])
        item.setdefault("image_url", "")
        expanded.append(item)
        seen.add(asin)
    return expanded[:15]


def prepare_state(
    instances: List[Dict[str, Any]],
    image_map: Dict[str, str],
    *,
    source_path: Optional[Path] = None,
    annotation_path: Optional[Path] = None,
) -> Dict[str, Any]:
    prepared = []
    for instance in instances:
        instance_copy = dict(instance)
        instance_domain = str((instance.get("world_state") or {}).get("domain") or "webshop").lower()
        turns = []
        for turn in instance.get("turns") or []:
            turn_copy = dict(turn)
            feedback = dict(turn_copy.get("env_feedback") or {})
            if instance_domain == "webshop":
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

    domain = "webshop"
    if prepared:
        world_state = prepared[0].get("world_state") or {}
        domain = str(world_state.get("domain") or domain).strip().lower()

    return {
        "domain": domain,
        "instances": prepared,
        "no_image_url": "/static-webshop/images/no-image-available.png",
        "source_dataset": str(source_path) if source_path else "",
        "annotation_output": str(annotation_path) if annotation_path else "",
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


def renumber_turns(turns: List[Dict[str, Any]]) -> None:
    for index, turn in enumerate(turns):
        turn["turn_id"] = index


def make_annotation_turn(previous_turn: Dict[str, Any]) -> Dict[str, Any]:
    """Create an editable turn without copying stale agent/environment evidence."""
    return {
        "turn_id": 0,
        "user_utterance": "",
        "gold_current_intention": copy.deepcopy(previous_turn.get("gold_current_intention") or {}),
        "gold_delta": {},
        "gold_action": {},
        "agent_action": {},
        "env_feedback": {},
        "rollout_trace": [],
    }


def annotation_turn_for_storage(turn: Dict[str, Any]) -> Dict[str, Any]:
    """Remove replay-only presentation fields before persisting a trajectory."""
    stored = copy.deepcopy(turn)
    stored.pop("rationales", None)
    feedback = stored.get("env_feedback") or {}
    for item in feedback.get("candidate_items") or []:
        if isinstance(item, dict):
            item.pop("image_url", None)
    return stored


def validate_gold_action(gold_action: Any, domain: str) -> Optional[str]:
    if not isinstance(gold_action, dict):
        return "gold_action must be an object"
    if not gold_action.get("confirmed"):
        return None
    payload = gold_action.get("action_payload") or {}
    if not isinstance(payload, dict):
        return "gold_action.action_payload must be an object"
    if domain == "travelplanner":
        itinerary = ((payload.get("plan") or {}).get("itinerary") or [])
        if not isinstance(itinerary, list) or not itinerary:
            return "A confirmed TravelPlanner gold action must contain at least one day"
        required_fields = (
            "day", "current_city", "transportation", "breakfast", "lunch",
            "dinner", "attraction", "accommodation",
        )
        for index, day in enumerate(itinerary):
            if not isinstance(day, dict):
                return f"Gold itinerary day {index + 1} must be an object"
            missing = [field for field in required_fields if str(day.get(field, "")).strip() == ""]
            if missing:
                return f"Gold itinerary day {index + 1} is missing: {', '.join(missing)}"
    elif not str(payload.get("selected_asin") or "").strip():
        return "A confirmed WebShop gold action must select a product ASIN"
    return None


def create_app(
    state: Dict[str, Any],
    instances: List[Dict[str, Any]],
    annotation_path: Path,
    catalog_items: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Flask:
    app = Flask(__name__)
    replay_catalog_items = catalog_items or {}

    @app.before_request
    def lock_annotation_mutation() -> None:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            SAVE_LOCK.acquire()
            request.environ["annotation_mutation_lock"] = True

    @app.teardown_request
    def unlock_annotation_mutation(_error: Optional[BaseException]) -> None:
        if request.environ.pop("annotation_mutation_lock", False):
            SAVE_LOCK.release()

    @app.route("/")
    def index() -> str:
        return render_template_string(HTML, state_json=json.dumps(state, ensure_ascii=False))

    @app.route("/api/state")
    def api_state():
        return jsonify(state)

    @app.route("/api/shard_status")
    def api_shard_status():
        return jsonify(
            {
                "ok": True,
                "shard_index": state.get("shard_index"),
                "shard_count": state.get("shard_count"),
            }
        )

    @app.route("/api/switch_shard", methods=["POST"])
    def api_switch_shard():
        payload = request.get_json(silent=True) or {}
        try:
            target_shard = int(payload.get("shard"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid shard number"}), 400
        shard_count = state.get("shard_count")
        if not shard_count:
            return jsonify({"ok": False, "error": "This replay was not started from a shard"}), 400
        if target_shard < 1 or target_shard > int(shard_count):
            return jsonify({"ok": False, "error": f"Shard must be between 1 and {shard_count}"}), 400
        if target_shard == state.get("shard_index"):
            return jsonify({"ok": True, "shard": target_shard, "already_active": True})

        launcher_path = PROJECT_ROOT / "annotation" / "start_webshop_shard.ps1"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher_path),
                "-Shard",
                str(target_shard),
                "-DelayMilliseconds",
                "1500",
                "-NoBrowser",
            ],
            cwd=str(PROJECT_ROOT),
            creationflags=creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"ok": True, "shard": target_shard})

    @app.route("/api/candidates/<int:instance_index>/<int:turn_index>")
    def api_candidates(instance_index: int, turn_index: int):
        if instance_index < 0 or instance_index >= len(state["instances"]):
            return jsonify({"ok": False, "error": "Instance index out of range"}), 404
        turns = state["instances"][instance_index].get("turns") or []
        if turn_index < 0 or turn_index >= len(turns):
            return jsonify({"ok": False, "error": "Turn index out of range"}), 404
        items = expanded_candidate_items(turns[turn_index], replay_catalog_items)
        return jsonify({"ok": True, "items": items, "count": len(items)})

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
        has_gold_action = "gold_action" in payload
        gold_action = payload.get("gold_action")
        if has_gold_action:
            instance_domain = str((instances[instance_index].get("world_state") or {}).get("domain") or "webshop").lower()
            gold_action_error = validate_gold_action(gold_action, instance_domain)
            if gold_action_error:
                return jsonify({"ok": False, "error": gold_action_error}), 400
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

        if has_gold_action:
            turn["gold_action"] = copy.deepcopy(gold_action)
            state_turn["gold_action"] = copy.deepcopy(gold_action)

        save_json(annotation_path, instances)
        return jsonify({"ok": True, "turn": state_turn})

    @app.route("/api/trajectories/<int:instance_index>", methods=["DELETE"])
    def api_delete_trajectory(instance_index: int):
        if instance_index < 0 or instance_index >= len(instances):
            return jsonify({"ok": False, "error": "Trajectory index out of range"}), 404
        if len(instances) <= 1:
            return jsonify({"ok": False, "error": "The annotation file must keep at least one trajectory"}), 400

        removed_instance = instances.pop(instance_index)
        removed_state_instance = state["instances"].pop(instance_index)
        try:
            save_json(annotation_path, instances)
        except Exception:
            instances.insert(instance_index, removed_instance)
            state["instances"].insert(instance_index, removed_state_instance)
            raise
        return jsonify(
            {
                "ok": True,
                "deleted_instance_id": removed_instance.get("instance_id"),
                "remaining_count": len(instances),
                "next_instance_index": min(instance_index, len(instances) - 1),
            }
        )

    @app.route("/api/turns", methods=["POST"])
    def api_add_turn():
        payload = request.get_json(silent=True) or {}
        try:
            instance_index = int(payload.get("instance_index"))
            after_turn_index = int(payload.get("after_turn_index"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid instance_index or after_turn_index"}), 400

        if instance_index < 0 or instance_index >= len(instances):
            return jsonify({"ok": False, "error": "Instance index out of range"}), 404
        turns = instances[instance_index].setdefault("turns", [])
        state_turns = state["instances"][instance_index].setdefault("turns", [])
        if not turns or after_turn_index < 0 or after_turn_index >= len(turns):
            return jsonify({"ok": False, "error": "Turn index out of range"}), 404

        insert_at = after_turn_index + 1
        new_turn = make_annotation_turn(turns[after_turn_index])
        turns.insert(insert_at, new_turn)
        state_turns.insert(insert_at, copy.deepcopy(new_turn))
        renumber_turns(turns)
        renumber_turns(state_turns)
        save_json(annotation_path, instances)
        return jsonify({"ok": True, "turn_index": insert_at, "turn": state_turns[insert_at]})

    @app.route("/api/update_trajectory", methods=["POST"])
    def api_update_trajectory():
        payload = request.get_json(silent=True) or {}
        try:
            instance_index = int(payload.get("instance_index"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid instance_index"}), 400
        if instance_index < 0 or instance_index >= len(instances):
            return jsonify({"ok": False, "error": "Instance index out of range"}), 404

        submitted_turns = payload.get("turns")
        if not isinstance(submitted_turns, list) or not submitted_turns:
            return jsonify({"ok": False, "error": "A trajectory must contain at least one turn"}), 400
        if not all(isinstance(turn, dict) for turn in submitted_turns):
            return jsonify({"ok": False, "error": "Every trajectory turn must be an object"}), 400

        domain = str((instances[instance_index].get("world_state") or {}).get("domain") or "webshop").lower()
        for turn_index, turn in enumerate(submitted_turns):
            gold_action = turn.get("gold_action")
            if gold_action:
                error = validate_gold_action(gold_action, domain)
                if error:
                    return jsonify({"ok": False, "error": f"Turn {turn_index}: {error}"}), 400

        prepared_turns = copy.deepcopy(submitted_turns)
        stored_turns = [annotation_turn_for_storage(turn) for turn in submitted_turns]
        renumber_turns(prepared_turns)
        renumber_turns(stored_turns)
        for turn in prepared_turns:
            turn["rationales"] = rationales_for_turn(turn)

        old_stored_turns = instances[instance_index].get("turns") or []
        old_prepared_turns = state["instances"][instance_index].get("turns") or []
        instances[instance_index]["turns"] = stored_turns
        state["instances"][instance_index]["turns"] = prepared_turns
        try:
            save_json(annotation_path, instances)
        except Exception:
            instances[instance_index]["turns"] = old_stored_turns
            state["instances"][instance_index]["turns"] = old_prepared_turns
            raise
        return jsonify({"ok": True, "turn_count": len(stored_turns)})

    @app.route("/api/turns/<int:instance_index>/<int:turn_index>", methods=["DELETE"])
    def api_delete_turn(instance_index: int, turn_index: int):
        if instance_index < 0 or instance_index >= len(instances):
            return jsonify({"ok": False, "error": "Instance index out of range"}), 404
        turns = instances[instance_index].get("turns") or []
        state_turns = state["instances"][instance_index].get("turns") or []
        if turn_index < 0 or turn_index >= len(turns):
            return jsonify({"ok": False, "error": "Turn index out of range"}), 404
        if len(turns) <= 1:
            return jsonify({"ok": False, "error": "An instance must keep at least one turn"}), 400

        del turns[turn_index]
        del state_turns[turn_index]
        renumber_turns(turns)
        renumber_turns(state_turns)
        save_json(annotation_path, instances)
        return jsonify({"ok": True, "next_turn_index": min(turn_index, len(turns) - 1)})

    @app.route("/static-webshop/images/<path:filename>")
    def webshop_image(filename: str):
        return send_from_directory(NO_IMAGE_PATH, filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay WebShop or TravelPlanner simulated data for annotation review.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Writable annotation JSON. Defaults to "
            "annotation/data/<dataset_stem>_annotated.json. "
            "If it already exists, annotation resumes from that file."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--item_cache", type=Path, default=DEFAULT_ITEM_CACHE_PATH)
    parser.add_argument(
        "--skip_full_catalog",
        action="store_true",
        help="Only use cached images and the 1k catalog. Startup is faster, but many images may be missing.",
    )
    parser.add_argument(
        "--skip_constraint_enrichment",
        action="store_true",
        help="Render constraints exactly as stored in the dataset without restoring selection-metadata attributes.",
    )
    return parser.parse_args()


def default_annotation_path(dataset_path: Path) -> Path:
    return ANNOTATION_DATA_DIR / f"{dataset_path.stem}_annotated{dataset_path.suffix}"


def annotation_input_path(dataset_path: Path, annotation_path: Path) -> Path:
    if dataset_path.resolve() == annotation_path.resolve():
        raise ValueError("--output must differ from --dataset so the source rollout remains unchanged")
    return annotation_path if annotation_path.is_file() else dataset_path


def infer_shard_context(dataset_path: Path) -> Tuple[Optional[int], Optional[int]]:
    match = re.fullmatch(r"shard_(\d+)", dataset_path.stem)
    if not match:
        return None, None
    shard_index = int(match.group(1))
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return shard_index, None
    manifest = load_json(manifest_path)
    shard_count = manifest.get("shard_count") if isinstance(manifest, dict) else None
    return shard_index, int(shard_count) if shard_count is not None else None


def main() -> None:
    args = parse_args()
    annotation_path = args.output or default_annotation_path(args.dataset)
    input_path = annotation_input_path(args.dataset, annotation_path)
    instances = load_json(input_path)
    if not isinstance(instances, list):
        raise ValueError(f"Expected dataset JSON list in {input_path}, got {type(instances).__name__}")
    enriched_constraints = 0
    if not args.skip_constraint_enrichment:
        enriched_constraints = enrich_webshop_constraints_from_metadata(instances)
    asins = collect_asins(instances)
    raw_candidate_asins = collect_raw_candidate_asins(instances)
    raw_candidate_asins.update(collect_webshop_goal_asins(instances))
    catalog_items = load_replay_catalog_items(
        raw_candidate_asins,
        cache_path=args.item_cache,
        scan_full_catalog=not args.skip_full_catalog,
    )
    image_map = load_catalog_images(
        asins,
        cache_path=args.cache,
        scan_full_catalog=False,
    )
    image_map.update(
        {
            asin: str(item.get("image_url") or "")
            for asin, item in catalog_items.items()
            if item.get("image_url")
        }
    )
    state = prepare_state(
        instances,
        image_map,
        source_path=args.dataset,
        annotation_path=annotation_path,
    )
    attached_gold_items = attach_webshop_gold_items(state, catalog_items)
    shard_index, shard_count = infer_shard_context(args.dataset)
    state["shard_index"] = shard_index
    state["shard_count"] = shard_count
    app = create_app(state, instances, annotation_path, catalog_items=catalog_items)
    print(f"Source dataset (read-only): {args.dataset.resolve()}")
    print(f"Annotation output: {annotation_path.resolve()}")
    if enriched_constraints:
        print(f"Restored {enriched_constraints} WebShop constraints from selection metadata.")
    if attached_gold_items:
        print(f"Attached {attached_gold_items} original WebShop gold products.")
    if input_path == annotation_path:
        print("Resuming from existing annotation output.")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
