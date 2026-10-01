// Storage backends with one async API.
//   LocalStore      keeps everything in this browser's localStorage (local testing only).
//   FirestoreStore  the deployed backend; access control lives in ../../firestore.rules.

const FIREBASE_SDK = "https://www.gstatic.com/firebasejs/11.10.0";
const LOCAL_DB_KEY = "evalsite:local-db:v1";
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

const clone = value => (value == null ? value : JSON.parse(JSON.stringify(value)));
const nowIso = () => new Date().toISOString();

// Local test mode reads the admin key from the git-ignored dotfile written by set_admin_key.py.
// Firebase Hosting never deploys dotfiles ("**/.*" in firebase.json); deployed, the Firestore
// rules check the key instead.
async function localAdminKey() {
  try {
    const response = await fetch("config/.local-admin.json", { cache: "no-store" });
    return response.ok ? (await response.json()).key : null;
  } catch {
    return null;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.append(script);
  });
}

// Firebase Hosting (and `firebase serve`) serves /__/firebase/init.js with the project config.
async function firebaseConfigured(site) {
  if (site.firebaseConfig) return true;
  try {
    const response = await fetch("/__/firebase/init.js", { method: "HEAD", cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

async function initFirebase(site) {
  for (const part of ["app", "auth", "firestore"]) await loadScript(`${FIREBASE_SDK}/firebase-${part}-compat.js`);
  if (site.firebaseConfig) window.firebase.initializeApp(site.firebaseConfig);
  else await loadScript("/__/firebase/init.js");
  if (!window.firebase.apps.length) throw new Error("Firebase did not initialize.");
  return window.firebase;
}

export async function createStore(site) {
  const localHost = LOCAL_HOSTS.has(location.hostname);
  const wantsLocal = site.backend === "local" || (site.backend === "auto" && localHost && !(await firebaseConfigured(site)));
  if (wantsLocal) {
    if (!localHost) {
      throw new Error('backend is "local" on a public host: answers would never leave the browser. Use "firebase".');
    }
    return new LocalStore(site);
  }
  const store = new FirestoreStore(await initFirebase(site));
  await store.init();
  return store;
}

// Firestore timestamps become ISO strings so both backends return the same shapes.
function normalizeDoc(data) {
  const out = {};
  for (const [key, value] of Object.entries(data || {})) {
    out[key] = value && typeof value.toDate === "function" ? value.toDate().toISOString() : value;
  }
  return out;
}

class FirestoreStore {
  constructor(firebase) {
    this.mode = "firebase";
    this.firebase = firebase;
    this.auth = firebase.auth();
    this.db = firebase.firestore();
  }

  async init() {
    await new Promise(resolve => {
      const stop = this.auth.onAuthStateChanged(() => {
        stop();
        resolve();
      });
    });
    if (!this.auth.currentUser) await this.auth.signInAnonymously();
    this.uid = this.auth.currentUser.uid;
  }

  doc(collection, id) {
    return this.db.collection(collection).doc(id);
  }

  async all(collection) {
    const snapshot = await this.db.collection(collection).get();
    return snapshot.docs.map(doc => normalizeDoc(doc.data()));
  }

  serverTime() {
    return this.firebase.firestore.FieldValue.serverTimestamp();
  }

  async claimEvaluator(code) {
    const ref = this.doc("eval_evaluators", code);
    const snapshot = await ref.get();
    if (!snapshot.exists) return { error: "not_found" };
    const evaluator = normalizeDoc(snapshot.data());
    if (evaluator.disabled) return { error: "disabled" };
    const patch = { uid: this.uid, last_seen_at: nowIso() };
    if (evaluator.uid !== this.uid) patch.claimed_at = patch.last_seen_at;
    await ref.update(patch);
    return { evaluator: { ...evaluator, ...patch } };
  }

  async isAdmin() {
    return (await this.doc("eval_admins", this.uid).get()).exists;
  }

  async claimAdmin(key) {
    if (await this.isAdmin()) return true;
    if (!key) return false;
    try {
      await this.doc("eval_admins", this.uid).set({ key, created_at: nowIso() });
      return true;
    } catch (error) {
      if (error.code === "permission-denied") return false;
      throw error;
    }
  }

  async signOut() {
    await this.auth.signOut();
  }

  listEvaluators() {
    return this.all("eval_evaluators");
  }

  async createEvaluators(records) {
    for (let i = 0; i < records.length; i += 10) {
      await Promise.all(records.slice(i, i + 10).map(record => this.doc("eval_evaluators", record.code).set(record)));
    }
  }

  async updateEvaluator(code, patch) {
    await this.doc("eval_evaluators", code).update(patch);
  }

  async getProgress(code) {
    const snapshot = await this.doc("eval_progress", code).get();
    return snapshot.exists ? normalizeDoc(snapshot.data()) : null;
  }

  async saveProgress(code, record) {
    await this.doc("eval_progress", code).set({ ...record, code, uid: this.uid, updated_at: this.serverTime() });
  }

  listProgress() {
    return this.all("eval_progress");
  }

  async getJudgment(code, itemId) {
    const snapshot = await this.doc("eval_judgments", `${code}__${itemId}`).get();
    return snapshot.exists ? normalizeDoc(snapshot.data()) : null;
  }

  async saveJudgment(code, itemId, record) {
    await this.doc("eval_judgments", `${code}__${itemId}`).set({
      ...record,
      code,
      item_id: itemId,
      uid: this.uid,
      updated_at: this.serverTime(),
    });
  }

  listJudgments() {
    return this.all("eval_judgments");
  }
}

class LocalStore {
  constructor(site) {
    this.mode = "local";
    this.site = site;
    this.uid = "local-browser";
  }

  load() {
    const empty = { evaluators: {}, progress: {}, judgments: {}, admins: {} };
    try {
      return { ...empty, ...JSON.parse(localStorage.getItem(LOCAL_DB_KEY) || "{}") };
    } catch {
      return empty;
    }
  }

  save(db) {
    localStorage.setItem(LOCAL_DB_KEY, JSON.stringify(db));
  }

  // Mirrors the Firestore rules so local testing catches the same mistakes.
  ownedEvaluator(db, code) {
    const evaluator = db.evaluators[code];
    if (!evaluator || evaluator.disabled || evaluator.uid !== this.uid) throw new Error(`Code ${code} is not signed in here.`);
    return evaluator;
  }

  async claimEvaluator(code) {
    const db = this.load();
    const evaluator = db.evaluators[code];
    if (!evaluator) return { error: "not_found" };
    if (evaluator.disabled) return { error: "disabled" };
    const now = nowIso();
    if (evaluator.uid !== this.uid) evaluator.claimed_at = now;
    Object.assign(evaluator, { uid: this.uid, last_seen_at: now });
    this.save(db);
    return { evaluator: clone(evaluator) };
  }

  async isAdmin() {
    return Boolean(this.load().admins[this.uid]);
  }

  async claimAdmin(key) {
    const db = this.load();
    if (db.admins[this.uid]) return true;
    if (!key || key !== (await localAdminKey())) return false;
    db.admins[this.uid] = { created_at: nowIso() };
    this.save(db);
    return true;
  }

  async signOut() {
    const db = this.load();
    delete db.admins[this.uid];
    this.save(db);
  }

  async listEvaluators() {
    return Object.values(this.load().evaluators).map(clone);
  }

  async createEvaluators(records) {
    const db = this.load();
    for (const record of records) db.evaluators[record.code] = clone(record);
    this.save(db);
  }

  async updateEvaluator(code, patch) {
    const db = this.load();
    if (!db.evaluators[code]) throw new Error(`Unknown code ${code}.`);
    Object.assign(db.evaluators[code], patch);
    this.save(db);
  }

  async getProgress(code) {
    return clone(this.load().progress[code] || null);
  }

  async saveProgress(code, record) {
    const db = this.load();
    this.ownedEvaluator(db, code);
    db.progress[code] = { ...record, code, uid: this.uid, updated_at: nowIso() };
    this.save(db);
  }

  async listProgress() {
    return Object.values(this.load().progress).map(clone);
  }

  async getJudgment(code, itemId) {
    return clone(this.load().judgments[`${code}__${itemId}`] || null);
  }

  async saveJudgment(code, itemId, record) {
    const db = this.load();
    if (!this.ownedEvaluator(db, code).item_ids.includes(itemId)) throw new Error(`${itemId} is not assigned to ${code}.`);
    db.judgments[`${code}__${itemId}`] = { ...record, code, item_id: itemId, uid: this.uid, updated_at: nowIso() };
    this.save(db);
  }

  async listJudgments() {
    return Object.values(this.load().judgments).map(clone);
  }

  async reset() {
    localStorage.removeItem(LOCAL_DB_KEY);
  }
}
