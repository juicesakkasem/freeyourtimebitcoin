// Backend for /plan: access codes, the AI interview, number extraction, the summary, and saved plans.
// The API key and every prompt stay here; the page only sends the conversation and the plan numbers.
import Anthropic from "@anthropic-ai/sdk";
import { getStore } from "@netlify/blobs";
import { RULES, FIRST, EXTRACT_LINES, SUMMARY_LINES } from "../lib/prompts.mjs";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from the Netlify environment

const env = (k, d) => (process.env[k] ?? "").trim() || d;
const MODEL = {
  chat: env("CHAT_MODEL", "claude-opus-5-5"),
  extract: env("EXTRACT_MODEL", "claude-opus-5-5"),
  summary: env("SUMMARY_MODEL", "claude-opus-5-5"),
};
const EFFORT = { chat: env("CHAT_EFFORT", "low"), extract: env("EXTRACT_EFFORT", "medium"), summary: env("SUMMARY_EFFORT", "low") };

// per-plan limits so one code cannot run up the bill
const LIMIT = { msgs: 80, extracts: 6, summaries: 4, saves: 30, turns: 120, userChars: 1200, aiChars: 4000, totalChars: 80000 };

const store = () => getStore({ name: "plan99", consistency: "strong" });
const json = (body, status = 200) => Response.json(body, { status, headers: { "cache-control": "no-store" } });
const fail = (code, status) => json({ error: code }, status);
const clean = (t) => String(t).replace(/\[\[[^\]]*\]\]/g, "").replace(/\[\[[^\]]*$/, "").trim();
const isId = (s) => typeof s === "string" && /^[a-f0-9]{32}$/.test(s);

// server-side fallback on refusals is only offered on these models
const fallbackOk = (m) => /^claude-(opus-5|fable-5|sonnet-5-5)/.test(m);

function aiParams(kind, extra) {
  const p = { model: MODEL[kind], output_config: { effort: EFFORT[kind] }, ...extra };
  if (fallbackOk(p.model)) {
    p.betas = ["server-side-fallback-2026-07-01"];
    p.fallbacks = "default";
  }
  return p;
}

async function loadPlan(id) {
  if (!isId(id)) return null;
  return store().get("plans/" + id, { type: "json" });
}
const savePlan = (id, plan) => store().setJSON("plans/" + id, plan);

function checkTurns(turns) {
  if (!Array.isArray(turns) || turns.length < 3 || turns.length > LIMIT.turns) return false;
  let total = 0;
  for (const t of turns) {
    if (!t || (t.role !== "user" && t.role !== "assistant") || typeof t.content !== "string") return false;
    if (t.content.length > (t.role === "user" ? LIMIT.userChars * 4 : LIMIT.aiChars)) return false;
    total += t.content.length;
  }
  return total <= LIMIT.totalChars && turns[0].role === "user" && turns[turns.length - 1].role === "user";
}

// stream text to the page; a line starting with \u0000ERR: tells the page something went wrong
function streamText(run) {
  const enc = new TextEncoder();
  return new Response(
    new ReadableStream({
      async start(ctrl) {
        try {
          await run((s) => ctrl.enqueue(enc.encode(s)));
          ctrl.enqueue(enc.encode("\u0000END"));
        } catch (e) {
          console.error("ai error", e?.status, e?.message);
          const code = e instanceof Anthropic.RateLimitError ? "busy" : e?.message === "refused" ? "refused" : e?.message === "cut" ? "cut" : "server";
          ctrl.enqueue(enc.encode("\u0000ERR:" + code));
        }
        ctrl.close();
      },
    }),
    { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" } },
  );
}

async function runStream(params, onText) {
  const stream = client.beta.messages.stream(params);
  for await (const ev of stream) {
    if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") onText(ev.delta.text);
  }
  const msg = await stream.finalMessage();
  if (msg.stop_reason === "refusal") throw new Error("refused");
  if (msg.stop_reason === "max_tokens") throw new Error("cut");
  return msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

// ---------- routes ----------
async function start(body) {
  // "JUICE01:5" typed by mistake still means JUICE01 (the number is only set in ACCESS_CODES)
  const code = String(body?.code ?? "").split(":")[0].trim().toUpperCase();
  if (!code || code.length > 40) return fail("bad_code", 400);
  const s = store();
  // a code listed in YEAR_CODES opens a 399 (one-year) account; the same code always opens the same account
  const yearCodes = env("YEAR_CODES", "").split(",").map((x) => x.trim().toUpperCase()).filter(Boolean);
  if (yearCodes.includes(code)) {
    let link = await s.get("yearcodes/" + code, { type: "json" });
    if (!link) {
      link = { accountId: await createAccount("test:" + code) };
      await s.setJSON("yearcodes/" + code, link);
    }
    return json({ accountId: link.accountId });
  }
  let rec = await s.get("codes/" + code, { type: "json" });
  if (!rec) {
    // codes listed in ACCESS_CODES ("CODE:plans,CODE2:plans") are created on first use; payment will add codes directly later
    const seed = env("ACCESS_CODES", "")
      .split(",")
      .map((x) => x.trim().split(":"))
      .find(([c]) => c && c.toUpperCase() === code);
    if (!seed) return fail("bad_code", 403);
    rec = { maxPlans: Math.max(1, parseInt(seed[1], 10) || 1), plansUsed: 0, created: new Date().toISOString() };
  }
  if (rec.plansUsed >= rec.maxPlans) return fail("code_used", 403);
  rec.plansUsed++;
  await s.setJSON("codes/" + code, rec);
  const planId = crypto.randomUUID().replace(/-/g, "");
  await savePlan(planId, { code, created: new Date().toISOString(), msgs: 0, extracts: 0, summaries: 0, saves: 0, turns: null, record: null });
  return json({ planId, plansLeft: rec.maxPlans - rec.plansUsed });
}

async function chat(body) {
  const plan = await loadPlan(body?.planId);
  if (!plan) return fail("no_plan", 404);
  if (plan.record?.inputs || plan.kind === "adjust") return fail("locked", 409);
  if (plan.msgs >= LIMIT.msgs) return fail("msg_limit", 429);
  const turns = body.turns;
  if (!checkTurns(turns) || turns[turns.length - 1].content.length > LIMIT.userChars) return fail("bad_input", 400);
  plan.msgs++;
  await savePlan(body.planId, plan);
  const messages = turns.map((t) => ({ role: t.role, content: t.content }));
  messages[1] = { role: "assistant", content: FIRST }; // the opening line always comes from the server copy
  return streamText(async (send) => {
    const text = await runStream(
      aiParams("chat", { max_tokens: 4000, system: RULES, cache_control: { type: "ephemeral" }, messages }),
      send,
    );
    plan.turns = turns.concat({ role: "assistant", content: text });
    await savePlan(body.planId, plan);
  });
}

async function extract(body) {
  const plan = await loadPlan(body?.planId);
  if (!plan) return fail("no_plan", 404);
  if (plan.record?.inputs || plan.kind === "adjust") return fail("locked", 409);
  if (plan.extracts >= LIMIT.extracts) return fail("extract_limit", 429);
  const turns = body.turns;
  if (!Array.isArray(turns) || !checkTurns(turns.concat({ role: "user", content: "." }))) return fail("bad_input", 400);
  plan.extracts++;
  await savePlan(body.planId, plan);
  const convo = turns.slice(1).map((t) => (t.role === "user" ? "ลูกค้า: " : "ผู้ช่วย: ") + clean(t.content)).join("\n");
  const prompt = EXTRACT_LINES.concat(["", "บทสนทนา:", convo]).join("\n");
  // spaces keep the connection alive while the model works; the page trims them before reading the JSON
  return streamText(async (send) => {
    const tick = setInterval(() => send(" "), 4000);
    try {
      const text = await runStream(aiParams("extract", { max_tokens: 8000, messages: [{ role: "user", content: prompt }] }), () => {});
      const a = text.indexOf("{"), b = text.lastIndexOf("}");
      const obj = a >= 0 && b > a ? JSON.parse(text.slice(a, b + 1)) : null;
      if (!obj || typeof obj !== "object") throw new Error("bad_json");
      send(JSON.stringify(obj));
    } finally {
      clearInterval(tick);
    }
  });
}

async function summary(body) {
  const plan = await loadPlan(body?.planId);
  if (!plan) return fail("no_plan", 404);
  if (plan.record?.summary && plan.record?.summaryOk) return fail("summary_done", 409);
  if (plan.summaries >= LIMIT.summaries) return fail("summary_limit", 429);
  const facts = body.facts;
  const factsText = JSON.stringify(facts ?? null);
  if (!facts || typeof facts !== "object" || Array.isArray(facts) || factsText.length > 8000) return fail("bad_input", 400);
  plan.summaries++;
  await savePlan(body.planId, plan);
  const prompt = SUMMARY_LINES.concat(["", "ข้อมูล:", factsText]).join("\n");
  return streamText((send) => runStream(aiParams("summary", { max_tokens: 16000, messages: [{ role: "user", content: prompt }] }), send));
}

async function save(body) {
  const plan = await loadPlan(body?.planId);
  if (!plan) return fail("no_plan", 404);
  if (plan.saves >= LIMIT.saves) return fail("save_limit", 429);
  const r = body.record;
  if (!r || typeof r !== "object" || JSON.stringify(r).length > 40000) return fail("bad_input", 400);
  // the first save fixes the plan's numbers; the customer may then change them once (edit: true),
  // which clears the summary so the new numbers get their own; after that only a missing summary can be added
  const rec = plan.record || {};
  const keep = {};
  let edited = false;
  if (!rec.inputs) {
    for (const k of ["inputs", "defaultsUsed"]) if (k in r) keep[k] = r[k];
  } else if (r.edit === true && "inputs" in r && (plan.edits || 0) < 1 && !plan.account) {
    keep.inputs = r.inputs;
    keep.defaultsUsed = r.defaultsUsed;
    keep.summary = null;
    keep.summaryG = null;
    keep.summaryOk = false;
    plan.edits = (plan.edits || 0) + 1;
    edited = true;
  }
  if (!rec.summaryOk && !edited) for (const k of ["summary", "summaryG", "summaryOk"]) if (k in r) keep[k] = r[k];
  if (!Object.keys(keep).length) return json({ ok: true, locked: true });
  plan.saves++;
  plan.record = { ...rec, ...keep, updatedAt: new Date().toISOString() };
  await savePlan(body.planId, plan);
  return json({ ok: true });
}

// ---------- waitlist ----------
// one record per email (key = hash of the email, so the same person signing up twice is counted once)
const WL_CHOICES = ["99", "399", "unsure"];
const WL_NEEDS = ["<5", "5-10", "10-20", "20-50", ">50", "unknown"];
// the count is only shown publicly once this many people have signed up
const wlShowFrom = () => Math.max(0, parseInt(env("WAITLIST_SHOW", "30"), 10) || 0);

async function sha(s) {
  const b = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");
}
async function wlCount() {
  const { blobs } = await store().list({ prefix: "waitlist/" });
  return blobs.length;
}
const wlPublic = (n) => (n >= wlShowFrom() ? n : null);

async function waitlistJoin(body) {
  // bots fill the hidden "website" field or submit within a couple of seconds; answer as if it worked
  if (body?.website || !(Number(body?.ms) >= 2500)) return json({ ok: true, n: null });
  const nick = String(body?.nick ?? "").trim();
  const email = String(body?.email ?? "").trim().toLowerCase();
  const choice = WL_CHOICES.includes(body?.choice) ? body.choice : null;
  const need = WL_NEEDS.includes(body?.need) ? body.need : "";
  if (!nick || nick.length > 60 || email.length > 120 || !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email) || !choice || body?.consent !== true)
    return fail("bad_input", 400);
  const s = store();
  const key = "waitlist/" + (await sha(email));
  const old = await s.get(key, { type: "json" });
  if (old) {
    // signing up again only updates the answers; the place in line stays the same
    await s.setJSON(key, { ...old, nick, choice, need: need || old.need, updated: new Date().toISOString() });
    return json({ ok: true, again: true, n: wlPublic(await wlCount()) === null ? null : old.n });
  }
  const n = (await wlCount()) + 1;
  const src = ["plan", "quickcheck", "fb", "tiktok", "line", "youtube", "ig"].includes(body?.src) ? body.src : "other";
  await s.setJSON(key, { n, nick, email, choice, need, src, consent: "ยินยอมให้เก็บข้อมูลและอีเมล เพื่อแจ้งข้อมูลเกี่ยวกับ Freeyourtime MAP เท่านั้น", created: new Date().toISOString() });
  return json({ ok: true, n: wlPublic(n) });
}

async function waitlistCount() {
  return json({ count: wlPublic(await wlCount()) });
}

// the owner downloads the list as a CSV that opens in Excel; needs WAITLIST_KEY (set in Netlify) in the x-admin-key header
async function waitlistExport(req) {
  const want = env("WAITLIST_KEY", "");
  const got = req.headers.get("x-admin-key") || "";
  if (want.length < 8 || (await sha(got)) !== (await sha(want))) return fail("no_access", 403);
  const s = store();
  const { blobs } = await s.list({ prefix: "waitlist/" });
  const rows = (await Promise.all(blobs.map((b) => s.get(b.key, { type: "json" })))).filter(Boolean).sort((a, b) => a.n - b.n);
  const q = (v) => '"' + String(v ?? "").replace(/"/g, '""') + '"';
  const csv = ["ลำดับ,ชื่อเล่น,อีเมล,สนใจแบบ,คิดว่าต้องมีเงิน (ล้านบาท),มาจากหน้า,ลงชื่อเมื่อ"]
    .concat(rows.map((r) => [r.n, r.nick, r.email, r.choice, r.need, r.src, r.created].map(q).join(",")))
    .join("\r\n");
  // the BOM makes Excel read Thai correctly
  return new Response("﻿" + csv, {
    headers: { "content-type": "text/csv; charset=utf-8", "content-disposition": 'attachment; filename="waitlist.csv"', "cache-control": "no-store" },
  });
}

// ---------- 399 accounts ----------
// one year from the day of payment; each month (counted from that day) allows 1 new interview with AI
// and 3 plans adjusted from the last plan's numbers. Unused ones do not carry over.
const QUOTA = { chat: 1, adjust: 3 };
const newId = () => crypto.randomUUID().replace(/-/g, "");
const loadAccount = async (id) => (isId(id) ? store().get("accounts/" + id, { type: "json" }) : null);
const saveAccount = (id, acc) => store().setJSON("accounts/" + id, acc);

function addMonths(d, n) {
  const x = new Date(d);
  const day = x.getUTCDate();
  x.setUTCDate(1);
  x.setUTCMonth(x.getUTCMonth() + n);
  const last = new Date(Date.UTC(x.getUTCFullYear(), x.getUTCMonth() + 1, 0)).getUTCDate();
  x.setUTCDate(Math.min(day, last));
  return x;
}
function period(start, now) {
  let k = (now.getUTCFullYear() - start.getUTCFullYear()) * 12 + (now.getUTCMonth() - start.getUTCMonth());
  if (addMonths(start, k) > now) k--;
  return { from: addMonths(start, k), to: addMonths(start, k + 1) };
}
function usage(acc, now = new Date()) {
  const per = period(new Date(acc.created), now);
  const used = (k) => acc.plans.filter((p) => p.kind === k && new Date(p.created) >= per.from).length;
  return { per, chat: used("chat"), adjust: used("adjust"), active: now < new Date(acc.expires) };
}
// payment will call this too; source says where the account came from (test code or a payment id)
async function createAccount(source, email = null) {
  const id = newId();
  const now = new Date();
  await saveAccount(id, { tier: "year", source, email, created: now.toISOString(), expires: addMonths(now, 12).toISOString(), plans: [] });
  return id;
}

async function accountView(url) {
  const id = url.searchParams.get("a");
  const acc = await loadAccount(id);
  if (!acc) return fail("no_account", 404);
  const u = usage(acc);
  const plans = await Promise.all(
    acc.plans.map(async (p) => {
      const pl = await loadPlan(p.id);
      return { id: p.id, created: p.created, kind: p.kind, done: !!pl?.record?.inputs };
    }),
  );
  return json({
    tier: acc.tier, created: acc.created, expires: acc.expires, active: u.active, periodEnd: u.per.to.toISOString(),
    quota: { chat: { used: u.chat, max: QUOTA.chat }, adjust: { used: u.adjust, max: QUOTA.adjust } },
    plans: plans.reverse(),
  });
}

async function accountNew(body) {
  const id = body?.accountId;
  const acc = await loadAccount(id);
  if (!acc) return fail("no_account", 404);
  const kind = body?.kind === "adjust" ? "adjust" : body?.kind === "chat" ? "chat" : null;
  if (!kind) return fail("bad_input", 400);
  const u = usage(acc);
  if (!u.active) return fail("expired", 403);
  if (u[kind] >= QUOTA[kind]) return fail("quota", 429);
  let seed;
  if (kind === "adjust" && body?.fromPlanId) {
    if (!acc.plans.some((p) => p.id === body.fromPlanId)) return fail("no_base", 409);
    const pl = await loadPlan(body.fromPlanId);
    if (!pl?.record?.inputs) return fail("no_base", 409);
    seed = pl.record.inputs;
  } else if (kind === "adjust") {
    // start from the newest plan that was built
    for (const p of [...acc.plans].reverse()) {
      const pl = await loadPlan(p.id);
      if (pl?.record?.inputs) { seed = pl.record.inputs; break; }
    }
    if (!seed) return fail("no_base", 409);
  }
  const planId = newId();
  const created = new Date().toISOString();
  await savePlan(planId, { code: null, account: id, kind, seed, created, msgs: 0, extracts: 0, summaries: 0, saves: 0, turns: null, record: null });
  acc.plans.push({ id: planId, created, kind });
  await saveAccount(id, acc);
  return json({ planId });
}

// ---------- payment (Stripe Checkout) ----------
// The buyer pays on Stripe's page, comes back to /plan/?paid=<session id>, and the server checks with Stripe
// that it was paid before making the plan (99) or the one-year account (399). The same link always opens the same thing.
const PRODUCTS = {
  "99": { amount: 9900, name: "Freeyourtime MAP · แผนเดียว" },
  "399": { amount: 39900, name: "Freeyourtime MAP · ทั้งปี (ใช้ได้ 1 ปี)" },
};
const stripeKey = () => env("STRIPE_SECRET_KEY", "");
const testMode = () => stripeKey().startsWith("sk_test_");
// SELL lists what the public can buy ("99" or "99,399"); in test mode testers can buy anything
const selling = () => (stripeKey() ? env("SELL", "").split(",").map((x) => x.trim()).filter((x) => PRODUCTS[x]) : []);

async function stripe(method, path, params) {
  const r = await fetch("https://api.stripe.com/v1/" + path, {
    method,
    headers: { authorization: "Bearer " + stripeKey(), "content-type": "application/x-www-form-urlencoded" },
    body: params ? new URLSearchParams(params).toString() : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error("stripe " + r.status + " " + (j?.error?.message || ""));
  return j;
}

async function shop() {
  return json({ open: selling(), test: testMode() });
}

async function buy(body, url) {
  const product = String(body?.product ?? "");
  const item = PRODUCTS[product];
  if (!item || !stripeKey()) return fail("not_selling", 403);
  if (!selling().includes(product) && !testMode()) return fail("not_selling", 403);
  const src = /^[a-z0-9]{1,20}$/.test(body?.src ?? "") ? body.src : "";
  const origin = url.origin;
  const session = await stripe("POST", "checkout/sessions", {
    mode: "payment",
    locale: "th",
    success_url: origin + "/plan/?paid={CHECKOUT_SESSION_ID}",
    cancel_url: origin + "/plan/?cancel=1",
    "line_items[0][quantity]": "1",
    "line_items[0][price_data][currency]": "thb",
    "line_items[0][price_data][unit_amount]": String(item.amount),
    "line_items[0][price_data][product_data][name]": item.name,
    "metadata[product]": product,
    "metadata[src]": src,
  });
  return json({ url: session.url });
}

async function paid(url) {
  const id = url.searchParams.get("s") ?? "";
  if (!/^cs_(test|live)_[A-Za-z0-9]{10,200}$/.test(id)) return fail("bad_session", 400);
  const s = store();
  const done = await s.get("paid/" + id, { type: "json" });
  if (done) return json({ product: done.product, planId: done.planId, accountId: done.accountId, emailed: await sendLink(id, done, url.origin) });
  if (!stripeKey()) return fail("not_selling", 403);
  let session;
  try {
    session = await stripe("GET", "checkout/sessions/" + encodeURIComponent(id));
  } catch (e) {
    if (/stripe 404/.test(e.message)) return fail("bad_session", 404);
    throw e;
  }
  if (!sessionOk(session)) return fail("bad_session", 409);
  // a QR / bank payment may still be on its way: the page asks again in a moment
  if (session.payment_status !== "paid") return json({ pending: true, status: session.status }, 202);
  const rec = await fulfil(session);
  const emailed = await sendLink(session.id, rec, url.origin);
  return json({ product: rec.product, planId: rec.planId, accountId: rec.accountId, emailed });
}

// the session must be one of ours, for the right price
function sessionOk(session) {
  const item = PRODUCTS[session?.metadata?.product];
  return !!item && session.currency === "thb" && session.amount_total === item.amount;
}

// makes the plan (99) or account (399) once per payment, even if the buyer's return and Stripe's webhook arrive together
async function fulfil(session) {
  const s = store();
  const id = session.id;
  let done = await s.get("paid/" + id, { type: "json" });
  if (done) return done;
  const lock = await s.setJSON("paidlock/" + id, { at: Date.now() }, { onlyIfNew: true });
  if (lock && lock.modified === false) {
    for (let i = 0; i < 16; i++) {
      await new Promise((r) => setTimeout(r, 500));
      done = await s.get("paid/" + id, { type: "json" });
      if (done) return done;
    }
    // the other attempt stalled; take over
    const held = await s.get("paidlock/" + id, { type: "json" });
    if (held && Date.now() - held.at < 30000) throw new Error("fulfil busy");
    await s.setJSON("paidlock/" + id, { at: Date.now() });
  }
  const product = session.metadata.product;
  const email = session.customer_details?.email || null;
  const rec = { product, email, amount: session.amount_total, src: session.metadata?.src || "", test: !session.livemode, created: new Date().toISOString() };
  if (product === "399") {
    rec.accountId = await createAccount("stripe:" + id, email);
  } else {
    rec.planId = newId();
    await savePlan(rec.planId, { code: null, paid: id, created: rec.created, msgs: 0, extracts: 0, summaries: 0, saves: 0, turns: null, record: null });
  }
  await s.setJSON("paid/" + id, rec);
  return rec;
}

// ---------- email: the buyer's private link, sent once (Resend) ----------
const escHtml = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
async function sendLink(id, rec, origin) {
  if (rec.emailed) return true;
  const key = env("RESEND_API_KEY", "");
  if (!key || !rec.email) return false;
  const year = rec.product === "399";
  const link = origin + "/plan/?" + (year ? "a=" + rec.accountId : "id=" + rec.planId);
  const subject = year ? "ลิงก์ dashboard Freeyourtime MAP (ทั้งปี) ของคุณ" : "ลิงก์แผน Freeyourtime MAP ของคุณ";
  const what = year ? "dashboard ทั้งปีของคุณ ใช้ได้ 1 ปีนับจากวันที่ชำระ" : "แผนของคุณ";
  const text = [
    "ขอบคุณที่ซื้อ Freeyourtime MAP" + (year ? " แบบทั้งปี" : " แผนเดียว"),
    "",
    "เปิด" + what + " ได้ที่ลิงก์นี้:",
    link,
    "",
    "เก็บอีเมลนี้ไว้ ลิงก์นี้คือกุญแจเข้า" + (year ? "บัญชี" : "แผน") + "ของคุณ ใครมีลิงก์ก็เปิดได้ อย่าส่งต่อ",
    "",
    "เครื่องมือเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน",
    "ติดต่อ: https://www.facebook.com/Juicesakkasemcashualme",
    "เงื่อนไขและนโยบาย: " + origin + "/policy",
  ].join("\n");
  const html = `<div style="font-family:Arial,sans-serif;font-size:16px;line-height:1.7;color:#222;max-width:560px">
<p>ขอบคุณที่ซื้อ <b>Freeyourtime MAP</b>${year ? " แบบทั้งปี" : " แผนเดียว"}</p>
<p>เปิด${escHtml(what)} ได้ที่ปุ่มนี้</p>
<p><a href="${escHtml(link)}" style="display:inline-block;background:#f7931a;color:#111;text-decoration:none;font-weight:bold;padding:12px 22px;border-radius:999px">${year ? "ไปที่ dashboard" : "เปิดแผนของฉัน"}</a></p>
<p style="font-size:14px;color:#555">หรือคัดลอกลิงก์:<br><a href="${escHtml(link)}">${escHtml(link)}</a></p>
<p style="font-size:14px;color:#555"><b>เก็บอีเมลนี้ไว้</b> ลิงก์นี้คือกุญแจเข้า${year ? "บัญชี" : "แผน"}ของคุณ ใครมีลิงก์ก็เปิดได้ อย่าส่งต่อ</p>
<hr style="border:0;border-top:1px solid #ddd">
<p style="font-size:13px;color:#777">เครื่องมือเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน · <a href="https://www.facebook.com/Juicesakkasemcashualme">ติดต่อเรา</a> · <a href="${escHtml(origin)}/policy">เงื่อนไขและนโยบาย</a></p>
</div>`;
  try {
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { authorization: "Bearer " + key, "content-type": "application/json", "idempotency-key": "paid-" + id },
      body: JSON.stringify({ from: env("MAIL_FROM", "Freeyourtime MAP <map@cashualme.com>"), to: [rec.email], subject, html, text }),
    });
    if (!r.ok) { console.error("email failed", r.status, (await r.text()).slice(0, 200)); return false; }
    rec.emailed = new Date().toISOString();
    await store().setJSON("paid/" + id, rec);
    return true;
  } catch (e) {
    console.error("email failed", e?.message);
    return false;
  }
}

// ---------- Stripe webhook: Stripe tells us about a payment even if the buyer closed the page ----------
async function hmacHex(secret, msg) {
  const k = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", k, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
const sameStr = (a, b) => { if (a.length !== b.length) return false; let d = 0; for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i); return d === 0; };

async function stripeWebhook(req, url) {
  const secret = env("STRIPE_WEBHOOK_SECRET", "");
  if (!secret) return fail("not_configured", 503);
  const raw = await req.text();
  const parts = (req.headers.get("stripe-signature") || "").split(",").map((x) => x.trim().split("="));
  const t = parts.find(([k]) => k === "t")?.[1];
  const sigs = parts.filter(([k]) => k === "v1").map(([, v]) => v || "");
  if (!t || !sigs.length || Math.abs(Date.now() / 1000 - Number(t)) > 300) return fail("bad_signature", 400);
  const want = await hmacHex(secret, t + "." + raw);
  if (!sigs.some((v) => sameStr(v, want))) return fail("bad_signature", 400);
  const event = JSON.parse(raw);
  if (event.type === "checkout.session.completed" || event.type === "checkout.session.async_payment_succeeded") {
    const session = event.data?.object;
    if (session?.payment_status === "paid" && sessionOk(session)) {
      const rec = await fulfil(session);
      await sendLink(session.id, rec, url.origin);
    }
  }
  return json({ received: true });
}

async function view(url) {
  const plan = await loadPlan(url.searchParams.get("id"));
  if (!plan) return fail("no_plan", 404);
  return json({ turns: plan.turns, record: plan.record, created: plan.created, edits: plan.account ? 1 : plan.edits || 0, inAccount: !!plan.account, seed: plan.record?.inputs ? undefined : plan.seed });
}

export default async (req) => {
  const url = new URL(req.url);
  const route = url.pathname.replace(/^\/api\//, "");
  try {
    if (req.method === "GET" && route === "plan") return await view(url);
    if (req.method === "GET" && route === "waitlist-count") return await waitlistCount();
    if (req.method === "GET" && route === "account") return await accountView(url);
    if (req.method === "GET" && route === "shop") return await shop();
    if (req.method === "GET" && route === "paid") return await paid(url);
    if (req.method === "GET" && route === "waitlist-export") return await waitlistExport(req);
    if (req.method !== "POST") return fail("not_found", 404);
    if (route === "stripe-webhook") return await stripeWebhook(req, url);
    const body = await req.json().catch(() => null);
    if (route === "waitlist") return await waitlistJoin(body);
    if (route === "account-new") return await accountNew(body);
    if (route === "buy") return await buy(body, url);
    if (route === "start") return await start(body);
    if (route === "chat") return await chat(body);
    if (route === "extract") return await extract(body);
    if (route === "summary") return await summary(body);
    if (route === "save") return await save(body);
    return fail("not_found", 404);
  } catch (e) {
    console.error("api error", route, e?.message);
    return fail("server", 500);
  }
};

export const config = { path: "/api/*" };
