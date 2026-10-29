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
const EFFORT = { chat: env("CHAT_EFFORT", "low"), extract: env("EXTRACT_EFFORT", "medium"), summary: env("SUMMARY_EFFORT", "medium") };

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
        } catch (e) {
          console.error("ai error", e?.status, e?.message);
          const code = e instanceof Anthropic.RateLimitError ? "busy" : e?.message === "refused" ? "refused" : "server";
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
  return msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

// ---------- routes ----------
async function start(body) {
  const code = String(body?.code ?? "").trim().toUpperCase();
  if (!code || code.length > 40) return fail("bad_code", 400);
  const s = store();
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
  if (plan.summaries >= LIMIT.summaries) return fail("summary_limit", 429);
  const facts = body.facts;
  const factsText = JSON.stringify(facts ?? null);
  if (!facts || typeof facts !== "object" || Array.isArray(facts) || factsText.length > 8000) return fail("bad_input", 400);
  plan.summaries++;
  await savePlan(body.planId, plan);
  const prompt = SUMMARY_LINES.concat(["", "ข้อมูล:", factsText]).join("\n");
  return streamText((send) => runStream(aiParams("summary", { max_tokens: 8000, messages: [{ role: "user", content: prompt }] }), send));
}

async function save(body) {
  const plan = await loadPlan(body?.planId);
  if (!plan) return fail("no_plan", 404);
  if (plan.saves >= LIMIT.saves) return fail("save_limit", 429);
  const r = body.record;
  if (!r || typeof r !== "object" || JSON.stringify(r).length > 40000) return fail("bad_input", 400);
  const keep = {};
  for (const k of ["inputs", "defaultsUsed", "summary", "summaryG"]) if (k in r) keep[k] = r[k];
  plan.saves++;
  plan.record = { ...(plan.record || {}), ...keep, updatedAt: new Date().toISOString() };
  await savePlan(body.planId, plan);
  return json({ ok: true });
}

async function view(url) {
  const plan = await loadPlan(url.searchParams.get("id"));
  if (!plan) return fail("no_plan", 404);
  return json({ turns: plan.turns, record: plan.record, created: plan.created });
}

export default async (req) => {
  const url = new URL(req.url);
  const route = url.pathname.replace(/^\/api\//, "");
  try {
    if (req.method === "GET" && route === "plan") return await view(url);
    if (req.method !== "POST") return fail("not_found", 404);
    const body = await req.json().catch(() => null);
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
