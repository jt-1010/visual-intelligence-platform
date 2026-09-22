/**
 * Agent comparison benchmark.
 *
 * Same scenarios, three policies:
 *   rules   — keyword/gloss baseline (always runs)
 *   ollama  — local Qwen (stock or fine-tuned)
 *   gateway — hosted frontier baseline
 *
 * Metrics per scenario: cart match, price integrity, latency.
 *
 *   cd web && npm run benchmark:agent
 *   cd web && npm run benchmark:agent -- --backends=rules,ollama
 */

import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { seed } from '@/lib/db/seed';
import { SCENARIOS } from '@/lib/agent/benchmark/scenarios';
import { scoreCart, scorePrices } from '@/lib/agent/benchmark/score';
import { runRulesBaseline } from '@/lib/agent/benchmark/rules';
import { backendAvailable, runLlmScenario } from '@/lib/agent/benchmark/run';
import type { Backend } from '@/lib/agent/model';

process.env.PGLITE_DATA_DIR = 'memory';

/** Load web/.env.local if present (AI_GATEWAY_API_KEY, OLLAMA_*, etc.). */
function loadEnvLocal() {
  const file = path.join(process.cwd(), '.env.local');
  if (!existsSync(file)) return;
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

loadEnvLocal();

type Policy = 'rules' | Backend;

type Row = {
  policy: Policy;
  scenario: string;
  cartMatch: boolean;
  priceOk: boolean;
  ms: number;
  detail: string;
  error?: string;
};

function parseBackends(argv: string[]): Policy[] {
  const flag = argv.find((a) => a.startsWith('--backends='));
  if (!flag) return ['rules', 'ollama', 'gateway'];
  const list = flag
    .slice('--backends='.length)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean) as Policy[];
  return list.length ? list : ['rules'];
}

function pct(n: number, d: number): string {
  if (d === 0) return 'n/a';
  return `${((100 * n) / d).toFixed(0)}%`;
}

async function main() {
  const wanted = parseBackends(process.argv.slice(2));
  console.log('SignOrder agent comparison\n');

  await seed();

  const policies: { id: Policy; skip?: string }[] = [];
  for (const id of wanted) {
    if (id === 'rules') {
      policies.push({ id });
      continue;
    }
    const avail = await backendAvailable(id);
    if (!avail.ok) {
      policies.push({ id, skip: avail.reason });
      console.log(`skip ${id}: ${avail.reason}`);
    } else {
      policies.push({ id });
    }
  }
  console.log('');

  const rows: Row[] = [];

  for (const policy of policies) {
    if (policy.skip) {
      for (const s of SCENARIOS) {
        rows.push({
          policy: policy.id,
          scenario: s.id,
          cartMatch: false,
          priceOk: false,
          ms: 0,
          detail: 'skipped',
          error: policy.skip,
        });
      }
      continue;
    }

    console.log(`── ${policy.id} (${SCENARIOS.length} scenarios)`);
    for (const scenario of SCENARIOS) {
      const sessionId = `bench-${policy.id}-${scenario.id}-${Date.now()}`;
      let reply = '';
      let ms = 0;
      let error: string | undefined;

      if (policy.id === 'rules') {
        const result = await runRulesBaseline(sessionId, scenario.turns);
        reply = result.reply;
        ms = result.ms;
      } else {
        const result = await runLlmScenario(policy.id, sessionId, scenario.turns);
        reply = result.reply;
        ms = result.ms;
        error = result.error;
      }

      if (error) {
        rows.push({
          policy: policy.id,
          scenario: scenario.id,
          cartMatch: false,
          priceOk: false,
          ms,
          detail: error.slice(0, 120),
          error,
        });
        console.log(`  ✗ ${scenario.id}  ERROR ${ms}ms  ${error.slice(0, 80)}`);
        continue;
      }

      const cart = await scoreCart(sessionId, scenario);
      const prices = await scorePrices(reply, sessionId);
      rows.push({
        policy: policy.id,
        scenario: scenario.id,
        cartMatch: cart.cartMatch,
        priceOk: prices.ok,
        ms,
        detail: cart.detail,
      });

      const mark = cart.cartMatch && prices.ok ? '✓' : '✗';
      console.log(
        `  ${mark} ${scenario.id}  cart=${cart.cartMatch ? 'ok' : 'FAIL'}  ` +
          `price=${prices.ok ? 'ok' : 'FAIL'}  ${ms}ms  ${cart.detail}`,
      );
    }
    console.log('');
  }

  // Summary table
  console.log('══════════════════════════════════════════════════════════');
  console.log('Summary (cart match rate · price integrity · median ms)');
  console.log('══════════════════════════════════════════════════════════');

  for (const id of wanted) {
    const mine = rows.filter((r) => r.policy === id && !r.error?.includes('skipped') && r.detail !== 'skipped');
    const skipped = rows.filter((r) => r.policy === id && r.detail === 'skipped');
    if (skipped.length && mine.length === 0) {
      console.log(`${id.padEnd(10)}  SKIPPED — ${skipped[0].error}`);
      continue;
    }
    const cartHits = mine.filter((r) => r.cartMatch).length;
    const priceHits = mine.filter((r) => r.priceOk).length;
    const times = mine.map((r) => r.ms).sort((a, b) => a - b);
    const median = times.length ? times[Math.floor(times.length / 2)] : 0;
    const both = mine.filter((r) => r.cartMatch && r.priceOk).length;
    console.log(
      `${id.padEnd(10)}  cart ${pct(cartHits, mine.length).padStart(4)}  ` +
        `price ${pct(priceHits, mine.length).padStart(4)}  ` +
        `pass ${pct(both, mine.length).padStart(4)}  ` +
        `p50 ${String(median).padStart(5)}ms  (n=${mine.length})`,
    );
  }

  console.log('\nPass = cart match AND no invented prices.');
  console.log('Re-run with: npm run benchmark:agent -- --backends=rules,ollama,gateway');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
