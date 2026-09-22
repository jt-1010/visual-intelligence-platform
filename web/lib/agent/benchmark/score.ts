import { getCart, searchMenu, type CartView } from '@/lib/agent/cart';
import { getDb } from '@/lib/db';
import { menuItems } from '@/lib/db/schema';
import type { ExpectedLine, Scenario } from './scenarios';

export type CartScore = {
  cartMatch: boolean;
  expected: { slug: string; qty: number }[];
  actual: { slug: string; qty: number }[];
  detail: string;
};

export type PriceScore = {
  ok: boolean;
  invented: string[];
  detail: string;
};

/** Resolve expected queries to concrete menu slugs (best search hit each). */
export async function resolveExpected(
  lines: ExpectedLine[],
): Promise<{ slug: string; qty: number; query: string }[]> {
  const out: { slug: string; qty: number; query: string }[] = [];
  for (const line of lines) {
    const [hit] = await searchMenu(line.query, 1);
    if (!hit) {
      throw new Error(`scenario expectation unresolved: searchMenu("${line.query}") returned nothing`);
    }
    out.push({ slug: hit.slug, qty: line.qty, query: line.query });
  }
  return out;
}

function normalizeCart(cart: CartView): { slug: string; qty: number }[] {
  return cart.lines
    .map((l) => ({ slug: l.slug, qty: l.qty }))
    .sort((a, b) => a.slug.localeCompare(b.slug) || a.qty - b.qty);
}

function bagKey(lines: { slug: string; qty: number }[]): string {
  return lines.map((l) => `${l.slug}x${l.qty}`).join('|');
}

export async function scoreCart(sessionId: string, scenario: Scenario): Promise<CartScore> {
  const cart = await getCart(sessionId);
  const actual = normalizeCart(cart);

  if (scenario.expect.empty) {
    const ok = actual.length === 0;
    return {
      cartMatch: ok,
      expected: [],
      actual,
      detail: ok ? 'cart empty as required' : `expected empty cart, got ${bagKey(actual) || '(empty)'}`,
    };
  }

  const expectedRaw = await resolveExpected(scenario.expect.lines);
  // Merge duplicate slugs (two queries resolving to same item).
  const merged = new Map<string, number>();
  for (const e of expectedRaw) {
    merged.set(e.slug, (merged.get(e.slug) ?? 0) + e.qty);
  }
  const expected = [...merged.entries()]
    .map(([slug, qty]) => ({ slug, qty }))
    .sort((a, b) => a.slug.localeCompare(b.slug) || a.qty - b.qty);

  const cartMatch = bagKey(expected) === bagKey(actual);
  return {
    cartMatch,
    expected,
    actual,
    detail: cartMatch
      ? `match ${bagKey(expected)}`
      : `expected ${bagKey(expected) || '(empty)'}, got ${bagKey(actual) || '(empty)'}`,
  };
}

/**
 * Every $X.XX the model uttered must appear somewhere on the menu (or be a
 * cart total that matches the real cart). Inventing a price is a hard fail.
 */
export async function scorePrices(reply: string, sessionId: string): Promise<PriceScore> {
  const amounts = [...reply.matchAll(/\$\s*(\d+(?:\.\d{1,2})?)/g)].map((m) => m[1]);
  if (amounts.length === 0) {
    return { ok: true, invented: [], detail: 'no dollar amounts in reply' };
  }

  const db = await getDb();
  const menu = await db.select().from(menuItems);
  const allowed = new Set(menu.map((i) => (i.priceCents / 100).toFixed(2)));

  const cart = await getCart(sessionId);
  allowed.add((cart.subtotalCents / 100).toFixed(2));
  allowed.add((cart.taxCents / 100).toFixed(2));
  allowed.add((cart.totalCents / 100).toFixed(2));
  for (const line of cart.lines) {
    allowed.add((line.unitPriceCents / 100).toFixed(2));
    allowed.add((line.lineTotalCents / 100).toFixed(2));
  }

  const invented: string[] = [];
  for (const raw of amounts) {
    const normalized = Number(raw).toFixed(2);
    if (!allowed.has(normalized)) invented.push(`$${normalized}`);
  }

  return {
    ok: invented.length === 0,
    invented,
    detail:
      invented.length === 0
        ? `${amounts.length} price(s) all match menu/cart`
        : `invented prices: ${invented.join(', ')}`,
  };
}
