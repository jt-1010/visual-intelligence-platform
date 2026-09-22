/**
 * Non-learned ordering baseline.
 *
 * Keyword/gloss rules over the same tools the LLM uses. Kept deliberately
 * dumb: it is the floor the trained/hosted models have to beat, and it always
 * runs (no API key, no Ollama).
 */

import { addToCart, clearCart, removeFromCart, searchMenu } from '@/lib/agent/cart';

const NUMBER_WORDS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
};

/** Map ASL / colloquial tokens to menu search queries. */
const GLOSS_TO_QUERY: Record<string, string> = {
  hamburger: 'hamburger',
  burger: 'hamburger',
  cheeseburger: 'cheeseburger',
  cheese: 'cheeseburger',
  fries: 'fries',
  frenchfries: 'fries',
  fry: 'fries',
  drink: 'drink',
  soda: 'soda',
  water: 'water',
  coffee: 'coffee',
  shake: 'shake',
};

function stripChannel(turn: string): string {
  return turn.replace(/^\[(SIGN|SPEECH|TEXT|TOUCH|PRESENCE)\]\s*/i, '').trim();
}

function tokens(turn: string): string[] {
  return stripChannel(turn)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function qtyFrom(toks: string[]): number {
  for (const t of toks) {
    if (NUMBER_WORDS[t]) return NUMBER_WORDS[t];
    if (/^\d+$/.test(t)) return Math.min(20, Math.max(1, Number(t)));
  }
  return 1;
}

function queriesFrom(toks: string[]): string[] {
  const joined = toks.join(' ');
  // Prefer multi-word / specific phrases first so "cheese burger" is one
  // cheeseburger, not cheeseburger + hamburger.
  const phrases: [string, string][] = [
    ['cheese burger', 'cheeseburger'],
    ['cheeseburger', 'cheeseburger'],
    ['french fries', 'fries'],
    ['frenchfries', 'fries'],
  ];
  for (const [phrase, query] of phrases) {
    if (joined.includes(phrase)) return [query];
  }

  const found: string[] = [];
  for (const t of toks) {
    const q = GLOSS_TO_QUERY[t];
    if (q && !found.includes(q)) found.push(q);
  }
  if (found.length === 0) {
    for (const key of Object.keys(GLOSS_TO_QUERY)) {
      if (joined.includes(key) && !found.includes(GLOSS_TO_QUERY[key])) {
        found.push(GLOSS_TO_QUERY[key]);
      }
    }
  }
  return found;
}

function isRemoval(toks: string[]): boolean {
  return toks.includes('no') || toks.includes('remove') || toks.includes('cancel');
}

/**
 * Apply one user turn with rules. Returns a short status string (no LLM prose).
 */
export async function rulesHandleTurn(sessionId: string, turn: string): Promise<string> {
  const toks = tokens(turn);
  if (toks.length === 0) return 'Empty input.';

  if (toks.includes('pizza') || toks.includes('taco') || toks.includes('hotdog')) {
    const alts = await searchMenu('burger', 2);
    return alts.length
      ? `We do not sell that. Closest: ${alts.map((a) => a.name).join(', ')}.`
      : 'We do not sell that.';
  }

  const queries = queriesFrom(toks);
  if (queries.length === 0) {
    return 'I did not recognise a menu item.';
  }

  if (isRemoval(toks)) {
    const parts: string[] = [];
    for (const q of queries) {
      const result = await removeFromCart(sessionId, q);
      parts.push(result.ok ? `Removed ${q}.` : `Could not remove ${q}.`);
    }
    return parts.join(' ');
  }

  const qty = qtyFrom(toks);
  const parts: string[] = [];
  for (const q of queries) {
    const result = await addToCart(sessionId, q, qty);
    if (!result.ok) {
      parts.push(`No match for ${q}.`);
      continue;
    }
    parts.push(`Added ${result.added.qty} ${result.added.name}.`);
  }
  return parts.join(' ') || 'Nothing added.';
}

export async function runRulesBaseline(
  sessionId: string,
  turns: string[],
): Promise<{ reply: string; ms: number }> {
  await clearCart(sessionId);
  const started = performance.now();
  const replies: string[] = [];
  for (const turn of turns) {
    replies.push(await rulesHandleTurn(sessionId, turn));
  }
  return { reply: replies.join(' '), ms: Math.round(performance.now() - started) };
}
