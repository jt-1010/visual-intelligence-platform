import { resolveMenuQuery, searchMenu, type MenuMatch } from './cart';

/**
 * Work out what the person wants WITHOUT asking a language model.
 *
 * This exists because of a measured failure, not a hunch. Asked for "two
 * burgers", the local qwen2.5-7b backend produced:
 *
 *   "Added two Burgers, $9.98. Which burger would you like?
 *    We have Big Mac and Quarter Pounder."
 *
 * It called no tool at all. Nothing was added, no such price exists, and it
 * named two of the thirteen burgers from memory. The cart was never at risk --
 * the database is only ever written by tool calls -- but every word on screen
 * was false, and a terminal that quotes a wrong price is worse than no
 * terminal. Prompting did not fix it: the same request sometimes calls
 * search_menu, sometimes add_to_cart, sometimes nothing.
 *
 * So the ordinary path does not depend on the model. "Two burgers" is not a
 * language problem; it is a lookup against a 71-row table. Doing it here makes
 * the common case exact, instant, and identical every time, and leaves the
 * model the job it is actually good at: conversation that is not a lookup.
 *
 * The bar for taking over is deliberately high. Anything that smells like a
 * question, a correction, or a removal goes to the model, because being wrong
 * here is worse than being slow.
 */

export type ResolvedItem = { item: MenuMatch; qty: number };

export type Intent =
  | { kind: 'none' }
  | { kind: 'readback' }
  | { kind: 'clear' }
  | { kind: 'add'; items: ResolvedItem[] }
  | { kind: 'choose'; query: string; qty: number; options: MenuMatch[]; totalMatches: number };

/**
 * Asking what is in the order, which must be answered from the order.
 *
 * Handed "read back my order", the model replied: "you have 4 piece Sweet N'
 * Spicy Honey BBQ Glazed Tenders for $3.89 and 1 Hamburger for $5.49. Your
 * total is $9.38." The cart held one Hamburger and totalled $6.00. It had
 * invented a line item it never added, then totalled its own invention -- and
 * the read-back is the exact moment a customer decides whether to pay.
 */
const READ_BACK = [
  'read back',
  'read it back',
  "that's everything",
  'that is everything',
  "that's all",
  'that is all',
  "that's it",
  'that is it',
  'my order',
  'my total',
  'confirm my order',
  "i'm done",
  'im done',
];

const NUMBER_WORDS: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5,
  six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
};

/**
 * Words that mean "this is not a plain addition".
 *
 * Without this guard, "remove the fries" scores against every fries row and
 * would be answered with "which fries?" -- turning a deletion into an offer to
 * add one. Anything here is handed to the model untouched.
 */
const START_OVER = [
  'start over',
  'cancel everything',
  'clear my order',
  'clear everything',
  'empty my order',
  'scrap it',
  'start again',
];

const NOT_A_PLAIN_ADD = [
  'remove', 'delete', 'cancel', 'clear', 'start over', 'take off', 'instead',
  'actually', 'change', 'without', 'what', 'which', 'how much',
  'how many', 'do you have', 'confirm', 'done',
  'yes', 'yeah', 'no', 'not', 'nope', 'thanks', 'thank you', 'help',
  // Greetings. Someone saying hello is not ordering anything yet, and the
  // terminal answering "we have 38 of those" to "hi" is the exact failure this
  // list exists to prevent.
  'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
];

/**
 * Word-boundary match, not substring.
 *
 * This list is checked against free text, so plain `includes` misfires in both
 * directions: "no" matched inside "nuggets" and would have refused a real
 * order, while the trailing-space hacks ("no ", "not ") that worked around it
 * failed whenever the word ended the sentence.
 */
function mentions(text: string, phrase: string): boolean {
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(^|[^a-z0-9])${escaped}($|[^a-z0-9])`).test(text);
}

/** Strip a leading quantity. Returns the count and the rest of the phrase. */
function splitQuantity(segment: string): { qty: number; phrase: string } {
  const words = segment.trim().split(/\s+/);
  if (words.length === 0) return { qty: 1, phrase: segment.trim() };

  const first = words[0].toLowerCase();
  const digits = /^\d+$/.test(first) ? Number(first) : undefined;
  const asWord = NUMBER_WORDS[first];
  const qty = digits ?? asWord;

  if (qty !== undefined && qty >= 1 && qty <= 20 && words.length > 1) {
    return { qty, phrase: words.slice(1).join(' ') };
  }
  return { qty: 1, phrase: segment.trim() };
}

/** Singularise just enough for menu lookup: burgers -> burger. */
function singular(phrase: string): string {
  return phrase.replace(/\b(\w{4,})s\b/g, '$1');
}

export async function resolveIntent(rawText: string): Promise<Intent> {
  const text = rawText.trim();
  if (!text) return { kind: 'none' };

  const lower = text.toLowerCase();

  /*
    Starting over is destructive, so it is done rather than described. Asked to
    cancel everything the model answered "Sure thing! Let's start over" and
    called nothing; the previous order was still there, and the next thing added
    joined it. A cheerful false confirmation is the worst outcome here, because
    the person stops checking.
  */
  if (START_OVER.some((p) => lower.includes(p))) return { kind: 'clear' };

  // After START_OVER, because "clear my order" contains "my order" and would
  // otherwise be read back instead of cleared. Before NOT_A_PLAIN_ADD, because
  // "confirm my order" contains "confirm" and the one answer that must come
  // from the database would go to the model.
  if (READ_BACK.some((p) => lower.includes(p))) return { kind: 'readback' };

  // Conversation, corrections and questions belong to the model.
  if (NOT_A_PLAIN_ADD.some((w) => mentions(lower, w))) return { kind: 'none' };
  if (text.includes('?')) return { kind: 'none' };
  if (text.split(/\s+/).length > 12) return { kind: 'none' };

  /*
    Note what is NOT split on: "with". Menu names contain it -- "Quarter
    Pounder with Cheese", "Premium Asian Salad w/ Grilled Chicken" -- and
    treating it as a separator turns one item into two unrecognisable halves.
  */
  const segments = lower
    .split(/\band\b|,|\+/)
    .map((s) => s.replace(/\b(i|we|want|would|like|please|get|have|can|could|the|some)\b/g, ' '))
    .map((s) => s.replace(/\s+/g, ' ').trim())
    .filter(Boolean);

  if (segments.length === 0) return { kind: 'none' };

  const adds: ResolvedItem[] = [];

  for (const segment of segments) {
    const { qty, phrase: raw } = splitQuantity(segment);
    // Trailing punctuation only. "big mac." must match the row named "Big Mac",
    // but the ® in "Quarter Pounder® with Cheese" is part of the name and stays.
    const phrase = raw.replace(/[.!?,;:]+$/, '').trim();
    if (!phrase) return { kind: 'none' };

    let matches = await searchMenu(phrase, 40);
    if (matches.length === 0) matches = await searchMenu(singular(phrase), 40);

    const resolved = resolveMenuQuery(matches);

    // Nothing recognisable: let the model explain, and offer alternatives.
    if (resolved.kind === 'none') return { kind: 'none' };

    // One question at a time -- settle this before looking at the rest.
    if (resolved.kind === 'many') {
      return {
        kind: 'choose',
        query: phrase,
        qty,
        options: resolved.options,
        totalMatches: resolved.totalMatches,
      };
    }

    adds.push({ item: resolved.item, qty });
  }

  return adds.length > 0 ? { kind: 'add', items: adds } : { kind: 'none' };
}
