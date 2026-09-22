/**
 * Ordering scenarios for the agent comparison harness.
 *
 * Expectations are menu-agnostic: `query` is resolved through searchMenu at
 * run time, the same way the cart tests avoid hardcoding item names/prices.
 *
 * Channels match production tags: [SIGN], [SPEECH], [TEXT], [TOUCH].
 */

export type ExpectedLine = {
  /** Free-text query resolved via searchMenu (e.g. "hamburger", "fries"). */
  query: string;
  qty: number;
};

export type Scenario = {
  id: string;
  description: string;
  /** User turns sent to the agent, in order. */
  turns: string[];
  expect: {
    lines: ExpectedLine[];
    /** If true, cart must be empty (e.g. we-do-not-sell + no mistaken add). */
    empty?: boolean;
  };
};

export const SCENARIOS: Scenario[] = [
  {
    id: 'sign-hamburger-one',
    description: 'ASL gloss: want one hamburger',
    turns: ['[SIGN] WANT HAMBURGER'],
    expect: { lines: [{ query: 'hamburger', qty: 1 }] },
  },
  {
    id: 'sign-hamburger-two',
    description: 'ASL gloss: want two hamburgers (topic-comment order)',
    turns: ['[SIGN] WANT I HAMBURGER TWO'],
    expect: { lines: [{ query: 'hamburger', qty: 2 }] },
  },
  {
    id: 'sign-fries',
    description: 'ASL gloss: fries',
    turns: ['[SIGN] FRENCHFRIES'],
    expect: { lines: [{ query: 'fries', qty: 1 }] },
  },
  {
    id: 'sign-combo-burger-fries',
    description: 'ASL gloss: burger and fries in one utterance',
    turns: ['[SIGN] WANT HAMBURGER FRIES'],
    expect: {
      lines: [
        { query: 'hamburger', qty: 1 },
        { query: 'fries', qty: 1 },
      ],
    },
  },
  {
    id: 'text-cheeseburger',
    description: 'Typed exact item name',
    turns: ['[TEXT] I want a cheeseburger'],
    expect: { lines: [{ query: 'cheeseburger', qty: 1 }] },
  },
  {
    id: 'speech-noisy-burger',
    description: 'Noisy STT that still means cheeseburger',
    turns: ['[SPEECH] aisle have a cheese burger'],
    expect: { lines: [{ query: 'cheeseburger', qty: 1 }] },
  },
  {
    id: 'sign-then-modify',
    description: 'Add burger, then remove fries that were never wanted',
    turns: ['[SIGN] WANT HAMBURGER FRIES', '[SIGN] FRIES NO'],
    expect: { lines: [{ query: 'hamburger', qty: 1 }] },
  },
  {
    id: 'we-do-not-sell-pizza',
    description: 'Item we do not sell must not be added',
    turns: ['[TEXT] I want a pizza'],
    expect: { lines: [], empty: true },
  },
  {
    id: 'drink-gloss',
    description: 'Bare ASL gloss DRINK should land a drink',
    turns: ['[SIGN] WANT DRINK'],
    expect: { lines: [{ query: 'drink', qty: 1 }] },
  },
  {
    id: 'quantity-three-text',
    description: 'Explicit quantity in text',
    turns: ['[TEXT] Add three hamburgers please'],
    expect: { lines: [{ query: 'hamburger', qty: 3 }] },
  },
];
