import { before, describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { eq } from 'drizzle-orm';
import { seed } from '@/lib/db/seed';
import {
  addToCart,
  suggestAlternatives,
  clearCart,
  confirmOrder,
  getCart,
  removeFromCart,
  resolveMenuQuery,
  searchMenu,
} from '@/lib/agent/cart';
import { resolveIntent } from '@/lib/agent/intent';
import { getDb } from '@/lib/db';
import { menuItems, type MenuItem } from '@/lib/db/schema';

// Throwaway in-memory database. Safe to set after the imports because the
// data directory is resolved on first connection, not at module load.
process.env.PGLITE_DATA_DIR = 'memory';

/**
 * The ordering fixture suite.
 *
 * Two properties worth stating, because both were earned the hard way:
 *
 * 1. LLM-free. Every scenario is a property of the cart layer, so the suite
 *    runs in CI in ~2s, costs nothing, and gives an unambiguous answer. When
 *    the fine-tuned model lands, these same scenarios get replayed THROUGH
 *    each backend to compare them -- but a failure here is always our logic.
 *
 * 2. MENU-AGNOSTIC. Nothing below hardcodes an item name or a price. Fixtures
 *    are looked up from the database at run time, so swapping data/menu/*.csv
 *    for a different restaurant does not break a single test. An earlier
 *    version hardcoded "Classic Burger, 599" and 13 tests broke the moment
 *    the menu became real data -- which is exactly the coupling a test suite
 *    should not have to a swappable data file.
 */

let counter = 0;
const newSession = () => `test-${Date.now()}-${counter++}`;

/** Representative items, resolved once from whatever menu is loaded. */
const fixtures: Record<string, MenuItem> = {};

before(async () => {
  await seed();
  const db = await getDb();
  const all = await db.select().from(menuItems);

  const pick = (category: string) => all.find((i) => i.category === category);
  const burger = pick('burgers') ?? all[0];
  const side = pick('sides') ?? all[1];
  const drink = pick('drinks') ?? all[2];
  const dessert = pick('desserts') ?? all[3];

  Object.assign(fixtures, { burger, side, drink, dessert });

  for (const [name, item] of Object.entries(fixtures)) {
    assert.ok(item, `no menu item available for fixture "${name}"`);
  }
});

describe('menu search', () => {
  it('matches an exact name', async () => {
    const [top] = await searchMenu(fixtures.burger.name);
    assert.equal(top.slug, fixtures.burger.slug);
  });

  it('matches by slug', async () => {
    const [top] = await searchMenu(fixtures.dessert.slug);
    assert.equal(top.slug, fixtures.dessert.slug);
  });

  it('resolves the bare ASL gloss DRINK to a drink', async () => {
    const [top] = await searchMenu('drink');
    assert.equal(top.category, 'drinks');
  });

  it('returns nothing for an item we do not sell', async () => {
    assert.deepEqual(await searchMenu('lobster thermidor'), []);
  });

  it('ranks an exact name above a partial match', async () => {
    const results = await searchMenu(fixtures.burger.name);
    assert.equal(results[0].slug, fixtures.burger.slug);
    assert.ok(results[0].score >= (results[1]?.score ?? 0));
  });
});

describe('items we do not sell', () => {
  /**
   * A bare "we do not have that" is a dead end, and dead ends cost far more
   * for someone who just spent real effort signing or typing the request.
   * These assert we always come back with somewhere to go.
   */
  it('always suggests something, even for food we do not sell at all', async () => {
    const suggestions = await suggestAlternatives('sushi platter', 3);
    assert.ok(suggestions.length > 0, 'must never return an empty suggestion list');
    assert.ok(suggestions.length <= 3);
  });

  it('gets from a word we do not stock to the right category', async () => {
    const suggestions = await suggestAlternatives('cheeseburger deluxe', 3);
    assert.ok(
      suggestions.some((s) => s.category === 'burgers'),
      `expected a burger, got ${suggestions.map((s) => s.name).join(', ')}`,
    );
  });

  it('handles a near-miss on an item we do stock', async () => {
    const suggestions = await suggestAlternatives('chicken sandwhich', 3); // misspelled
    assert.ok(
      suggestions.some((s) => s.category === 'chicken'),
      `expected chicken, got ${suggestions.map((s) => s.name).join(', ')}`,
    );
  });

  it('a failed add returns suggestions and leaves the cart untouched', async () => {
    const s = newSession();
    const result = await addToCart(s, 'pepperoni pizza');

    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.equal(result.reason, 'not_found', 'we stock nothing like a pizza');
    if (result.reason !== 'not_found') return;
    assert.ok(result.suggestions.length > 0, 'a refusal must carry alternatives');
    assert.equal((await getCart(s)).itemCount, 0);
  });
});

describe('adding items', () => {
  it('adds one item at the menu price', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.slug);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.name, fixtures.burger.name);
    assert.equal(result.added.unitPriceCents, fixtures.burger.priceCents);
    assert.equal(result.cart.itemCount, 1);
  });

  it('multiplies the line total by quantity', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.slug, 3);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.lineTotalCents, fixtures.burger.priceCents * 3);
  });

  it('applies a modifier price delta', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.slug, 1, ['add bacon']);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.modifiers[0].name, 'Add Bacon');
    assert.equal(result.added.lineTotalCents, fixtures.burger.priceCents + 150);
  });

  it('applies a free modifier without changing the price', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.slug, 1, ['no onion']);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.lineTotalCents, fixtures.burger.priceCents);
    assert.equal(result.added.modifiers.length, 1);
  });

  it('applies several modifiers at once', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.slug, 1, ['no onion', 'extra cheese']);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.modifiers.length, 2);
    assert.equal(result.added.lineTotalCents, fixtures.burger.priceCents + 90);
  });

  it('refuses to guess at an unknown item, and suggests instead', async () => {
    const s = newSession();
    const result = await addToCart(s, 'lobster thermidor');

    assert.equal(result.ok, false);
    assert.equal((await getCart(s)).itemCount, 0, 'nothing should have been added');
  });

  it('treats quantity 0 as 1 rather than adding a free item', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.dessert.slug, 0);

    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.added.qty, 1);
  });
});

describe('changing an order', () => {
  it('removes an item by name', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);
    await addToCart(s, fixtures.side.slug);

    const result = await removeFromCart(s, fixtures.side.name);
    assert.equal(result.ok, true);
    assert.equal(result.cart.itemCount, 1);
    assert.equal(result.cart.lines[0].name, fixtures.burger.name);
  });

  it('reports cleanly when asked to remove something not in the cart', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);

    const result = await removeFromCart(s, 'lobster thermidor');
    assert.equal(result.ok, false);
    assert.equal(result.cart.itemCount, 1, 'the cart must be untouched');
  });

  it('handles a mid-order change of mind', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);
    await addToCart(s, fixtures.drink.slug);
    await removeFromCart(s, fixtures.drink.name);
    await addToCart(s, fixtures.side.slug);

    const cart = await getCart(s);
    assert.deepEqual(
      cart.lines.map((l) => l.name),
      [fixtures.burger.name, fixtures.side.name],
    );
  });

  it('clears the whole order', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);
    await addToCart(s, fixtures.dessert.slug);

    const cart = await clearCart(s);
    assert.equal(cart.itemCount, 0);
    assert.equal(cart.totalCents, 0);
  });
});

describe('totals', () => {
  it('sums lines, applies tax, and totals correctly', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);
    await addToCart(s, fixtures.side.slug);
    await addToCart(s, fixtures.drink.slug);

    const cart = await getCart(s);
    const subtotal =
      fixtures.burger.priceCents + fixtures.side.priceCents + fixtures.drink.priceCents;

    assert.equal(cart.subtotalCents, subtotal);
    assert.equal(cart.taxCents, Math.round(subtotal * 0.09375));
    assert.equal(cart.totalCents, cart.subtotalCents + cart.taxCents);
  });

  it('keeps every total an integer number of cents', async () => {
    const s = newSession();
    await addToCart(s, fixtures.side.slug, 3);
    await addToCart(s, fixtures.burger.slug, 2, ['add bacon']);

    const cart = await getCart(s);
    for (const value of [cart.subtotalCents, cart.taxCents, cart.totalCents]) {
      assert.equal(Number.isInteger(value), true, `${value} must be an integer`);
    }
  });

  it('an empty cart totals zero, not NaN', async () => {
    const cart = await getCart(newSession());
    assert.equal(cart.subtotalCents, 0);
    assert.equal(cart.totalCents, 0);
  });
});

describe('price integrity', () => {
  /**
   * The one test that must never fail. Every price the terminal can quote has
   * to trace back to a menu row -- this is the whole reason the menu lives in
   * Postgres instead of in a prompt.
   */
  it('every line price matches the menu row it came from', async () => {
    const db = await getDb();
    const menu = await db.select().from(menuItems);
    const s = newSession();

    for (const item of menu) {
      const result = await addToCart(s, item.slug);
      assert.equal(result.ok, true, `could not add ${item.slug}`);
      if (!result.ok) continue;
      assert.equal(
        result.added.unitPriceCents,
        item.priceCents,
        `${item.name} was added at ${result.added.unitPriceCents}, menu says ${item.priceCents}`,
      );
    }

    const cart = await getCart(s);
    assert.equal(
      cart.subtotalCents,
      menu.reduce((sum, i) => sum + i.priceCents, 0),
      'cart subtotal must equal the sum of the menu prices',
    );
  });

  it('a price change after adding does not retroactively alter the cart', async () => {
    const s = newSession();
    const original = fixtures.dessert.priceCents;
    await addToCart(s, fixtures.dessert.slug);

    const db = await getDb();
    await db
      .update(menuItems)
      .set({ priceCents: original + 500 })
      .where(eq(menuItems.slug, fixtures.dessert.slug));

    const cart = await getCart(s);
    assert.equal(cart.lines[0].unitPriceCents, original, 'the snapshotted price must hold');

    await db
      .update(menuItems)
      .set({ priceCents: original })
      .where(eq(menuItems.slug, fixtures.dessert.slug));
  });
});

describe('provenance', () => {
  /**
   * Prices in the shipped CSVs are estimates, not observed values. This test
   * does not fail on that -- it makes sure the fact is recorded, so nobody
   * can quote a price in the report without the source being one query away.
   */
  it('records where every price and nutrition figure came from', async () => {
    const db = await getDb();
    const all = await db.select().from(menuItems);

    for (const item of all) {
      assert.ok(item.priceSource.length > 0, `${item.slug} has no price_source`);
      assert.ok(item.nutritionSource.length > 0, `${item.slug} has no nutrition_source`);
    }

    const estimated = all.filter((i) => i.priceSource === 'estimated').length;
    if (estimated > 0) {
      console.log(
        `      note: ${estimated}/${all.length} prices are estimated, not observed ` +
          `(see data/menu/README.md)`,
      );
    }
  });
});

describe('confirmation', () => {
  it('confirms a non-empty order and returns an order number', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug);

    const result = await confirmOrder(s);
    assert.equal(result.ok, true);
    assert.equal(typeof result.orderId, 'number');
    assert.ok(result.cart.totalCents > 0);
  });

  it('refuses to confirm an empty order', async () => {
    const result = await confirmOrder(newSession());
    assert.equal(result.ok, false);
  });

  it('starts a fresh cart after confirmation', async () => {
    const s = newSession();
    await addToCart(s, fixtures.dessert.slug);
    await confirmOrder(s);

    const cart = await getCart(s);
    assert.equal(cart.itemCount, 0, 'a confirmed order must not stay in the cart');
  });
});

describe('session isolation', () => {
  it('keeps two customers carts apart', async () => {
    const a = newSession();
    const b = newSession();

    await addToCart(a, fixtures.burger.slug);
    await addToCart(b, fixtures.dessert.slug);

    assert.equal((await getCart(a)).lines[0].name, fixtures.burger.name);
    assert.equal((await getCart(b)).lines[0].name, fixtures.dessert.name);
    assert.equal((await getCart(a)).itemCount, 1);
  });
});

describe('disambiguation', () => {
  /**
   * The behaviour these cover is the difference between a terminal that serves
   * a signer and one that does not. The classifier knows a small vocabulary, so
   * BURGER is the most specific thing a signer can say -- and the menu has
   * thirteen of them. Picking one silently is not a shortcut, it is the wrong
   * item in a real order.
   *
   * Menu-agnostic, like the rest of this file: the category words are looked up
   * at run time rather than assumed.
   */
  async function wordMatchingSeveral(): Promise<string | null> {
    for (const word of ['burger', 'chicken', 'fries', 'drink', 'salad', 'nugget']) {
      const r = resolveMenuQuery(await searchMenu(word, 40));
      if (r.kind === 'many') return word;
    }
    return null;
  }

  it('a category word offers a choice instead of adding something', async () => {
    const word = await wordMatchingSeveral();
    assert.ok(word, 'this menu has no word matching several items; test is moot');

    const s = newSession();
    const result = await addToCart(s, word!, 2);

    assert.equal(result.ok, false);
    if (result.ok || result.reason !== 'ambiguous') {
      assert.fail(`expected an ambiguous result for "${word}"`);
    }
    assert.ok(result.totalMatches > 1);
    assert.ok(result.options.length > 1, 'a choice needs options');
    assert.equal((await getCart(s)).itemCount, 0, 'nothing may be added while asking');
  });

  it('naming one item exactly still adds it', async () => {
    const s = newSession();
    const result = await addToCart(s, fixtures.burger.name);

    assert.equal(result.ok, true, `"${fixtures.burger.name}" names exactly one item`);
    assert.equal((await getCart(s)).itemCount, 1);
  });

  it('offers a short list, not the whole menu', async () => {
    const word = await wordMatchingSeveral();
    if (!word) return;
    const r = resolveMenuQuery(await searchMenu(word, 40));
    if (r.kind !== 'many') return;
    assert.ok(r.options.length <= 5, `offered ${r.options.length} options`);
    assert.ok(r.totalMatches >= r.options.length, 'the true count must not shrink');
  });

  it('every offered option is a real menu row', async () => {
    const word = await wordMatchingSeveral();
    if (!word) return;
    const r = resolveMenuQuery(await searchMenu(word, 40));
    if (r.kind !== 'many') return;

    const db = await getDb();
    const all = await db.select().from(menuItems);
    const names = new Set(all.map((m) => m.name));
    for (const o of r.options) {
      assert.ok(names.has(o.name), `${o.name} is not on the menu`);
      assert.ok(o.priceCents > 0, `${o.name} has no price`);
    }
  });

  it('a word we do not stock is still not-found, not a choice', async () => {
    assert.equal(resolveMenuQuery(await searchMenu('lobster thermidor', 40)).kind, 'none');
  });
});

describe('deterministic intent', () => {
  /**
   * The ordinary ordering path must not depend on the model. A local 7B was
   * measured answering "two burgers" with "Added two Burgers, $9.98" while
   * calling no tool and adding nothing -- fluent, confident, and false in every
   * particular. These tests pin the cases that must never reach it.
   */

  it('a category word asks instead of guessing, and keeps the quantity', async () => {
    const intent = await resolveIntent('two burgers');
    assert.equal(intent.kind, 'choose');
    if (intent.kind !== 'choose') return;
    assert.equal(intent.qty, 2, 'the quantity must survive the question');
    assert.ok(intent.options.length > 1);
    assert.ok(intent.totalMatches > intent.options.length - 1);
  });

  it('names one item exactly and adds it', async () => {
    const intent = await resolveIntent(fixtures.burger.name);
    assert.equal(intent.kind, 'add');
    if (intent.kind !== 'add') return;
    assert.equal(intent.items.length, 1);
    assert.equal(intent.items[0].item.name, fixtures.burger.name);
    assert.equal(intent.items[0].qty, 1);
  });

  it('ignores trailing punctuation from a tapped choice', async () => {
    // The choice buttons send "2 Big Mac." -- the full stop must not stop it
    // matching the row called "Big Mac", which it previously did.
    const intent = await resolveIntent(`2 ${fixtures.burger.name}.`);
    assert.equal(intent.kind, 'add');
    if (intent.kind !== 'add') return;
    assert.equal(intent.items[0].qty, 2);
    assert.equal(intent.items[0].item.name, fixtures.burger.name);
  });

  it('does not split an item name containing "with"', async () => {
    const db = await getDb();
    const all = await db.select().from(menuItems);
    const withName = all.find((m) => / with /i.test(m.name));
    if (!withName) return; // menu has no such item; nothing to prove

    const intent = await resolveIntent(withName.name);
    assert.equal(intent.kind, 'add', `"${withName.name}" must resolve as one item`);
    if (intent.kind !== 'add') return;
    assert.equal(intent.items.length, 1);
  });

  it('hands removals, questions and chatter to the model', async () => {
    for (const text of [
      'remove the fries',
      'what do you have',
      'how much is that',
      'actually make that one burger',
      'yes',
      'no thanks',
      '',
    ]) {
      assert.equal((await resolveIntent(text)).kind, 'none', `"${text}" must not be auto-handled`);
    }
  });

  it('does not invent an order out of something we do not sell', async () => {
    assert.equal((await resolveIntent('lobster thermidor')).kind, 'none');
  });
});

describe('chatter is not an order', () => {
  /**
   * From a real session. Typing "hi" produced "We have 38 of those. Which one?
   * 4 Piece Chicken McNuggets..." -- because "chicken" contains "hi", so a
   * greeting scored a substring match against every chicken row.
   */
  it('a greeting matches nothing on the menu', async () => {
    for (const greeting of ['hi', 'hey', 'hello']) {
      assert.equal(
        (await searchMenu(greeting, 60)).length,
        0,
        `"${greeting}" must not match menu items`,
      );
    }
  });

  it('a greeting is never treated as an order', async () => {
    for (const greeting of ['hi', 'hello', 'hey there', 'good morning']) {
      assert.equal((await resolveIntent(greeting)).kind, 'none', greeting);
    }
  });

  it('matches whole words, not fragments inside them', async () => {
    // "ice" must not reach "Spice"; "rib" must not reach "Caribbean".
    for (const [q, forbidden] of [['ice', 'spice'], ['rib', 'caribbean'], ['hi', 'chicken']]) {
      const names = (await searchMenu(q, 60)).map((m) => m.name.toLowerCase());
      assert.ok(
        !names.some((n) => n.includes(forbidden) && !n.includes(` ${q}`) && !n.startsWith(q)),
        `"${q}" should not match via "${forbidden}"`,
      );
    }
  });

  it('still finds the things people do mean', async () => {
    assert.ok((await searchMenu('spicy', 60)).length > 0, 'spicy');
    assert.ok((await searchMenu('burger', 60)).length > 1, 'burger');
    assert.ok((await searchMenu(fixtures.burger.name, 60)).length > 0, 'full name');
  });

  it('does not refuse an order because a stop word hides inside a menu word', async () => {
    // "no" lives inside "nuggets"; a substring check refused real orders.
    const intent = await resolveIntent(fixtures.burger.name);
    assert.equal(intent.kind, 'add');
  });
});

describe('reading the order back', () => {
  /**
   * Asked to read the order back, the model answered "...and 1 Hamburger for
   * $5.49. Your total is $9.38" over a cart holding one Hamburger and
   * totalling $6.00. The read-back is where a customer decides to pay, so it
   * is answered from the database and never by the model.
   */
  it('a confirm request is recognised as a read-back', async () => {
    for (const text of [
      "That's everything. Please read back my order and confirm it.",
      'read back my order',
      "that's all",
      'confirm my order',
    ]) {
      assert.equal((await resolveIntent(text)).kind, 'readback', text);
    }
  });

  it('a read-back is not mistaken for an order', async () => {
    const intent = await resolveIntent("that's everything");
    assert.notEqual(intent.kind, 'add');
    assert.notEqual(intent.kind, 'choose');
  });
});

describe('starting over', () => {
  it('is recognised rather than described', async () => {
    for (const text of ['Cancel everything and start over.', 'start over', 'clear my order']) {
      assert.equal((await resolveIntent(text)).kind, 'clear', text);
    }
  });

  it('actually empties the order', async () => {
    const s = newSession();
    await addToCart(s, fixtures.burger.slug, 2);
    assert.equal((await getCart(s)).itemCount, 2);

    await clearCart(s);
    assert.equal((await getCart(s)).itemCount, 0, 'a cleared order must really be empty');
  });
});
