import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { buildTools } from '../lib/agent/tools';
import { resolveIntent } from '../lib/agent/intent';
import { getCart } from '../lib/agent/cart';

describe('agent tool contracts & execution', () => {
  it('search_menu returns schema-conforming items with preformatted prices', async () => {
    const sessionId = randomUUID();
    const tools = buildTools(sessionId);

    const res = await (tools.search_menu as any).execute({ query: 'burger' });

    assert.equal(typeof res.count, 'number');
    assert.ok(res.count > 0, 'Expected matches for burger');
    assert.ok(Array.isArray(res.items));

    const first = res.items[0];
    assert.equal(typeof first.slug, 'string');
    assert.equal(typeof first.name, 'string');
    assert.equal(typeof first.price, 'string');
    assert.ok(first.price.startsWith('$'), 'Price string must start with $');
    assert.equal(typeof first.priceCents, 'number');
  });

  it('add_to_cart enforces ambiguity guards for broad categories', async () => {
    const sessionId = randomUUID();
    const tools = buildTools(sessionId);

    // "drink" matches multiple beverages in the DB
    const res = await (tools.add_to_cart as any).execute({
      item: 'drink',
      quantity: 1,
      modifiers: [],
    });

    assert.equal(res.added, false);
    assert.equal(res.needsChoice, true);
    assert.ok(Array.isArray(res.options));
    assert.ok(res.options.length > 1);

    // Verify cart was not modified on ambiguous queries
    const cart = await getCart(sessionId);
    assert.equal(cart.lines.length, 0);
  });

  it('add_to_cart correctly adds specific items with modifiers', async () => {
    const sessionId = randomUUID();
    const tools = buildTools(sessionId);

    const res = await (tools.add_to_cart as any).execute({
      item: 'Big Mac',
      quantity: 2,
      modifiers: ['no onion'],
    });

    assert.equal(res.added, true);
    assert.equal(res.item.quantity, 2);
    assert.ok(res.item.name.toLowerCase().includes('big mac'));
    assert.ok(Array.isArray(res.item.modifiers));
    assert.equal(res.item.modifiers[0], 'No Onion');

    // Confirm stored in cart
    const cart = await getCart(sessionId);
    assert.equal(cart.lines.length, 1);
    assert.equal(cart.lines[0].qty, 2);
  });

  it('confirm_order marks non-empty carts as confirmed', async () => {
    const sessionId = randomUUID();
    const tools = buildTools(sessionId);

    // Add an item using the known exact name
    const addRes = await (tools.add_to_cart as any).execute({
      item: 'Big Mac',
      quantity: 1,
      modifiers: [],
    });
    assert.equal(addRes.added, true);

    const res = await (tools.confirm_order as any).execute({});
    assert.equal(res.confirmed, true);
    assert.equal(typeof res.orderNumber, 'number');
    assert.ok(res.orderNumber > 0);
  });
});

describe('deterministic intent routing contracts', () => {
  it('routes recognized ASL ordering syntax directly to menu items', async () => {
    const intent = await resolveIntent('TWO HAMBURGER');

    assert.notEqual(intent.kind, 'none');
    if (intent.kind === 'add') {
      assert.ok(intent.items.length > 0);
      assert.equal(intent.items[0].qty, 2);
      assert.ok(intent.items[0].item.name.toLowerCase().includes('hamburger'));
    }
  });

  it('bypasses LLM for order clear and reset phrases', async () => {
    const intent = await resolveIntent('clear my order');
    assert.equal(intent.kind, 'clear');

    const intentRestart = await resolveIntent('start over');
    assert.equal(intentRestart.kind, 'clear');
  });

  it('hands conversational questions and greetings to the LLM agent', async () => {
    const intentGreeting = await resolveIntent('hello how are you');
    assert.equal(intentGreeting.kind, 'none');

    const intentAllergy = await resolveIntent('do your fries contain gluten?');
    assert.equal(intentAllergy.kind, 'none');
  });
});
