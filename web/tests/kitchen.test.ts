import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { parseNames, ticketText } from '@/lib/translate/kitchen';

describe('parseNames', () => {
  it('reads a JSON array even when the model wraps it in prose', () => {
    assert.deepEqual(parseNames('Here you go:\n["Hamburguesa", "Papas fritas"]', 2), [
      'Hamburguesa',
      'Papas fritas',
    ]);
  });

  it('rejects a list that dropped or added an item', () => {
    assert.equal(parseNames('["Hamburguesa"]', 2), null);
    assert.equal(parseNames('["a", "b", "c"]', 2), null);
    assert.equal(parseNames('not json', 1), null);
  });
});

describe('ticketText', () => {
  const lines = [
    { qty: 2, name: 'X' },
    { qty: 1, name: 'Y' },
  ];
  it('puts the order quantities into each language', () => {
    assert.equal(ticketText(lines, null), 'Kitchen order: 2 X, 1 Y.');
    assert.equal(ticketText(lines, 'es'), 'Pedido para cocina: 2 X, 1 Y.');
    assert.equal(ticketText(lines, 'zh'), '厨房订单：X 2份，Y 1份。');
  });
});
