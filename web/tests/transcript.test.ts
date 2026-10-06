import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { latestReply, tagged, toTurns } from '@/lib/interaction/transcript';

/**
 * The transcript is the Deaf customer's entire view of the conversation, so a
 * bug here is not cosmetic -- it is the difference between seeing what the
 * terminal understood and ordering blind.
 */

function user(id: string, text: string) {
  return { id, role: 'user', parts: [{ type: 'text', text }] };
}
function assistant(id: string, text: string) {
  return { id, role: 'assistant', parts: [{ type: 'text', text }] };
}

describe('toTurns', () => {
  it('strips the channel tag and labels how the customer spoke', () => {
    const turns = toTurns([
      user('1', tagged('sign', 'WANT BURGER TWO')),
      user('2', tagged('speech', 'and a large coke')),
      user('3', tagged('text', 'no onions')),
      user('4', tagged('touch', 'Add one Fries.')),
    ]);

    assert.deepEqual(
      turns.map((t) => [t.label, t.text]),
      [
        ['You signed', 'WANT BURGER TWO'],
        ['You said', 'and a large coke'],
        ['You typed', 'no onions'],
        ['You tapped', 'Add one Fries.'],
      ],
    );
    assert.ok(turns.every((t) => t.role === 'customer'));
  });

  it('drops presence events, which the room noticed and the customer did not say', () => {
    const turns = toTurns([
      user('1', tagged('presence', 'A customer has just stepped up to the counter.')),
      assistant('2', 'Hi there! What can I get you?'),
    ]);

    assert.equal(turns.length, 1);
    assert.equal(turns[0].role, 'terminal');
    assert.equal(turns[0].text, 'Hi there! What can I get you?');
  });

  it('keeps untagged user text rather than dropping it', () => {
    const turns = toTurns([user('1', 'a plain message')]);
    assert.deepEqual(
      turns.map((t) => [t.label, t.text, t.channel]),
      [['You', 'a plain message', null]],
    );
  });

  it('skips messages with no text, such as a tool-only assistant step', () => {
    const turns = toTurns([
      { id: '1', role: 'assistant', parts: [{ type: 'tool-addToCart', state: 'output' }] },
      assistant('2', 'Added.'),
    ]);
    assert.deepEqual(turns.map((t) => t.text), ['Added.']);
  });

  it('preserves order so the history reads top to bottom', () => {
    const turns = toTurns([
      user('1', tagged('text', 'one burger')),
      assistant('2', 'One burger. Anything else?'),
      user('3', tagged('text', 'thats it')),
    ]);
    assert.deepEqual(turns.map((t) => t.role), ['customer', 'terminal', 'customer']);
  });
});

describe('latestReply', () => {
  it('returns the most recent terminal line, not the most recent turn', () => {
    const turns = toTurns([
      assistant('1', 'first answer'),
      assistant('2', 'second answer'),
      user('3', tagged('text', 'thanks')),
    ]);
    assert.equal(latestReply(turns), 'second answer');
  });

  it('is empty before the terminal has said anything', () => {
    assert.equal(latestReply(toTurns([user('1', tagged('text', 'hello'))])), '');
  });
});
