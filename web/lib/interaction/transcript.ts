/**
 * How an order was spoken, and how it reads back.
 *
 * Every input path tags its message so the agent knows how the words arrived --
 * a misread sign deserves more charity than a typed sentence, and the prompt
 * says so. Those tags used to be write-only: the model saw them, the customer
 * never did. Reading them back is what lets the transcript say "You signed"
 * rather than a flat, channel-blind "You".
 *
 * That distinction is the point. When the sign model mishears, the person needs
 * to see that it was their SIGNING that was misread, not their typing -- which
 * tells them to repeat the sign rather than check the keyboard.
 */

export type Channel = 'sign' | 'speech' | 'text' | 'touch' | 'presence';

const TAG: Record<Channel, string> = {
  sign: 'SIGN',
  speech: 'SPEECH',
  text: 'TEXT',
  touch: 'TOUCH',
  presence: 'PRESENCE',
};

/** Human label for a customer turn. `presence` never reaches the screen. */
const LABEL: Record<Channel, string> = {
  sign: 'You signed',
  speech: 'You said',
  text: 'You typed',
  touch: 'You tapped',
  presence: '',
};

/** Builds the tagged string sent to the agent. */
export function tagged(channel: Channel, text: string): string {
  return `[${TAG[channel]}] ${text}`;
}

const BY_TAG = new Map<string, Channel>(
  (Object.keys(TAG) as Channel[]).map((c) => [TAG[c], c]),
);

export type Turn = {
  id: string;
  role: 'customer' | 'terminal';
  /** Null for the terminal's own turns, and for untagged legacy messages. */
  channel: Channel | null;
  label: string;
  text: string;
};

/** The shape we need from an AI SDK UI message -- deliberately minimal. */
type UiMessage = {
  id: string;
  role: string;
  parts: unknown[];
};

function textOf(message: UiMessage): string {
  return message.parts
    .filter((p): p is { type: 'text'; text: string } => {
      if (typeof p !== 'object' || p === null) return false;
      const part = p as { type?: unknown; text?: unknown };
      return part.type === 'text' && typeof part.text === 'string';
    })
    .map((p) => p.text)
    .join('');
}

/**
 * Turns the raw message list into what a customer should see.
 *
 * Presence messages are dropped. "A customer has just stepped up to the
 * counter" is something the ROOM noticed, not something the person said, and
 * showing it back to them as their own words is both wrong and a little eerie.
 */
export function toTurns(messages: UiMessage[]): Turn[] {
  const turns: Turn[] = [];

  for (const message of messages) {
    const text = textOf(message).trim();
    if (!text) continue;

    if (message.role === 'assistant') {
      turns.push({ id: message.id, role: 'terminal', channel: null, label: '', text });
      continue;
    }
    if (message.role !== 'user') continue;

    const match = /^\[([A-Z]+)\]\s*/.exec(text);
    const channel = match ? BY_TAG.get(match[1]) : undefined;
    if (channel === 'presence') continue;

    turns.push({
      id: message.id,
      role: 'customer',
      channel: channel ?? null,
      label: channel ? LABEL[channel] : 'You',
      text: match ? text.slice(match[0].length) : text,
    });
  }

  return turns;
}

/** The terminal's most recent line -- the one that gets the big type and the voice. */
export function latestReply(turns: Turn[]): string {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === 'terminal') return turns[i].text;
  }
  return '';
}
