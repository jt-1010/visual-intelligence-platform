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

export type ChoiceOption = { name: string; price: string; calories: number | null };

export type PendingChoice = {
  askedFor: string;
  totalMatches: number;
  quantity: number;
  options: ChoiceOption[];
};

/**
 * The choice the terminal is currently waiting on, read from the tool result.
 *
 * This exists because the sentence cannot be trusted. Asked for "two burgers",
 * a local 7B was handed needsChoice -- nothing added, thirteen options -- and
 * replied "Added two burgers, $11.98", inventing both the action and the price
 * while the order stayed empty. The database was never at risk; the claim about
 * it was false.
 *
 * So the options are rendered from the TOOL OUTPUT, not from the reply: real
 * rows, real prices, as buttons the person can press. If the model narrates it
 * badly the choice on screen is still correct and still works. It is also
 * faster and easier than reading a list back, which matters when the person is
 * standing at a counter.
 */
export function pendingChoice(messages: UiMessage[]): PendingChoice | null {
  /*
    Only the newest reply can be asking a question. Searching the whole history
    left a stale set of buttons on screen: once the terminal had moved on to
    something else, the last tool call it happened to make was still the most
    recent one anywhere in the conversation, so "which one?" stayed up offering
    nuggets nobody had asked about for several turns.
  */
  let last = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant') {
      last = i;
      break;
    }
  }
  if (last === -1) return null;

  {
    const parts = messages[last].parts;
    for (let j = parts.length - 1; j >= 0; j--) {
      const part = parts[j] as {
        type?: unknown;
        state?: unknown;
        input?: Record<string, unknown>;
        output?: Record<string, unknown>;
      };
      if (part?.state !== 'output-available') continue;

      const out = part.output;
      if (!out) continue;

      // The most recent add decides. If it succeeded, the question that came
      // before it has been answered and must not reappear.
      if (part.type === 'tool-add_to_cart') {
        if (out.needsChoice !== true) return null;
        const options = Array.isArray(out.options) ? (out.options as ChoiceOption[]) : [];
        if (options.length === 0) return null;
        return {
          askedFor: typeof out.askedFor === 'string' ? out.askedFor : '',
          totalMatches: typeof out.totalMatches === 'number' ? out.totalMatches : options.length,
          quantity: typeof out.quantityTheyWanted === 'number' ? out.quantityTheyWanted : 1,
          options,
        };
      }

      /*
        A search that came back with several things is the same moment wearing a
        different hat. Told "two burgers", a local model may reach for
        search_menu rather than add_to_cart -- both end with the person needing
        to pick one, and which tool it chose is not something they should feel.
        Quantity is unknown on this path, so the prompt stays neutral.
      */
      if (part.type === 'tool-search_menu') {
        const items = Array.isArray(out.items) ? (out.items as ChoiceOption[]) : [];
        if (items.length < 2) return null;
        const query = part.input?.query;
        return {
          askedFor: typeof query === 'string' ? query : '',
          totalMatches: typeof out.count === 'number' ? out.count : items.length,
          quantity: 0, // unknown: the question is "which", not "how many"
          options: items.slice(0, 5),
        };
      }
    }
  }
  return null;
}

/** The terminal's most recent line -- the one that gets the big type and the voice. */
export function latestReply(turns: Turn[]): string {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === 'terminal') return turns[i].text;
  }
  return '';
}
