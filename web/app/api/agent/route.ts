import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  isStepCount,
  streamText,
  toUIMessageStream,
  type UIMessage,
  type UIMessageStreamWriter,
} from 'ai';
import { activeBackend, getModel } from '@/lib/agent/model';
import { SYSTEM_PROMPT } from '@/lib/agent/prompt';
import { buildTools } from '@/lib/agent/tools';
import { resolveIntent } from '@/lib/agent/intent';
import { addToCart, formatMoney, getCart } from '@/lib/agent/cart';

export const maxDuration = 30;

/**
 * Turn a provider failure into something the person reading the screen can act on.
 *
 * The AI SDK defaults to "An error occurred." on purpose -- provider errors can
 * leak keys and internals, so it refuses to forward them. That default is right
 * for production and useless in development, where the answer is almost always
 * one of two boring configuration problems.
 *
 * So: match the boring cases explicitly and say exactly what to do about them,
 * and fall through to the safe generic message for anything unrecognised. The
 * real error always goes to the server log regardless.
 */
function explainError(error: unknown, backend: string): string {
  const raw = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.error(`[agent] backend=${backend} FAILED:`, error);

  const text = raw.toLowerCase();

  if (backend === 'ollama') {
    if (text.includes('econnrefused') || text.includes('fetch failed')) {
      return `Ollama is not reachable at ${process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434/v1'}. Start it with "ollama serve", then pull the model with "ollama pull ${process.env.OLLAMA_MODEL ?? 'qwen2.5:3b-instruct'}".`;
    }
    if (text.includes('not found') || text.includes('404')) {
      return `Ollama is running but does not have the model "${process.env.OLLAMA_MODEL ?? 'qwen2.5:3b-instruct'}". Pull it with "ollama pull ${process.env.OLLAMA_MODEL ?? 'qwen2.5:3b-instruct'}".`;
    }
  }

  if (
    text.includes('api key') ||
    text.includes('unauthorized') ||
    text.includes('401') ||
    text.includes('403') ||
    text.includes('authentication')
  ) {
    return 'No LLM credential is configured. Either set AI_GATEWAY_API_KEY in web/.env.local, or install Ollama and set LLM_BACKEND=ollama to run a model locally. See docs/development.md.';
  }

  if (text.includes('rate limit') || text.includes('429')) {
    return 'The model provider is rate limiting us. Wait a moment and try again.';
  }

  return `The assistant could not respond (${backend} backend). Check the server log for details.`;
}

/** The plain text the customer sent, with its channel tag removed. */
function lastUserText(messages: UIMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== 'user') continue;
    const text = message.parts
      .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
      .map((p) => p.text)
      .join('')
      .trim();
    // Presence is the room noticing someone, not a request. The model greets.
    if (/^\[PRESENCE\]/.test(text)) return '';
    return text.replace(/^\[[A-Z]+\]\s*/, '');
  }
  return '';
}

function say(writer: UIMessageStreamWriter, text: string) {
  writer.write({ type: 'text-start', id: 'txt-0' });
  writer.write({ type: 'text-delta', id: 'txt-0', delta: text });
  writer.write({ type: 'text-end', id: 'txt-0' });
}

/**
 * Answer an ordinary order without a language model.
 *
 * Every number in the reply is read from the row that was just written, so a
 * price on screen cannot disagree with the price in the order. That is the one
 * guarantee this system has to make, and it costs a model call to get wrong.
 */
async function respondDeterministically(
  sessionId: string,
  intent: Exclude<Awaited<ReturnType<typeof resolveIntent>>, { kind: 'none' }>,
) {
  return createUIMessageStream({
    execute: async ({ writer }) => {
      writer.write({ type: 'start' });
      writer.write({ type: 'start-step' });

      if (intent.kind === 'choose') {
        // Emitted as an add_to_cart result so the UI renders the same choice
        // buttons it would for a tool call. Nothing is written to the order.
        const callId = `local-${Date.now()}`;
        writer.write({
          type: 'tool-input-available',
          toolCallId: callId,
          toolName: 'add_to_cart',
          input: { item: intent.query, quantity: intent.qty },
        });
        writer.write({
          type: 'tool-output-available',
          toolCallId: callId,
          output: {
            added: false,
            needsChoice: true,
            askedFor: intent.query,
            totalMatches: intent.totalMatches,
            quantityTheyWanted: intent.qty,
            options: intent.options.map((o) => ({
              name: o.name,
              price: formatMoney(o.priceCents),
              calories: o.calories,
            })),
          },
        });

        const names = intent.options.map((o) => `${o.name} ${formatMoney(o.priceCents)}`);
        say(
          writer,
          `We have ${intent.totalMatches} of those. Which one? ${names.join(', ')}.`,
        );
      } else {
        const added: string[] = [];
        for (const { item, qty } of intent.items) {
          const result = await addToCart(sessionId, item.slug, qty);
          if (!result.ok) continue;

          added.push(
            `${result.added.qty} ${result.added.name} ${formatMoney(result.added.lineTotalCents)}`,
          );

          /*
            Emitted as a completed add_to_cart call even though no model asked
            for one. The UI decides whether a choice is still outstanding by
            looking at the most recent add_to_cart result, so an add that left
            no trace would leave the "which one?" buttons on screen after the
            question had been answered.
          */
          const callId = `local-add-${result.added.lineId}`;
          writer.write({
            type: 'tool-input-available',
            toolCallId: callId,
            toolName: 'add_to_cart',
            input: { item: item.slug, quantity: qty },
          });
          writer.write({
            type: 'tool-output-available',
            toolCallId: callId,
            output: {
              added: true,
              item: {
                name: result.added.name,
                quantity: result.added.qty,
                lineTotal: formatMoney(result.added.lineTotalCents),
              },
            },
          });
        }

        const cart = await getCart(sessionId);
        say(
          writer,
          added.length === 0
            ? 'Sorry, I could not add that. Please try again.'
            : `Added ${added.join(' and ')}. Your total is ${formatMoney(cart.totalCents)}. Anything else?`,
        );
      }

      writer.write({ type: 'finish-step' });
      writer.write({ type: 'finish' });
    },
  });
}

export async function POST(req: Request) {
  const started = performance.now();
  const { messages, sessionId } = (await req.json()) as {
    messages: UIMessage[];
    sessionId?: string;
  };

  if (!sessionId) {
    return Response.json({ error: 'sessionId is required' }, { status: 400 });
  }

  const backend = activeBackend();

  // The deterministic path first. If this is a plain order, it is answered from
  // the database and the model is never called -- exact prices, no waiting.
  const intent = await resolveIntent(lastUserText(messages));
  if (intent.kind !== 'none') {
    console.log(`[agent] resolved locally: ${intent.kind} (${Math.round(performance.now() - started)}ms)`);
    return createUIMessageStreamResponse({
      stream: await respondDeterministically(sessionId, intent),
    });
  }

  const result = streamText({
    model: getModel(backend),
    instructions: SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    tools: buildTools(sessionId),
    // Ordering routinely needs search -> add -> read-back in one turn.
    stopWhen: isStepCount(8),
    onFinish({ usage }) {
      // Part of the end-to-end latency budget from the spec. Logged from M0
      // onward so a regression shows up the day it lands, not at demo time.
      console.log(
        `[agent] backend=${backend} ${Math.round(performance.now() - started)}ms ` +
          `in=${usage?.inputTokens ?? '?'} out=${usage?.outputTokens ?? '?'}`,
      );
    },
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({
      stream: result.stream,
      onError: (error) => explainError(error, backend),
    }),
  });
}
