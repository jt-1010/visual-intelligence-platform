'use client';

import { useEffect, useRef } from 'react';
import type { Turn } from '@/lib/interaction/transcript';

type Props = {
  turns: Turn[];
  pendingGlosses: string[];
  thinking: boolean;
};

/**
 * The conversation -- and for this terminal's primary user, the whole of it.
 *
 * A Deaf customer does not hear a word the system says, so this text is not a
 * transcript running alongside the real interaction: it IS the interaction.
 * That is why the newest line is the largest type on the screen and holds the
 * centre, and why everything else -- camera, cart, menu -- is arranged around
 * it rather than competing with it.
 *
 * It keeps the turns that came before, which an earlier version threw away.
 * A hearing customer gets the reply twice, in audio and in text; a Deaf
 * customer got it once, and it vanished the moment the next line arrived. If
 * you looked down at the cart while the terminal answered, the answer was
 * simply gone. The history is the equivalent of being able to say "sorry, what
 * was that?" -- and it is also, incidentally, what fills a column that used to
 * be two-thirds empty.
 */
export function CaptionPanel({ turns, pendingGlosses, thinking }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const empty = turns.length === 0 && pendingGlosses.length === 0 && !thinking;

  // Newest turn pinned into view. A customer should never have to scroll to
  // read the thing that was just said to them.
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns, pendingGlosses, thinking]);

  // While a reply is still coming, nothing gets the display size. Leaving the
  // PREVIOUS answer in the big type would present a stale line as the current
  // one -- the exact moment a customer is most likely to be reading it.
  const awaitingReply = thinking && turns[turns.length - 1]?.role !== 'terminal';

  let focus: Turn | undefined;
  if (!awaitingReply) {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === 'terminal') {
        focus = turns[i];
        break;
      }
    }
  }

  return (
    <div
      ref={scroller}
      role="log"
      aria-label="Conversation"
      aria-atomic="false"
      className={[
        'flex min-h-0 flex-1 flex-col overflow-y-auto rounded-panel border border-line',
        'bg-card px-10 py-9 shadow-[0_1px_2px_rgba(0,0,0,0.04)]',
      ].join(' ')}
    >
      {empty ? (
        <div className="my-auto">
          <p className="max-w-[26ch] text-balance text-[clamp(1.75rem,2.9vw,2.75rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink-faint">
            Step up to order.
          </p>
          <p className="mt-4 max-w-[38ch] text-[1.0625rem] leading-relaxed text-ink-soft">
            Sign, speak, type, or tap the menu — whichever suits you.
          </p>
        </div>
      ) : (
        // mt-auto keeps the newest line pinned to the bottom, next to the
        // composer, so it never moves as the history grows above it. (justify-end
        // would do the same until the content overflows, then clip the oldest
        // turns out of reach -- an auto margin scrolls correctly.)
        <div className="mt-auto space-y-6">
          {turns.map((turn) => {
            if (turn.role === 'customer') {
              return (
                <div key={turn.id}>
                  <p className="text-[0.9375rem] text-ink-faint">{turn.label}</p>
                  <p className="mt-1 max-w-[46ch] text-[1.125rem] leading-snug text-ink-soft">
                    {turn.text}
                  </p>
                </div>
              );
            }

            // The newest reply is the one being read right now, so it carries
            // the display size. Older replies stay legible but step back.
            const newest = turn === focus;
            return (
              <p
                key={turn.id}
                className={
                  newest
                    ? 'max-w-[26ch] text-balance text-[clamp(1.75rem,2.9vw,2.75rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink'
                    : 'max-w-[46ch] text-[1.125rem] leading-relaxed text-ink-soft'
                }
              >
                {turn.text}
                {newest && thinking && (
                  <span aria-hidden="true" className="caret ml-1 inline-block text-action">
                    ▌
                  </span>
                )}
              </p>
            );
          })}

          {/* Signs recognised but not yet submitted -- the customer mid-sentence. */}
          {pendingGlosses.length > 0 && (
            <div>
              <p className="text-[0.9375rem] text-ink-faint">You&rsquo;re signing</p>
              <p className="mt-1 flex flex-wrap items-center gap-2">
                {pendingGlosses.map((gloss, i) => (
                  <span
                    key={`${gloss}-${i}`}
                    className="rounded-control bg-action-soft px-3 py-1 text-[1.0625rem] font-bold text-action"
                  >
                    {gloss}
                  </span>
                ))}
                <span className="text-[0.9375rem] text-ink-faint">keep going…</span>
              </p>
            </div>
          )}

          {awaitingReply && (
            <p className="text-[clamp(1.75rem,2.9vw,2.75rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink-faint">
              One moment…
            </p>
          )}
        </div>
      )}
    </div>
  );
}
