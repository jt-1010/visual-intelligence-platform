'use client';

import { useEffect, useRef } from 'react';
import type { Turn } from '@/lib/interaction/transcript';

type Props = {
  turns: Turn[];
  /** Signs recognised but not yet sent -- the person mid-sentence. */
  glosses: string[];
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
export function CaptionPanel({ turns, glosses, thinking }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const empty = turns.length === 0 && glosses.length === 0 && !thinking;

  // Newest turn pinned into view. A customer should never have to scroll to
  // read the thing that was just said to them.
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns, glosses, thinking]);

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
      /*
        The conversation takes the room again. As a short strip it could not hold
        one reply about tender sizes without scrolling, which is no use to
        someone who reads this instead of hearing it.

        pb reserves the bottom-right corner the camera is inset into, so text
        never slides underneath it.
      */
      className={[
        'flex min-h-0 flex-1 flex-col overflow-y-auto rounded-panel border border-line',
        'bg-card px-8 py-7 pb-[12.5rem] shadow-[0_1px_2px_rgba(0,0,0,0.04)]',
      ].join(' ')}
    >
      {empty ? (
        <div className="my-auto">
          <p className="max-w-[34ch] text-balance text-[clamp(1.5rem,2.2vw,2.125rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink-faint">
            Step up to order.
          </p>
          <p className="mt-2 max-w-[44ch] text-[1.0625rem] leading-relaxed text-ink-soft">
            Sign, speak, type, or tap the menu — whichever suits you.
          </p>
        </div>
      ) : (
        // mt-auto keeps the newest line pinned to the bottom, next to the
        // composer, so it never moves as the history grows above it. (justify-end
        // would do the same until the content overflows, then clip the oldest
        // turns out of reach -- an auto margin scrolls correctly.)
        <div className="mt-auto space-y-5">
          {turns.map((turn) => {
            if (turn.role === 'customer') {
              return (
                <div key={turn.id}>
                  <p className="text-[0.9375rem] text-ink-faint">{turn.label}</p>
                  <p className="mt-1 max-w-[68ch] text-[1.1875rem] leading-snug text-ink-soft">
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
                    ? 'max-w-[34ch] text-balance text-[clamp(1.5rem,2.2vw,2.125rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink'
                    : 'max-w-[68ch] text-[1.1875rem] leading-relaxed text-ink-soft'
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


          {/*
            The signs read so far, while you are still signing.

            Deliberately NOT bg-white/text-ink: that pair is white-on-white once
            the dark palette is active, which rendered each recognised word as a
            blank chip -- the word was there and simply could not be seen. These
            colours are defined for both themes.
          */}
          {glosses.length > 0 && (
            <div aria-live="polite">
              <p className="text-[0.9375rem] text-ink-faint">You&rsquo;re signing</p>
              <p className="mt-2 flex flex-wrap items-center gap-2">
                <span className="sr-only">Signs recognised so far:</span>
                {glosses.map((gloss, i) => (
                  <span
                    key={`${gloss}-${i}`}
                    className="rounded-control border border-action bg-action-soft px-3.5 py-1.5 text-[1.25rem] font-bold tracking-[-0.01em] text-action"
                  >
                    {gloss}
                  </span>
                ))}
                <span className="text-[1rem] text-ink-faint">keep going…</span>
              </p>
            </div>
          )}

          {awaitingReply && (
            <p className="text-[clamp(1.5rem,2.2vw,2.125rem)] font-bold leading-[1.15] tracking-[-0.015em] text-ink-faint">
              One moment…
            </p>
          )}
        </div>
      )}
    </div>
  );
}
