'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Collects recognised signs into a phrase before sending it to the agent.
 *
 * Sending each sign the moment it is recognised would be a mistake. "BURGER"
 * alone is ambiguous - the person may be about to sign "TWO", or "NO CHEESE".
 * Firing on the first sign means the agent commits to an interpretation
 * halfway through the sentence and then has to be argued out of it.
 *
 * So we wait for a gap. Signers pause between phrases the same way speakers do,
 * and that pause is the natural sentence boundary. Terminator signs jump the
 * queue because a person who has just signed FINISH should not have to wait
 * out a timer.
 */

const PHRASE_GAP_MS = 1800;
const TERMINATORS = new Set(['FINISH', 'DONE', 'THAT-ALL', 'THANK-YOU']);

/**
 * How long the same sign is treated as still being held rather than signed again.
 *
 * The recogniser re-classifies several times a second, so one held sign arrives
 * as a burst of identical labels. Those repeats used to be appended AND to
 * restart the phrase timer, which had two effects: the buffer filled with
 * "want want want want want", and because the timer never ran down, the phrase
 * was never sent at all -- the screen just said "still signing" indefinitely.
 *
 * A deliberate repetition comes after a visible pause, which is longer than
 * this window, so it still gets through.
 */
const REPEAT_WINDOW_MS = 1200;

export function useGlossBuffer(onPhrase: (glosses: string[]) => void) {
  const [glosses, setGlosses] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bufferRef = useRef<string[]>([]);
  const lastAddedAtRef = useRef(0);
  const onPhraseRef = useRef(onPhrase);

  useEffect(() => {
    onPhraseRef.current = onPhrase;
  });

  const flush = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;

    const phrase = bufferRef.current;
    bufferRef.current = [];
    setGlosses([]);
    if (phrase.length > 0) onPhraseRef.current(phrase);
  }, []);

  const add = useCallback(
    (label: string) => {
      const now = Date.now();
      const buffer = bufferRef.current;
      const stillHolding =
        buffer.length > 0 &&
        buffer[buffer.length - 1] === label &&
        now - lastAddedAtRef.current < REPEAT_WINDOW_MS;

      // Returning BEFORE the timer is cleared is the important part: a held
      // sign must not keep postponing the phrase boundary.
      if (stillHolding) {
        lastAddedAtRef.current = now;
        return;
      }
      lastAddedAtRef.current = now;

      bufferRef.current = [...buffer, label];
      setGlosses(bufferRef.current);

      if (timerRef.current) clearTimeout(timerRef.current);

      if (TERMINATORS.has(label)) {
        flush();
        return;
      }

      timerRef.current = setTimeout(flush, PHRASE_GAP_MS);
    },
    [flush],
  );

  const clear = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    bufferRef.current = [];
    setGlosses([]);
  }, []);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  return { glosses, add, flush, clear };
}
