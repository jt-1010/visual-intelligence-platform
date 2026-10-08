'use client';

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

/**
 * Text-to-speech and speech-to-text via the Web Speech API.
 *
 * TTS speaks sentence by sentence as the response streams, rather than waiting
 * for the whole message. On a two-sentence reply that is roughly a second of
 * perceived latency removed for a blind user, who has nothing to look at while
 * the text fills in -- for them the spoken output IS the interface, so its
 * latency is the system's latency.
 */

// The Web Speech API is not in TypeScript's DOM lib. Minimal shapes only.
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: (() => void) | null;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
};

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  }
}

/**
 * Capability detection that survives server rendering.
 *
 * `typeof window` checks during render cause a hydration mismatch, and setting
 * the flag in an effect causes a cascading render. useSyncExternalStore exists
 * for exactly this: it hands React a different snapshot on the server (false)
 * and the client (the real answer) without either problem.
 */
const NEVER_CHANGES = () => () => {};
const serverFalse = () => false;

/** Split off complete sentences, leaving any trailing partial behind. */
function completeSentences(text: string): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  const re = /[^.!?]+[.!?]+[\s"')\]]*/g;
  let lastIndex = 0;
  let m: RegExpExecArray | null;

  while ((m = re.exec(text)) !== null) {
    sentences.push(m[0].trim());
    lastIndex = re.lastIndex;
  }

  return { sentences, rest: text.slice(lastIndex) };
}

export function useSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const spokenUpTo = useRef(0);

  const supported = useSyncExternalStore(
    NEVER_CHANGES,
    () => 'speechSynthesis' in window,
    serverFalse,
  );

  const enqueue = useCallback((text: string) => {
    if (!text.trim() || typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.02;
    utterance.pitch = 1.0;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(window.speechSynthesis.pending);
    window.speechSynthesis.speak(utterance);
  }, []);

  /**
   * Feed the growing assistant message. Speaks each sentence once, the moment
   * it is complete.
   */
  const speakStreaming = useCallback(
    (fullText: string) => {
      const pending = fullText.slice(spokenUpTo.current);
      const { sentences, rest } = completeSentences(pending);
      if (sentences.length === 0) return;

      spokenUpTo.current = fullText.length - rest.length;
      for (const s of sentences) enqueue(s);
    },
    [enqueue],
  );

  /** Flush whatever is left when the stream ends mid-sentence. */
  const flush = useCallback(
    (fullText: string) => {
      const rest = fullText.slice(spokenUpTo.current).trim();
      spokenUpTo.current = fullText.length;
      if (rest) enqueue(rest);
    },
    [enqueue],
  );

  const reset = useCallback(() => {
    spokenUpTo.current = 0;
  }, []);

  const cancel = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeaking(false);
  }, []);

  return { speakStreaming, flush, reset, cancel, speaking, supported };
}

/**
 * Always-on listening, with the microphone closed while the terminal talks.
 *
 * `enabled` is the whole interface. Pressing a button before speaking is a
 * barrier for exactly the people this terminal is for -- someone with limited
 * motor control, or a blind customer who cannot find the button -- so the
 * caller keeps this true and simply talks.
 *
 * The hard part is that an open microphone hears the terminal's own voice and
 * transcribes it back as a customer utterance, which answers itself, which it
 * then hears again. The caller closes the microphone while speech is playing by
 * passing `enabled: false`, and this hook makes that cheap: the recogniser is
 * stopped and restarted rather than rebuilt, so no audio is captured from the
 * speakers at all. Browser echo cancellation is not enough on its own, because
 * laptop speakers and microphone share a chassis.
 *
 * Chrome also ends recognition on its own every few seconds of silence, so a
 * session that is meant to stay open has to be restarted each time it stops.
 */
export function useSpeechRecognition(
  onTranscript: (text: string) => void,
  enabled = false,
) {
  const [listening, setListening] = useState(false);
  const recRef = useRef<SpeechRecognitionLike | null>(null);
  const onTranscriptRef = useRef(onTranscript);
  // Read by handlers that outlive the render they were created in.
  const enabledRef = useRef(enabled);
  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Written after render, never during it: a ref mutated mid-render can be
  // read by a concurrent render that then sees a stale callback.
  useEffect(() => {
    onTranscriptRef.current = onTranscript;
  });

  const supported = useSyncExternalStore(
    NEVER_CHANGES,
    () => Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition),
    serverFalse,
  );

  useEffect(() => {
    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!Ctor) return;

    const rec = new Ctor();
    rec.lang = 'en-US';
    rec.continuous = true;
    rec.interimResults = false;

    // Listening state comes from the recogniser, not from our intent to start
    // it: start() can throw, be refused, or end on its own a moment later, and
    // a dot that says "listening" while the microphone is shut is worse than
    // no dot at all.
    rec.onstart = () => setListening(true);

    rec.onresult = (e) => {
      const last = e.results[e.results.length - 1];
      const text = last?.[0]?.transcript?.trim();
      if (text) onTranscriptRef.current(text);
    };
    rec.onerror = (e) => {
      // 'no-speech' and 'aborted' are routine in a session held open all day.
      if (e.error !== 'no-speech' && e.error !== 'aborted') {
        console.warn('[speech] recognition error:', e.error);
      }
    };
    rec.onend = () => {
      setListening(false);
      // Chrome ends the session on its own after a stretch of silence. If
      // listening is still wanted, open it again -- but after a beat, so a
      // recogniser that is failing immediately cannot spin.
      if (!enabledRef.current) return;
      if (restartTimer.current) clearTimeout(restartTimer.current);
      restartTimer.current = setTimeout(() => {
        if (!enabledRef.current) return;
        try {
          rec.start();
        } catch {
          // Already running: harmless.
        }
      }, 400);
    };

    recRef.current = rec;
    return () => {
      enabledRef.current = false;
      if (restartTimer.current) clearTimeout(restartTimer.current);
      rec.abort();
    };
  }, []);

  // Open and close the microphone as the caller asks. This runs whenever
  // `enabled` flips, which is how the terminal stays deaf while it is talking.
  useEffect(() => {
    enabledRef.current = enabled;
    const rec = recRef.current;
    if (!rec) return;

    if (enabled) {
      try {
        rec.start();
      } catch {
        // Already running.
      }
    } else {
      if (restartTimer.current) clearTimeout(restartTimer.current);
      rec.stop();
    }
  }, [enabled]);

  return { listening, supported };
}
