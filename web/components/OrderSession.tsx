'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { CameraStage } from './CameraStage';
import { DetectionPanel } from './DetectionPanel';
import { CaptionPanel } from './CaptionPanel';
import { CartPanel, type Cart } from './CartPanel';
import { MenuGrid, type MenuItem } from './MenuGrid';
import { Composer } from './Composer';
import { useSignSocket, type SignEvent } from '@/lib/sign/useSignSocket';
import { useGlossBuffer } from '@/lib/interaction/useGlossBuffer';
import { useSpeech, useSpeechRecognition } from '@/lib/interaction/useSpeech';
import { latestReply, tagged, toTurns } from '@/lib/interaction/transcript';
import type { CapturedFrame } from '@/lib/mediapipe/landmarks';
import {
  DEFAULT_THRESHOLDS,
  type PresenceState,
  type PresenceThresholds,
} from '@/lib/mediapipe/presence';

/**
 * One ordering session.
 *
 * The design principle worth stating: there is no "select your disability"
 * screen. When someone steps up, the terminal greets them out loud AND in large
 * captions AND shows the menu, then follows whichever channel they answer on.
 * Making a person declare a disability to a machine before it will serve them
 * is slow, and it is the wrong thing to build.
 *
 * Every input path -- sign, speech, touch -- becomes a tagged message to the
 * same agent, so the order stays one conversation no matter how it was placed.
 */

type Props = {
  sessionId: string;
  onSessionEnd: () => void;
  /**
   * Developer tuning instruments, off unless ?tune=1 is in the URL.
   *
   * Threshold sliders and landmark readouts are for whoever is calibrating the
   * camera, not for a customer standing at a counter trying to order lunch.
   * Putting them in the default view asked a person with a disability to
   * understand our detection internals before they could buy a burger.
   */
  tuning?: boolean;
};

export function OrderSession({ sessionId, onSessionEnd, tuning = false }: Props) {
  const [presence, setPresence] = useState<PresenceState>('absent');
  const [handsVisible, setHandsVisible] = useState(0);
  const [shoulderWidth, setShoulderWidth] = useState<number | null>(null);
  const [heldMs, setHeldMs] = useState(0);
  const [bodyDetected, setBodyDetected] = useState(false);
  const [rawDropouts, setRawDropouts] = useState(0);
  const [thresholds, setThresholds] = useState<PresenceThresholds>(DEFAULT_THRESHOLDS);
  const [showOverlay, setShowOverlay] = useState(tuning);
  const [cart, setCart] = useState<Cart | null>(null);
  const [menu, setMenu] = useState<Record<string, MenuItem[]>>({});
  const [muted, setMuted] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: '/api/agent', body: { sessionId } }),
    [sessionId],
  );

  // Held in a ref so `onFinish` always runs the latest handler. The Chat
  // instance captures its callbacks once, so a plain closure here would go on
  // reading the `muted` value from the render where the chat was created.
  const finishRef = useRef<(finalText: string) => void>(() => {});

  const { messages, sendMessage, status, error } = useChat({
    transport,
    onFinish: ({ message }) => {
      const text = message.parts
        .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
        .map((p) => p.text)
        .join('');
      finishRef.current(text);
    },
  });
  const speech = useSpeech();
  const greetedRef = useRef(false);

  // --- What the customer sees, and what gets spoken -------------------------
  const turns = useMemo(() => toTurns(messages), [messages]);
  const assistantText = useMemo(() => latestReply(turns), [turns]);

  const send = useCallback(
    (text: string) => {
      speech.cancel();
      speech.reset();
      sendMessage({ text });
    },
    [sendMessage, speech],
  );

  // --- Sign input ----------------------------------------------------------
  const glossBuffer = useGlossBuffer(
    useCallback(
      (glosses: string[]) => {
        send(tagged('sign', glosses.join(' ')));
      },
      [send],
    ),
  );

  const sign = useSignSocket(
    useCallback(
      (event: SignEvent) => {
        glossBuffer.add(event.label);
      },
      [glossBuffer],
    ),
  );

  // --- Speech input --------------------------------------------------------
  const recognition = useSpeechRecognition(
    useCallback(
      (text: string) => {
        send(tagged('speech', text));
      },
      [send],
    ),
  );

  // --- Camera frames -------------------------------------------------------
  const handleFrame = useCallback(
    (frame: CapturedFrame) => {
      setPresence(frame.presence);
      setHandsVisible(frame.handsVisible);
      setShoulderWidth(frame.shoulderWidth);
      setHeldMs(frame.heldMs);
      setBodyDetected(frame.bodyDetected);
      setRawDropouts(frame.rawDropouts);
      // Only stream landmarks while someone is actually there. Feeding an empty
      // frame to the recogniser all day is wasted CPU and wasted bandwidth.
      if (frame.presence === 'present') sign.send(frame.lm);
    },
    [sign],
  );

  const handleArrive = useCallback(() => {
    if (greetedRef.current) return;
    greetedRef.current = true;
    send(tagged('presence', 'A customer has just stepped up to the counter.'));
  }, [send]);

  const handleDepart = useCallback(() => {
    glossBuffer.clear();
    speech.cancel();
    onSessionEnd();
  }, [glossBuffer, speech, onSessionEnd]);

  const updateThresholds = useCallback(
    (next: Partial<PresenceThresholds>) => setThresholds((t) => ({ ...t, ...next })),
    [],
  );

  // --- Speak the response as it streams ------------------------------------
  useEffect(() => {
    if (muted || !assistantText) return;
    speech.speakStreaming(assistantText);
  }, [assistantText, muted, speech]);

  // --- On turn end: finish speaking, refresh the cart -----------------------
  const refreshCart = useCallback(async () => {
    try {
      const res = await fetch(`/api/cart?sessionId=${encodeURIComponent(sessionId)}`);
      if (res.ok) setCart(await res.json());
    } catch {
      // Non-fatal: the caption already told the person what happened.
    }
  }, [sessionId]);

  // Reacting to the finish EVENT rather than to a `status === 'ready'`
  // transition. The turn ending is a real thing that happens once; watching a
  // derived state for it means re-running on every unrelated re-render that
  // lands while the status happens to be 'ready'.
  useEffect(() => {
    finishRef.current = (finalText: string) => {
      if (!muted && finalText) speech.flush(finalText);
      void refreshCart();
    };
  });

  // --- Menu ----------------------------------------------------------------
  useEffect(() => {
    fetch('/api/menu')
      .then((r) => r.json())
      .then((d) => setMenu(d.categories ?? {}))
      .catch(() => setMenu({}));
  }, []);

  const confirmOrder = useCallback(() => {
    send(tagged('touch', "That's everything. Please read back my order and confirm it."));
  }, [send]);

  const startOver = useCallback(() => {
    send(tagged('touch', 'Cancel everything and start over.'));
  }, [send]);

  const busy = status === 'submitted' || status === 'streaming';

  return (
    <div className="flex h-[calc(100dvh-4.5rem)] min-h-0 gap-6">
      {/* ----------------------------------------------------------------
          The conversation holds the centre. For a Deaf customer this column
          is the entire interaction, so it gets the width and the largest type.
         ---------------------------------------------------------------- */}
      <main className="flex min-w-0 flex-1 flex-col gap-4">
        <CaptionPanel turns={turns} pendingGlosses={glossBuffer.glosses} thinking={busy} />

        {error && (
          <p
            role="alert"
            className="rounded-panel border border-danger bg-danger-soft px-6 py-4 text-[1rem] text-danger"
          >
            {error.message}
          </p>
        )}

        <Composer
          onSend={(text) => send(tagged('text', text))}
          onSpeak={recognition.start}
          onStopListening={recognition.stop}
          listening={recognition.listening}
          speechSupported={recognition.supported}
          disabled={busy}
        />

        {/* The self-view sits beside the controls, not above the fold. */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <CameraStage
            onFrame={handleFrame}
            onArrive={handleArrive}
            onDepart={handleDepart}
            presence={presence}
            thresholds={thresholds}
            showPose={showOverlay}
          />

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              aria-expanded={menuOpen}
              className="min-h-11 rounded-control border border-line-strong px-4 text-[1rem] font-bold text-ink transition hover:bg-sunk"
            >
              {menuOpen ? 'Hide menu' : 'Browse the menu'}
            </button>
            <button
              type="button"
              onClick={() => {
                setMuted((m) => !m);
                speech.cancel();
              }}
              aria-pressed={muted}
              className="min-h-11 rounded-control border border-line-strong px-4 text-[1rem] text-ink-soft transition hover:bg-sunk hover:text-ink"
            >
              {muted ? 'Sound off' : 'Sound on'}
            </button>
          </div>
        </div>

        {tuning && (
          <DetectionPanel
            presence={presence}
            shoulderWidth={shoulderWidth}
            heldMs={heldMs}
            handsVisible={handsVisible}
            bodyDetected={bodyDetected}
            rawDropouts={rawDropouts}
            thresholds={thresholds}
            onChange={updateThresholds}
            showOverlay={showOverlay}
            onToggleOverlay={setShowOverlay}
          />
        )}
      </main>

      {/* ----------------------------------------------------------------
          The order, always visible. Swaps to the menu when asked for, so the
          menu gets real room instead of a cramped drawer.
         ---------------------------------------------------------------- */}
      <div className="flex w-[23rem] shrink-0 flex-col xl:w-[26rem]">
        {menuOpen ? (
          <section
            aria-label="Menu"
            className="flex h-full min-h-0 flex-col rounded-panel border border-line bg-card"
          >
            <div className="flex items-baseline justify-between border-b border-line px-6 py-5">
              <h2 className="text-[1.375rem] font-bold tracking-[-0.01em]">Menu</h2>
              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                className="rounded-control px-2 py-1 text-[0.9375rem] text-ink-soft underline-offset-4 transition hover:text-ink hover:underline"
              >
                Back to order
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
              <MenuGrid
                categories={menu}
                disabled={busy}
                onPick={(item) => send(tagged('touch', `Add one ${item.name}.`))}
              />
            </div>
          </section>
        ) : (
          <CartPanel cart={cart} onConfirm={confirmOrder} onClear={startOver} busy={busy} />
        )}
      </div>
    </div>
  );
}
