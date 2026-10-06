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
import { ChoicePrompt } from './ChoicePrompt';
import { useSignSocket, type SignEvent } from '@/lib/sign/useSignSocket';
import { useGlossBuffer } from '@/lib/interaction/useGlossBuffer';
import { useSpeech, useSpeechRecognition } from '@/lib/interaction/useSpeech';
import { latestReply, pendingChoice, tagged, toTurns } from '@/lib/interaction/transcript';
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
  // Speech output is off by default. The work right now is sign recognition,
  // and a terminal that talks over you while you are trying to sign at it is a
  // distraction from the thing being tested. The toggle stays: the spoken
  // channel is how a blind customer uses this, so it is switched off, not gone.
  const [muted, setMuted] = useState(true);

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
  // Read from the tool result rather than the reply, because the reply has
  // been observed claiming an item was added when none was.
  const choice = useMemo(() => pendingChoice(messages), [messages]);

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
    <div className="flex h-[calc(100dvh-4.5rem)] min-h-0 gap-5">
      {/* ----------------------------------------------------------------
          You sign AT the camera, so the camera gets the room. Underneath it,
          only the two things you need while signing: what the terminal just
          said, and the other ways to say something.
         ---------------------------------------------------------------- */}
      <main className="flex min-w-0 flex-1 flex-col gap-4">
        <div className="relative min-h-0 flex-1">
          <CameraStage
            onFrame={handleFrame}
            onArrive={handleArrive}
            onDepart={handleDepart}
            presence={presence}
            thresholds={thresholds}
            showPose={showOverlay}
          />

          {/* Sound lives on the view it belongs to, out of the reading path. */}
          <button
            type="button"
            onClick={() => {
              setMuted((m) => !m);
              speech.cancel();
            }}
            aria-pressed={!muted}
            className="absolute right-4 top-4 min-h-11 rounded-control bg-paper/90 px-4 text-[1rem] font-bold text-ink-soft backdrop-blur-sm transition hover:text-ink"
          >
            {muted ? 'Sound off' : 'Sound on'}
          </button>
        </div>

        <CaptionPanel turns={turns} pendingGlosses={glossBuffer.glosses} thinking={busy} />

        {choice && (
          <ChoicePrompt
            choice={choice}
            disabled={busy}
            onPick={(name) =>
              send(tagged('touch', `${choice.quantity > 1 ? choice.quantity : 'One'} ${name}.`))
            }
          />
        )}

        {error && (
          <p
            role="alert"
            className="shrink-0 rounded-panel border border-danger bg-danger-soft px-6 py-4 text-[1rem] text-danger"
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
          Menu above, order below, both on screen at once.

          These used to share one panel and a toggle. That was wrong in a way
          that mattered: being asked "which burger?" is the normal shape of this
          conversation, and answering it meant switching away from the running
          total to go and look. Worse, a signer whose vocabulary cannot express
          "Double Quarter Pounder" needs the menu VISIBLE to point at. The total
          stays pinned at the bottom, which is the number people check most.
         ---------------------------------------------------------------- */}
      <div className="flex w-[25rem] shrink-0 flex-col gap-4 xl:w-[29rem]">
        <section
          aria-label="Menu"
          className="flex min-h-0 flex-1 flex-col rounded-panel border border-line bg-card"
        >
          <div className="flex items-baseline justify-between border-b border-line px-6 py-4">
            <h2 className="text-[1.25rem] font-bold tracking-[-0.01em]">Menu</h2>
            <p className="text-[0.9375rem] text-ink-faint">Tap to add</p>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
            <MenuGrid
              categories={menu}
              disabled={busy}
              onPick={(item) => send(tagged('touch', `Add one ${item.name}.`))}
            />
          </div>
        </section>

        <CartPanel cart={cart} onConfirm={confirmOrder} onClear={startOver} busy={busy} />
      </div>
    </div>
  );
}
