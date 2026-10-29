'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { CameraStage } from './CameraStage';
import { DetectionPanel } from './DetectionPanel';
import { CaptionPanel } from './CaptionPanel';
import { CartPanel, type Cart } from './CartPanel';
import { MenuGrid, type MenuItem } from './MenuGrid';
import { MenuDialog } from './MenuDialog';
import { Composer } from './Composer';
import { ChoicePrompt } from './ChoicePrompt';
import { LanguagePicker } from "./LanguagePicker";
import { KitchenTicket, type Ticket } from "./KitchenTicket";
import { LANGUAGES, type Lang } from "@/lib/translate/languages";
import { useTranslations } from "@/lib/translate/useTranslations";
import { useSignSocket, type SignEvent } from "@/lib/sign/useSignSocket";
import { useGlossBuffer } from "@/lib/interaction/useGlossBuffer";
import { useSpeech, useSpeechRecognition } from "@/lib/interaction/useSpeech";
import {
  replyToSpeak,
  pendingChoice,
  latestConfirmedOrder,
  tagged,
  toTurns,
  type PendingChoice,
} from "@/lib/interaction/transcript";

/** Number signs the classifier knows, for picking a numbered option. */
const CHOICE_NUMBERS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
};
import type { CapturedFrame } from "@/lib/mediapipe/landmarks";
import {
  DEFAULT_THRESHOLDS,
  type PresenceState,
  type PresenceThresholds,
} from "@/lib/mediapipe/presence";

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

export function OrderSession({
  sessionId,
  onSessionEnd,
  tuning = false,
}: Props) {
  const [presence, setPresence] = useState<PresenceState>("absent");
  const [handsVisible, setHandsVisible] = useState(0);
  const [shoulderWidth, setShoulderWidth] = useState<number | null>(null);
  const [heldMs, setHeldMs] = useState(0);
  const [bodyDetected, setBodyDetected] = useState(false);
  const [rawDropouts, setRawDropouts] = useState(0);
  const [thresholds, setThresholds] =
    useState<PresenceThresholds>(DEFAULT_THRESHOLDS);
  const [showOverlay, setShowOverlay] = useState(tuning);
  const [cart, setCart] = useState<Cart | null>(null);
  const [menu, setMenu] = useState<Record<string, MenuItem[]>>({});
  const [menuOpen, setMenuOpen] = useState(false);
    /*
    The terminal speaks, and listens, without being asked to.

    Pressing a button before you may talk is a barrier for the people this is
    built for: someone with limited motor control, or a blind customer who
    cannot find the button. So the speaker is on and the microphone is open,
    and a customer simply talks -- or signs, which runs at the same time
    through a different channel entirely.

    The toggle stays for the room this stands in, not for the customer.
  */
  const [muted, setMuted] = useState(false);
  const [translateTo, setTranslateTo] = useState<Lang | null>(null);
  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [sendingToKitchen, setSendingToKitchen] = useState(false);
  const [kitchenError, setKitchenError] = useState<string | null>(null);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/agent", body: { sessionId } }),
    [sessionId],
  );

  // Held in a ref so `onFinish` always runs the latest handler. The Chat
  // instance captures its callbacks once, so a plain closure here would go on
  // reading the `muted` value from the render where the chat was created.
  const finishRef = useRef<(messageId: string, finalText: string) => void>(
    () => {},
  );

  const { messages, setMessages, sendMessage, status, error } = useChat({
    transport,
    onFinish: ({ message }) => {
      const text = message.parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join("");
      finishRef.current(message.id, text);
    },
  });
  const speech = useSpeech();
  const greetedRef = useRef(false);

  // --- What the customer sees, and what gets spoken -------------------------
  const turns = useMemo(() => toTurns(messages), [messages]);
  const reply = useMemo(() => replyToSpeak(turns), [turns]);
  // Read from the tool result rather than the reply, because the reply has
  // been observed claiming an item was added when none was.
  const choice = useMemo(() => pendingChoice(messages), [messages]);
  const confirmedOrder = useMemo(() => latestConfirmedOrder(messages), [messages]);

  // Translation of signed turns
  const latestSignId = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].channel === "sign") return turns[i].id;
    }
    return null;
  }, [turns]);
  const latestSignRef = useRef(latestSignId);
  const mutedRef = useRef(muted);
  useEffect(() => {
    latestSignRef.current = latestSignId;
    mutedRef.current = muted;
  });

  const speakTranslation = useCallback(
    (text: string, lang: Lang) => speech.speakIn(text, LANGUAGES[lang].speech),
    [speech],
  );
  const translations = useTranslations(
    turns,
    translateTo,
    useCallback(
      (turnId: string, text: string, lang: Lang) => {
        // Only the sign just made is read aloud; switching language must not replay history.
        if (!mutedRef.current && turnId === latestSignRef.current)
          speakTranslation(text, lang);
      },
      [speakTranslation],
    ),
  );

  const send = useCallback(
    (text: string) => {
      speech.cancel();
      sendMessage({ text });
    },
    [sendMessage, speech],
  );

  const pickChoice = useCallback(
    (picked: PendingChoice, name: string) =>
      send(
        tagged(
          "touch",
          `${picked.quantity > 1 ? picked.quantity : "One"} ${name}.`,
        ),
      ),
    [send],
  );

  // Read inside the sign callback without re-creating the gloss buffer.
  const choiceRef = useRef(choice);
  useEffect(() => {
    choiceRef.current = choice;
  });

  // --- Sign input ----------------------------------------------------------
  // Bridge socket events to the buffer so phrase flushes can reset the socket.
  const addGlossRef = useRef<(label: string) => void>(() => {});
  const sign = useSignSocket((event: SignEvent) => {
    addGlossRef.current(event.label);
  });
  const { reset: resetSign } = sign;

  const glossBuffer = useGlossBuffer(
    useCallback(
      (glosses: string[]) => {
        resetSign();
        /*
          While "which one?" is open, a number sign picks that option even
          when recognition adds noise glosses alongside it.
          Menu names are far outside the sign vocabulary ("Premium Crispy
          Chicken Deluxe Sandwich"), but ONE to NINE are all in it, so the
          options are numbered and the person signs the number.
        */
        const open = choiceRef.current;
        if (open) {
          const number = glosses.find((g) => {
            const n = CHOICE_NUMBERS[g.toLowerCase()];
            return n !== undefined && n >= 1 && n <= open.options.length;
          });
          if (number !== undefined) {
            const n = CHOICE_NUMBERS[number.toLowerCase()];
            pickChoice(open, open.options[n - 1].name);
            return;
          }
        }
        send(tagged("sign", glosses.join(" ")));
      },
      [send, pickChoice, resetSign],
    ),
  );

  useEffect(() => {
    addGlossRef.current = glossBuffer.add;
  }, [glossBuffer.add]);

  // --- Speech input --------------------------------------------------------
  /*
    The microphone is open except while the terminal is talking or thinking.

    An open microphone hears the terminal's own voice through the laptop
    speakers and sends it back as a customer utterance, which answers itself,
    which it hears again. Closing it while `speaking` is the cure, and closing
    it while `busy` keeps room noise out of the gap between question and reply.
  */
  const answering = status === "submitted" || status === "streaming";
  const speechHandsFree = !muted && !speech.speaking && !answering;
  const recognition = useSpeechRecognition(
    useCallback(
      (text: string) => {
        send(tagged("speech", text));
      },
      [send],
    ),
    speechHandsFree,
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
      // Keep signing uninterrupted while presence tracking recovers.
      if (
        frame.presence === "present" ||
        frame.presence === "leaving" ||
        frame.handsVisible > 0
      ) sign.send(frame.lm);
    },
    [sign],
  );

  const handleArrive = useCallback(() => {
    resetSign();
    if (greetedRef.current) return;
    greetedRef.current = true;
    send(tagged("presence", "A customer has just stepped up to the counter."));
  }, [send, resetSign]);

  const handleDepart = useCallback(() => {
    resetSign();
    // Tracking blips must not discard an order already in progress.
    if (confirmedOrder || (cart && cart.lines.length > 0)) return;
    glossBuffer.clear();
    speech.cancel();
    onSessionEnd();
  }, [cart, confirmedOrder, glossBuffer, speech, resetSign, onSessionEnd]);

  const updateThresholds = useCallback(
    (next: Partial<PresenceThresholds>) =>
      setThresholds((t) => ({ ...t, ...next })),
    [],
  );

  // --- Speak the response as it streams ------------------------------------
  useEffect(() => {
    if (muted || !reply) return;
    speech.speakStreaming(reply.id, reply.text);
  }, [reply, muted, speech]);

  // --- On turn end: finish speaking, refresh the cart -----------------------
  const refreshCart = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/cart?sessionId=${encodeURIComponent(sessionId)}`,
      );
      if (res.ok) setCart(await res.json());
    } catch {
      // Non-fatal: the caption already told the person what happened.
    }
  }, [sessionId]);

  // Sync on mount, message additions, and status transitions, including ready
  // when a turn completes without delivering onFinish.
  useEffect(() => {
    // State updates only after the cart request completes, never synchronously.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshCart();
  }, [messages.length, status, refreshCart]);

  // Keep speech completion tied to the finish event.
  useEffect(() => {
    finishRef.current = (messageId: string, finalText: string) => {
      if (!muted && finalText) speech.flush(messageId, finalText);
      void refreshCart();
    };
  });

  // --- Menu ----------------------------------------------------------------
  useEffect(() => {
    fetch("/api/menu")
      .then((r) => r.json())
      .then((d) => setMenu(d.categories ?? {}))
      .catch(() => setMenu({}));
  }, []);

  const confirmOrder = useCallback(() => {
    send(
      tagged(
        "touch",
        "Confirm my order.",
      ),
    );
  }, [send]);

  const speakTicket = useCallback(
    (t: Ticket) => {
      speech.cancel();
      speech.speakIn(t.text, t.lang ? LANGUAGES[t.lang].speech : "en-US");
    },
    [speech],
  );

  const sendToKitchen = useCallback(async () => {
    setSendingToKitchen(true);
    setKitchenError(null);
    try {
      const res = await fetch("/api/kitchen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, lang: translateTo }),
      });
      const data = (await res.json()) as Ticket & { error?: string };
      if (!res.ok) throw new Error(data.error ?? "could not send the order");
      setTicket(data);
      // The kitchen hears it even when the customer-facing voice is muted.
      speakTicket(data);
    } catch (e) {
      setKitchenError(
        e instanceof Error ? e.message : "could not send the order",
      );
    } finally {
      setSendingToKitchen(false);
    }
  }, [sessionId, translateTo, speakTicket]);

  const startOver = useCallback(() => {
    setMessages([]);
    setTicket(null);
    setKitchenError(null);
    glossBuffer.clear();
    speech.cancel();
    send(tagged("touch", "Cancel everything and start over."));
  }, [send, setMessages, glossBuffer, speech]);

  const busy = status === "submitted" || status === "streaming";

  return (
    <div className="flex h-[calc(100dvh-4.5rem)] min-h-0 gap-5">
      {/* ----------------------------------------------------------------
          Left: the conversation, which is what you read. Right: the camera you
          sign at, and the order it is building. The menu is a popup, so it
          takes no room until someone wants to browse.
         ---------------------------------------------------------------- */}
      <main className="flex min-w-0 flex-1 flex-col gap-4">
        <div className="flex min-h-0 flex-1 flex-col">
          <CaptionPanel
            turns={turns}
            glosses={glossBuffer.glosses}
            thinking={busy}
            translateTo={translateTo}
            translations={translations}
            onSpeakTranslation={speakTranslation}
          />
        </div>

        {choice && (
          <ChoicePrompt
            choice={choice}
            disabled={busy}
            onPick={(name) => pickChoice(choice, name)}
          />
        )}

        {ticket && (
          <KitchenTicket
            ticket={ticket}
            onReplay={() => speakTicket(ticket)}
            onClose={() => setTicket(null)}
          />
        )}

        {kitchenError && (
          <p
            role="alert"
            className="shrink-0 rounded-panel border border-danger bg-danger-soft px-6 py-4 text-[1rem] text-danger"
          >
            Kitchen: {kitchenError}
          </p>
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
          onSend={(text) => send(tagged("text", text))}
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

      {/* The order gets its own full-height column, between the chat and the
          camera, so a long order stays readable without scrolling. */}
      <div className="flex w-[20rem] shrink-0 flex-col xl:w-[24rem]">
        <CartPanel
          cart={cart}
          confirmedOrder={confirmedOrder}
          onNewOrder={startOver}
          onConfirm={confirmOrder}
          onClear={startOver}
          busy={busy}
          onSendToKitchen={sendToKitchen}
          kitchenLanguage={
            translateTo ? LANGUAGES[translateTo].native : "English"
          }
          sendingToKitchen={sendingToKitchen}
        />
      </div>

      <aside className="flex w-[18rem] shrink-0 flex-col gap-4 xl:w-[22rem]">
        <section aria-label="Camera" className="flex shrink-0 flex-col gap-2">
          <div className="aspect-video w-full overflow-hidden rounded-panel border border-line-strong">
            <CameraStage
              onFrame={handleFrame}
              onArrive={handleArrive}
              onDepart={handleDepart}
              presence={presence}
              thresholds={thresholds}
              showPose={showOverlay}
            />
          </div>

          <p className="rounded-panel border border-line bg-card px-4 py-2 text-[0.9375rem] font-bold text-ink-soft">
            {handsVisible ? "I can see your hands" : "Raise your hands to sign"}
          </p>
        </section>

        {/* The two controls people reach for most, big enough to find at a glance. */}
        <div className="flex shrink-0 gap-3">
          <button
            type="button"
            onClick={() => setMenuOpen(true)}
            className="flex min-h-16 flex-1 items-center justify-center gap-3 rounded-panel bg-action text-[1.375rem] font-bold text-on-action transition hover:bg-action-hover"
          >
            Open menu
          </button>
          <button
            type="button"
            onClick={() => {
              setMuted((m) => !m);
              speech.cancel();
            }}
            aria-pressed={muted}
            aria-label={muted ? "Unmute" : "Mute"}
            className={[
              "flex min-h-16 w-24 shrink-0 flex-col items-center justify-center rounded-panel border-2 text-[0.9375rem] font-bold transition",
              muted
                ? "border-danger bg-danger-soft text-danger"
                : "border-line-strong bg-card text-ink hover:border-ink-soft",
            ].join(" ")}
          >
            <span aria-hidden="true" className="text-[1.5rem] leading-none">
              {muted ? "🔇" : "🔊"}
            </span>
            <span className="mt-1">{muted ? "Unmute" : "Mute"}</span>
          </button>
        </div>

        <LanguagePicker value={translateTo} onChange={setTranslateTo} />

        {/* Fills the rest of the column so it is never empty: the menu, one
            tap per item. The popup is still there for browsing big. */}
        <section
          aria-label="Quick add"
          className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-panel border border-line bg-card"
        >
          <div className="flex shrink-0 items-baseline justify-between border-b border-line px-4 py-3">
            <h2 className="text-[1.125rem] font-bold tracking-[-0.01em]">
              Quick add
            </h2>
            <button
              type="button"
              onClick={() => setMenuOpen(true)}
              className="rounded-control px-1 text-[0.9375rem] font-bold text-action underline-offset-4 hover:underline"
            >
              See all
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            <MenuGrid
              categories={menu}
              disabled={busy}
              onPick={(item) => send(tagged("touch", `Add one ${item.name}.`))}
            />
          </div>
        </section>
      </aside>

      <MenuDialog
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        categories={menu}
        disabled={busy}
        onPick={(item) => send(tagged("touch", `Add one ${item.name}.`))}
      />
    </div>
  );
}
