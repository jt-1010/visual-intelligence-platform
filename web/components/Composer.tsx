'use client';

import { useEffect, useRef, useState } from 'react';

type Props = {
  onSend: (text: string) => void;
  listening: boolean;
  speechSupported: boolean;
  disabled: boolean;
};

/**
 * Type, or talk. Both reach the same agent.
 *
 * Typing is not a developer convenience here, it is a primary way in. Someone
 * who cannot speak and does not sign -- which includes most people with an
 * acquired speech impairment -- still has to be able to order, and until the
 * sign model covers a wider vocabulary that is a lot of people.
 */
export function Composer({ onSend, listening, speechSupported, disabled }: Props) {
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focused on mount so a keyboard-only customer can start typing immediately
  // rather than tabbing past the camera first.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function submit() {
    const value = text.trim();
    if (!value || disabled) return;
    setText('');
    onSend(value);
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="flex items-end gap-3 rounded-panel border border-line bg-card p-3"
    >
      <label htmlFor="composer" className="sr-only">
        Type your order
      </label>
      <textarea
        id="composer"
        ref={inputRef}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // Enter sends; Shift+Enter makes a new line.
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Type what you'd like…"
        disabled={disabled}
        className="max-h-32 min-h-14 flex-1 resize-none rounded-control bg-transparent px-4 py-3.5 text-[1.125rem] leading-snug text-ink placeholder:text-ink-faint focus:outline-none disabled:opacity-50"
      />

      {/*
        A readout, not a button. The microphone is open on its own, so there is
        nothing to press -- but a person has to be able to tell whether they are
        being heard, and whether this terminal is listening to the room, which
        they are entitled to know without having to ask.
      */}
      {speechSupported && (
        <p
          aria-live="polite"
          className={[
            'flex min-h-14 shrink-0 items-center gap-2.5 rounded-control border px-5',
            'text-[1.0625rem] font-bold',
            listening
              ? 'border-action bg-action-soft text-action'
              : 'border-line text-ink-faint',
          ].join(' ')}
        >
          <span
            aria-hidden="true"
            className={[
              'inline-block h-2.5 w-2.5 rounded-full',
              listening ? 'bg-action' : 'bg-line-strong',
            ].join(' ')}
          />
          {listening ? 'Listening' : 'Mic off'}
        </p>
      )}

      <button
        type="submit"
        disabled={disabled || text.trim().length === 0}
        className="min-h-14 shrink-0 rounded-control bg-action px-7 text-[1.0625rem] font-bold text-white transition hover:bg-action-hover disabled:cursor-not-allowed disabled:bg-line-strong disabled:text-ink-faint"
      >
        Send
      </button>
    </form>
  );
}
