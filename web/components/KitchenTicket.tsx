'use client';

import { LANGUAGES, type Lang } from '@/lib/translate/languages';

export type Ticket = {
  lines: { qty: number; name: string; english: string }[];
  lang: Lang | null;
  text: string;
};

type Props = {
  ticket: Ticket;
  onReplay: () => void;
  onClose: () => void;
};

export function KitchenTicket({ ticket, onReplay, onClose }: Props) {
  const speech = ticket.lang ? LANGUAGES[ticket.lang].speech : 'en-US';
  const heading = ticket.lang ? LANGUAGES[ticket.lang].native : 'English';

  return (
    <section
      aria-label="Kitchen ticket"
      className="shrink-0 rounded-panel border-2 border-sign bg-sign-soft px-6 py-5"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[1.25rem] font-bold tracking-[-0.01em] text-ink">
          Sent to kitchen · {heading}
        </h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onReplay}
            className="rounded-control border border-line-strong bg-card px-3 py-1.5 text-[0.9375rem] font-bold text-ink hover:border-ink-soft"
          >
            <span aria-hidden="true">🔊</span> Read again
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-control px-3 py-1.5 text-[0.9375rem] text-ink-soft hover:text-ink"
          >
            Close
          </button>
        </div>
      </div>

      <ul className="mt-3 space-y-1.5" lang={speech}>
        {ticket.lines.map((line, i) => (
          <li key={i} className="flex items-baseline gap-3 text-[1.375rem] font-bold leading-snug text-ink">
            <span className="tnum min-w-8">{line.qty}×</span>
            <span>
              {line.name}
              {ticket.lang && (
                <span lang="en" className="ml-2 text-[0.9375rem] font-normal text-ink-soft">
                  {line.english}
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
