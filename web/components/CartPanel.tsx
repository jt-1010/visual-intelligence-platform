'use client';

import { formatMoney } from '@/lib/money';

export type CartLine = {
  lineId: number;
  slug: string;
  name: string;
  qty: number;
  unitPriceCents: number;
  modifiers: { slug: string; name: string; priceDeltaCents: number }[];
  lineTotalCents: number;
};

export type Cart = {
  lines: CartLine[];
  subtotalCents: number;
  taxCents: number;
  totalCents: number;
  itemCount: number;
};

type Props = {
  cart: Cart | null;
  onConfirm: () => void;
  onClear: () => void;
  busy: boolean;
};

/**
 * The order, permanently on screen.
 *
 * Previously this appeared only after the first item was added, tucked inside
 * the conversation. That is backwards for a till: the running total is the one
 * thing a customer checks repeatedly, and for a Deaf customer it is the
 * receipt they are reading in place of hearing it read back.
 *
 * The aria-live region matters just as much in the other direction: a blind
 * customer has no panel to glance at, so this is where additions are announced.
 */
export function CartPanel({ cart, onConfirm, onClear, busy }: Props) {
  const lines = cart?.lines ?? [];
  const empty = lines.length === 0;

  return (
    <aside
      aria-label="Your order"
      // flex-1, not h-full: this panel now shares its column with the menu, and
      // h-full resolved to the whole column and squeezed the menu to nothing.
      className="flex min-h-0 flex-1 flex-col rounded-panel border border-line bg-card"
    >
      <div className="flex items-baseline justify-between border-b border-line px-6 py-5">
        <h2 className="text-[1.375rem] font-bold tracking-[-0.01em]">Your order</h2>
        {!empty && (
          <button
            type="button"
            onClick={onClear}
            className="rounded-control px-2 py-1 text-[0.9375rem] text-ink-soft underline-offset-4 transition hover:text-ink hover:underline"
          >
            Start over
          </button>
        )}
      </div>

      <div aria-live="polite" aria-atomic="false" className="min-h-0 flex-1 overflow-y-auto px-6">
        {empty ? (
          <p className="py-8 text-[1.0625rem] leading-relaxed text-ink-soft">
            Nothing yet. Tell us what you&rsquo;d like and it will appear here.
          </p>
        ) : (
          <ul className="divide-y divide-line">
            {lines.map((line) => (
              <li key={line.lineId} className="flex items-start gap-4 py-4">
                <span
                  aria-hidden="true"
                  className="tnum mt-0.5 min-w-8 shrink-0 rounded-control bg-sunk px-2 py-0.5 text-center text-[0.9375rem] font-bold text-ink-soft"
                >
                  {line.qty}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[1.0625rem] font-bold leading-snug">
                    <span className="sr-only">{line.qty} × </span>
                    {line.name}
                  </span>
                  {line.modifiers.length > 0 && (
                    <span className="mt-0.5 block text-[0.9375rem] leading-snug text-ink-soft">
                      {line.modifiers.map((m) => m.name).join(', ')}
                    </span>
                  )}
                </span>
                <span className="tnum shrink-0 text-[1.0625rem] font-bold">
                  {formatMoney(line.lineTotalCents)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t border-line px-6 py-5">
        <dl className="space-y-1.5">
          <div className="flex justify-between text-[1rem] text-ink-soft">
            <dt>Subtotal</dt>
            <dd className="tnum">{formatMoney(cart?.subtotalCents ?? 0)}</dd>
          </div>
          <div className="flex justify-between text-[1rem] text-ink-soft">
            <dt>Tax</dt>
            <dd className="tnum">{formatMoney(cart?.taxCents ?? 0)}</dd>
          </div>
          <div className="flex items-baseline justify-between pt-2.5 text-[1.75rem] font-bold tracking-[-0.02em]">
            <dt>Total</dt>
            <dd className="tnum">{formatMoney(cart?.totalCents ?? 0)}</dd>
          </div>
        </dl>

        <button
          type="button"
          onClick={onConfirm}
          disabled={empty || busy}
          className="mt-5 min-h-14 w-full rounded-control bg-action px-6 text-[1.125rem] font-bold text-white transition hover:bg-action-hover disabled:cursor-not-allowed disabled:bg-line-strong disabled:text-ink-faint"
        >
          {empty ? 'Add something first' : 'Confirm order'}
        </button>
      </div>
    </aside>
  );
}
