'use client';

import { formatMoney } from '@/lib/money';
import { useEffect, useRef } from 'react';
import type { ConfirmedOrder } from '@/lib/interaction/transcript';

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
  confirmedOrder?: ConfirmedOrder | null;
  onNewOrder?: () => void;
  onSendToKitchen?: () => void;
  /** Shown on the kitchen button, e.g. "Español"; English when translation is off. */
  kitchenLanguage?: string;
  sendingToKitchen?: boolean;
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
export function CartPanel({
  cart,
  onConfirm,
  onClear,
  busy,
  confirmedOrder,
  onNewOrder,
  onSendToKitchen,
  kitchenLanguage = 'English',
  sendingToKitchen = false,
}: Props) {
  const lines = cart?.lines ?? [];
  const empty = lines.length === 0;
  const confirmationHeading = useRef<HTMLHeadingElement>(null);
  const confirmedOrderNumber = confirmedOrder?.orderNumber;
  useEffect(() => {
    if (confirmedOrderNumber !== undefined) confirmationHeading.current?.focus();
  }, [confirmedOrderNumber]);

  if (confirmedOrder) return (
    <aside aria-label="Order confirmation" className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-panel border border-action/30 bg-card">
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-8">
        <div className="text-center">
          <div className="animate-pop-in mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-action-soft text-action">
            <div className="animate-pulse-glow flex h-16 w-16 items-center justify-center rounded-full bg-action text-on-action">
              <svg aria-hidden="true" viewBox="0 0 48 48" className="h-11 w-11" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                <polyline className="check-icon-polyline" points="12 25 20 33 36 16" />
              </svg>
            </div>
          </div>
          <h2 ref={confirmationHeading} tabIndex={-1} className="text-[2rem] font-bold leading-tight tracking-[-0.03em]">Thanks for ordering!</h2>
          <p className="tnum mt-4 inline-block rounded-full bg-action-soft px-4 py-1.5 font-bold text-action">Order #{confirmedOrder.orderNumber}</p>
          <p className="mt-4 text-[1.0625rem] leading-relaxed text-ink-soft">Your food is being prepared right now.</p>
          <p className="mt-3 inline-block rounded-full bg-sunk px-3 py-1.5 text-sm text-ink-soft">No payment needed for demo</p>
        </div>
        <section aria-label="Receipt summary" className="animate-slide-up mt-7 rounded-panel border border-line bg-paper p-5" style={{ animationDelay: '160ms' }}>
          <h3 className="mb-3 text-sm font-bold uppercase tracking-wider text-ink-soft">Your receipt</h3>
          <ul className="divide-y divide-line">
            {confirmedOrder.items?.map((line, index) => (
              <li key={line.lineId} className="animate-slide-up flex items-start gap-3 py-3" style={{ animationDelay: `${240 + Math.min(index, 8) * 50}ms` }}>
                <span className="tnum font-bold text-action">{line.qty} ×</span>
                <span className="min-w-0 flex-1 text-sm font-bold">{line.name}
                  {line.modifiers.length > 0 && <span className="mt-1 block font-normal text-ink-soft">{line.modifiers.map((m) => m.name).join(', ')}</span>}
                </span>
                <span className="tnum text-sm font-bold">{formatMoney(line.lineTotalCents)}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex justify-between border-t border-line pt-4 text-xl font-bold"><span>Total</span><span className="tnum">{confirmedOrder.total}</span></div>
          <p className="mt-1 text-right text-xs text-ink-soft">Includes tax</p>
        </section>
        <p role="status" className="animate-slide-up mt-5 flex items-center justify-center gap-2 text-sm font-bold text-action" style={{ animationDelay: '320ms' }}>
          <span aria-hidden="true" className="h-2 w-2 rounded-full bg-action" />Sent to kitchen display
        </p>
      </div>
      <div className="shrink-0 border-t border-line p-6">
        <button type="button" onClick={onNewOrder} disabled={busy} className="cart-action min-h-14 w-full rounded-control bg-action px-5 text-lg font-bold text-on-action hover:bg-action-hover disabled:cursor-wait disabled:opacity-60">Start a new order</button>
      </div>
    </aside>
  );

  return (
    <aside
      aria-label="Your order"
      // flex-1, not h-full: this panel now shares its column with the menu, and
      // h-full resolved to the whole column and squeezed the menu to nothing.
      className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-panel border border-line bg-card"
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
              <li key={line.lineId} className="cart-line-item flex items-start gap-4 py-4">
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
          className="cart-action mt-5 min-h-14 w-full rounded-control bg-action px-6 text-[1.125rem] font-bold text-on-action hover:bg-action-hover disabled:cursor-not-allowed disabled:bg-line-strong disabled:text-ink-faint"
        >
          {empty ? 'Add something first' : 'Confirm order'}
        </button>

        {onSendToKitchen && (
          <button
            type="button"
            onClick={onSendToKitchen}
            disabled={empty || busy || sendingToKitchen}
            className="cart-action mt-3 min-h-12 w-full rounded-control border-2 border-sign bg-sign-soft px-6 text-[1.0625rem] font-bold text-sign hover:brightness-95 disabled:cursor-not-allowed disabled:border-line disabled:bg-sunk disabled:text-ink-faint"
          >
            {sendingToKitchen ? 'Sending…' : `Send to kitchen · ${kitchenLanguage}`}
          </button>
        )}
      </div>
    </aside>
  );
}
