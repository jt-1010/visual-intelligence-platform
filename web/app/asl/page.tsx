import { randomUUID } from 'node:crypto';
import Link from 'next/link';
import { SessionShell } from '@/components/SessionShell';

// Rendered fresh per request, so each page load is a genuinely new session
// and never resurrects a previous customer's cart.
export const dynamic = 'force-dynamic';

export default async function OrderPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  // ?tune=1 reveals the camera calibration instruments. Read on the server so
  // there is no flash of developer UI before the client decides to hide it.
  const tuning = (await searchParams).tune === '1';

  return (
    <div className="mx-auto flex h-dvh max-w-[112rem] flex-col px-6 pb-6">
      {/*
        A till, not a product page. The header earns its height by holding the
        name and a way out, and nothing else -- every pixel it takes is a pixel
        off the conversation.
      */}
      <header className="flex h-[4.5rem] shrink-0 items-center justify-between">
        <Link
          href="/"
          className="flex items-baseline gap-2.5 rounded-control text-[1.25rem] font-bold tracking-[-0.015em]"
        >
          SignOrder
          <span className="text-[0.9375rem] font-normal text-ink-soft">Order here</span>
        </Link>

        {tuning && (
          <span className="rounded-control bg-attention-soft px-3 py-1 text-[0.875rem] font-bold text-attention">
            Calibration view
          </span>
        )}
      </header>

      <SessionShell baseSessionId={randomUUID()} tuning={tuning} />
    </div>
  );
}
