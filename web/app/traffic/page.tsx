import { TrafficSession } from '@/components/TrafficSession';
import { ThemeToggle } from '@/components/ThemeToggle';

export const dynamic = 'force-dynamic';

export default function TrafficPage() {
  return (
    <main className="mx-auto min-h-screen max-w-[100rem] px-6 py-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-bold tracking-tight text-ink">Traffic Control</h1>
        <div className="flex flex-wrap items-center gap-4">
          <p className="text-sm text-ink-soft">Generative reasoning for adaptive signal timing</p>
          <ThemeToggle />
        </div>
      </header>

      <TrafficSession />
    </main>
  );
}
