import { TrafficSession } from '@/components/TrafficSession';

export const dynamic = 'force-dynamic';

export default function TrafficPage() {
  return (
    <main className="mx-auto min-h-screen max-w-[100rem] px-6 py-6">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-bold tracking-tight text-white">Traffic Control</h1>
        <p className="text-sm text-slate-400">Generative reasoning for adaptive signal timing</p>
      </header>

      <TrafficSession />
    </main>
  );
}
