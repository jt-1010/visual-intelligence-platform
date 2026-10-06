import Link from 'next/link';

/**
 * The project's front door — for the team, the advisor, and demos.
 *
 * Deliberately not styled like the ordering terminal. This page explains a
 * research project; /asl is a till a customer stands at. Collapsing the two
 * would make the terminal read as a portfolio piece and the project read as
 * a product, and neither is true.
 */
export default function LandingPage() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-5xl flex-col justify-center px-6 py-16">
      <p className="text-[1rem] text-ink-soft">San José State University · CMPE 295A</p>

      <h1 className="mt-4 max-w-[18ch] text-balance text-[clamp(2.25rem,5.5vw,3.75rem)] font-bold leading-[1.05] tracking-[-0.03em]">
        One vision pipeline, two problems worth solving.
      </h1>

      <p className="mt-6 max-w-[62ch] text-[1.1875rem] leading-relaxed text-ink-soft">
        Both systems below read continuous video, model how it changes over time, and hand the
        result to a language model that may phrase an answer but never invent a fact. A price
        comes from the database; a signal plan is checked against safe timing limits before
        anyone sees it.
      </p>

      <div className="mt-12 grid gap-4 sm:grid-cols-2">
        <Link
          href="/asl"
          className="group rounded-panel border border-line bg-card p-7 transition hover:border-action"
        >
          <h2 className="text-[1.375rem] font-bold tracking-[-0.015em]">Ordering terminal</h2>
          <p className="mt-2.5 text-[1.0625rem] leading-relaxed text-ink-soft">
            Order food in American Sign Language, speech, typing, or touch. Recognises 64 signs
            at 88.6% accuracy on signers it has never seen.
          </p>
          <span className="mt-5 inline-block text-[1rem] font-bold text-action">
            Open the terminal
          </span>
        </Link>

        <Link
          href="/traffic"
          className="group rounded-panel border border-line bg-card p-7 transition hover:border-action"
        >
          <h2 className="text-[1.375rem] font-bold tracking-[-0.015em]">Signal timing</h2>
          <p className="mt-2.5 text-[1.0625rem] leading-relaxed text-ink-soft">
            Count vehicles from intersection video, forecast the next few minutes, and recommend
            a signal plan with a plain-language reason. Design stage.
          </p>
          <span className="mt-5 inline-block text-[1rem] font-bold text-action">
            Open the dashboard
          </span>
        </Link>
      </div>
    </main>
  );
}
