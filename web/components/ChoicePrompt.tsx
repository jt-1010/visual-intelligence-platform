'use client';

import type { PendingChoice } from '@/lib/interaction/transcript';

type Props = {
  choice: PendingChoice;
  onPick: (optionName: string) => void;
  disabled: boolean;
};

/**
 * "Which one?", as buttons rather than a sentence.
 *
 * Being asked to choose is the normal shape of this conversation, not a repair
 * for something going wrong. Sign recognition knows a small vocabulary, so a
 * signer can say BURGER and cannot say "Double Quarter Pounder with Cheese" --
 * the terminal has to close that gap, every time, for most of the menu.
 *
 * Every name and price here comes from the tool result, which read them from
 * the database. That is deliberate: the model has been observed claiming it
 * added two burgers for $11.98 when nothing was added and no such price
 * existed. Whatever it says above, these buttons are the truth and they work.
 *
 * Answering by pressing one is also simply faster than signing it, and a person
 * who has just been told their signing was ambiguous should not have to sign
 * the harder word to recover.
 */
export function ChoicePrompt({ choice, onPick, disabled }: Props) {
  const more = choice.totalMatches - choice.options.length;

  return (
    <section
      aria-label="Choose an item"
      className="shrink-0 rounded-panel border border-attention bg-attention-soft px-5 py-4"
    >
      {/*
        Deliberately not "Which burger?". The word comes from what the person
        said, and naive singularising turns "fries" into "frie" and "chicken"
        into something worse. The options directly below say what is being
        chosen far better than the heading could.
      */}
      <p className="text-[1.0625rem] font-bold text-attention">
        {choice.quantity > 1 ? `Which one? You asked for ${choice.quantity}.` : 'Which one?'}
      </p>

      <ul className="mt-3 flex flex-wrap gap-2">
        {choice.options.map((option) => (
          <li key={option.name}>
            <button
              type="button"
              onClick={() => onPick(option.name)}
              disabled={disabled}
              className="min-h-12 rounded-control border border-line-strong bg-card px-4 text-left text-[1rem] font-bold text-ink transition hover:border-action hover:bg-action-soft disabled:cursor-not-allowed disabled:opacity-40"
            >
              {option.name}
              <span className="tnum ml-2 font-normal text-ink-soft">{option.price}</span>
            </button>
          </li>
        ))}
      </ul>

      {more > 0 && (
        <p className="mt-2.5 text-[0.9375rem] text-ink-soft">
          {more} more on the menu — scroll it and tap any item.
        </p>
      )}
    </section>
  );
}
