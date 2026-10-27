'use client';

import { LANGUAGES, type Lang } from '@/lib/translate/languages';

type Props = {
  value: Lang | null;
  onChange: (lang: Lang | null) => void;
};

const OPTIONS: { value: Lang | null; label: string }[] = [
  { value: null, label: 'Off' },
  ...(Object.keys(LANGUAGES) as Lang[]).map((l) => ({ value: l, label: LANGUAGES[l].native })),
];

export function LanguagePicker({ value, onChange }: Props) {
  return (
    <div role="radiogroup" aria-label="Translate signing into" className="flex shrink-0 flex-col gap-2">
      <p className="text-[0.9375rem] font-bold text-ink-soft">Translate signing into</p>
      <div className="flex gap-2">
        {OPTIONS.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.label}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(o.value)}
              className={[
                'min-h-12 flex-1 rounded-control border-2 px-3 text-[1rem] font-bold transition',
                active
                  ? 'border-action bg-action text-on-action'
                  : 'border-line-strong bg-card text-ink hover:border-ink-soft',
              ].join(' ')}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
