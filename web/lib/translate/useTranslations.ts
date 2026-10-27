'use client';

import { useEffect, useRef, useState } from 'react';
import type { Turn } from '@/lib/interaction/transcript';
import type { Lang } from './languages';

export type Translation = { lang: Lang; text: string } | { lang: Lang; failed: true };

/**
 * Translates every signed turn into `lang`, once per turn and language.
 * Returns translations keyed by turn id for the current language only.
 */
export function useTranslations(
  turns: Turn[],
  lang: Lang | null,
  onTranslated?: (turnId: string, text: string, lang: Lang) => void,
): Record<string, Translation> {
  const [results, setResults] = useState<Record<string, Translation>>({});
  // Keys already requested, so re-renders never refetch.
  const requested = useRef(new Set<string>());
  const onTranslatedRef = useRef(onTranslated);
  useEffect(() => {
    onTranslatedRef.current = onTranslated;
  });

  useEffect(() => {
    if (!lang) return;
    for (const turn of turns) {
      if (turn.channel !== 'sign') continue;
      const key = `${turn.id}:${lang}`;
      if (requested.current.has(key)) continue;
      requested.current.add(key);

      fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: turn.text, lang }),
      })
        .then(async (res) => {
          const data = (await res.json()) as { translation?: string };
          if (!res.ok || !data.translation) throw new Error('translation failed');
          setResults((r) => ({ ...r, [key]: { lang, text: data.translation! } }));
          onTranslatedRef.current?.(turn.id, data.translation, lang);
        })
        .catch(() => {
          setResults((r) => ({ ...r, [key]: { lang, failed: true } }));
        });
    }
  }, [turns, lang]);

  const current: Record<string, Translation> = {};
  if (lang) {
    for (const turn of turns) {
      const hit = results[`${turn.id}:${lang}`];
      if (hit) current[turn.id] = hit;
    }
  }
  return current;
}
