export const LANGUAGES = {
  es: { name: 'Spanish', native: 'Español', speech: 'es-ES' },
  zh: { name: 'Simplified Chinese', native: '中文', speech: 'zh-CN' },
} as const;

export type Lang = keyof typeof LANGUAGES;

export function isLang(value: unknown): value is Lang {
  return typeof value === 'string' && Object.hasOwn(LANGUAGES, value);
}

// Few-shot pairs keep a 3B model from adding items or explaining the gloss.
const EXAMPLES: Record<Lang, [string, string][]> = {
  es: [
    ['want hamburger two', 'Quiero dos hamburguesas.'],
    ['coffee hot big please', 'Un café caliente grande, por favor.'],
    ['frenchfries no salt', 'Papas fritas sin sal.'],
  ],
  zh: [
    ['want hamburger two', '我要两个汉堡。'],
    ['coffee hot big please', '请给我一杯大杯热咖啡。'],
    ['frenchfries no salt', '薯条不要加盐。'],
  ],
};

export function translationInstructions(lang: Lang): string {
  const { name } = LANGUAGES[lang];
  const examples = EXAMPLES[lang].map(([g, t]) => `Gloss: ${g}\n${name}: ${t}`).join('\n\n');
  return [
    `You translate American Sign Language gloss into natural ${name}.`,
    'The gloss is the sequence of signs a customer made at a restaurant counter, in signing order, without grammar words.',
    'Compound signs are written as one word: frenchfries, icecream, hotdog, thankyou. "pop" means soda.',
    'Speak as the customer, in the first person. Translate only what was signed: never add items, sizes or quantities.',
    `Reply with the ${name} sentence only. No quotes, no notes, no romanization.`,
    '',
    examples,
  ].join('\n');
}

export function translationPrompt(gloss: string, lang: Lang): string {
  return `Gloss: ${gloss.trim().toLowerCase()}\n${LANGUAGES[lang].name}:`;
}

/** Small models still echo labels or wrap the answer in quotes; keep the sentence only. */
export function cleanTranslation(raw: string, lang: Lang): string {
  const label = new RegExp(`^\\s*(${LANGUAGES[lang].name}|Translation)\\s*[:：]\\s*`, 'i');
  const firstLine = raw.trim().split(/\r?\n/).find((l) => l.trim()) ?? '';
  return firstLine
    .replace(label, '')
    .trim()
    .replace(/^["'“”「『]+|["'“”」』]+$/g, '')
    .trim();
}
