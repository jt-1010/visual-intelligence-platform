import { generateText } from 'ai';
import { activeBackend, getModel, type Backend } from '@/lib/agent/model';
import {
  cleanTranslation,
  isLang,
  translationInstructions,
  translationPrompt,
} from '@/lib/translate/languages';

export const maxDuration = 30;

export async function POST(req: Request) {
  const started = performance.now();
  const { text, lang } = (await req.json()) as { text?: unknown; lang?: unknown };

  if (typeof text !== 'string' || !text.trim()) {
    return Response.json({ error: 'text is required' }, { status: 400 });
  }
  if (!isLang(lang)) {
    return Response.json({ error: 'unsupported language' }, { status: 400 });
  }

  // TRANSLATE_BACKEND lets translation run on local Qwen while ordering uses another backend.
  const backend = (process.env.TRANSLATE_BACKEND as Backend | undefined) ?? activeBackend();

  try {
    const result = await generateText({
      model: getModel(backend),
      instructions: translationInstructions(lang),
      prompt: translationPrompt(text, lang),
      temperature: 0,
      maxOutputTokens: 120,
    });
    const translation = cleanTranslation(result.text, lang);
    console.log(
      `[translate] backend=${backend} lang=${lang} ${Math.round(performance.now() - started)}ms`,
    );
    if (!translation) {
      return Response.json({ error: 'empty translation' }, { status: 502 });
    }
    return Response.json({ translation, lang });
  } catch (error) {
    console.error(`[translate] backend=${backend} FAILED:`, error);
    return Response.json({ error: 'translation failed' }, { status: 502 });
  }
}
