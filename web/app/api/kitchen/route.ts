import { generateText } from 'ai';
import { getCart } from '@/lib/agent/cart';
import { activeBackend, getModel, type Backend } from '@/lib/agent/model';
import { isLang, type Lang } from '@/lib/translate/languages';
import { namesInstructions, parseNames, ticketText } from '@/lib/translate/kitchen';

export const maxDuration = 30;

async function translateNames(names: string[], lang: Lang): Promise<string[] | null> {
  const backend = (process.env.TRANSLATE_BACKEND as Backend | undefined) ?? activeBackend();
  try {
    const result = await generateText({
      model: getModel(backend),
      instructions: namesInstructions(lang),
      prompt: JSON.stringify(names),
      temperature: 0,
      maxOutputTokens: 400,
    });
    return parseNames(result.text, names.length);
  } catch (error) {
    console.error(`[kitchen] backend=${backend} FAILED:`, error);
    return null;
  }
}

export async function POST(req: Request) {
  const { sessionId, lang } = (await req.json()) as { sessionId?: unknown; lang?: unknown };
  if (typeof sessionId !== 'string' || !sessionId) {
    return Response.json({ error: 'sessionId is required' }, { status: 400 });
  }
  const target = isLang(lang) ? lang : null;

  const cart = await getCart(sessionId);
  if (cart.lines.length === 0) {
    return Response.json({ error: 'the order is empty' }, { status: 400 });
  }

  const english = cart.lines.map((l) =>
    l.modifiers.length ? `${l.name} (${l.modifiers.map((m) => m.name).join(', ')})` : l.name,
  );
  // An untranslatable ticket still goes out in English rather than not at all.
  const names = (target && (await translateNames(english, target))) || english;
  const translated = names !== english;

  const lines = cart.lines.map((l, i) => ({ qty: l.qty, name: names[i], english: english[i] }));
  const ticketLang = translated ? target : null;
  console.log(`[kitchen] ${lines.length} lines lang=${ticketLang ?? 'en'} requested=${target ?? 'en'}`);

  return Response.json({ lines, lang: ticketLang, text: ticketText(lines, ticketLang) });
}
