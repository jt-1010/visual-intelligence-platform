import { LANGUAGES, type Lang } from './languages';

export type TicketLine = { qty: number; name: string };

export function namesInstructions(lang: Lang): string {
  const { name } = LANGUAGES[lang];
  return [
    `Translate restaurant menu item names into ${name} for a kitchen ticket.`,
    'Reply with a JSON array of strings only: one translation per input name, same order, same length.',
    'Keep brand names and sizes; never add or drop items.',
  ].join('\n');
}

//Parses the model's JSON array; null when it is not exactly one string per name.
export function parseNames(raw: string, expected: number): string[] | null {
  const match = /\[[\s\S]*\]/.exec(raw);
  if (!match) return null;
  try {
    const parsed: unknown = JSON.parse(match[0]);
    if (!Array.isArray(parsed) || parsed.length !== expected) return null;
    if (!parsed.every((s) => typeof s === 'string' && s.trim())) return null;
    return parsed.map((s: string) => s.trim());
  } catch {
    return null;
  }
}

// Quantities come from the order, never from the model.
export function ticketText(lines: TicketLine[], lang: Lang | null): string {
  if (lang === 'es') {
    return `Pedido para cocina: ${lines.map((l) => `${l.qty} ${l.name}`).join(', ')}.`;
  }
  if (lang === 'zh') {
    return `厨房订单：${lines.map((l) => `${l.name} ${l.qty}份`).join('，')}。`;
  }
  return `Kitchen order: ${lines.map((l) => `${l.qty} ${l.name}`).join(', ')}.`;
}
