/**
 * Run one ordering scenario through an LLM backend (ollama or gateway).
 */

import { generateText, isStepCount } from 'ai';
import { clearCart } from '@/lib/agent/cart';
import { getModel, type Backend } from '@/lib/agent/model';
import { SYSTEM_PROMPT } from '@/lib/agent/prompt';
import { buildTools } from '@/lib/agent/tools';

export async function runLlmScenario(
  backend: Backend,
  sessionId: string,
  turns: string[],
): Promise<{ reply: string; ms: number; toolCalls: number; error?: string }> {
  await clearCart(sessionId);
  const started = performance.now();
  const model = getModel(backend);
  const tools = buildTools(sessionId);

  const messages: { role: 'user' | 'assistant'; content: string }[] = [];
  let reply = '';
  let toolCalls = 0;

  try {
    for (const turn of turns) {
      messages.push({ role: 'user', content: turn });
      const result = await generateText({
        model,
        system: SYSTEM_PROMPT,
        messages,
        tools,
        // Ordering routinely needs search -> add -> read-back in one turn.
        stopWhen: isStepCount(8),
      });
      reply = result.text ?? '';
      toolCalls += result.steps?.reduce((n, s) => n + (s.toolCalls?.length ?? 0), 0) ?? 0;
      messages.push({ role: 'assistant', content: reply || '(no text)' });
    }
    return { reply, ms: Math.round(performance.now() - started), toolCalls };
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    return {
      reply,
      ms: Math.round(performance.now() - started),
      toolCalls,
      error,
    };
  }
}

export async function backendAvailable(backend: Backend): Promise<{ ok: boolean; reason?: string }> {
  if (backend === 'ollama') {
    const base = (process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434/v1').replace(/\/v1\/?$/, '');
    try {
      const res = await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(2000) });
      if (!res.ok) return { ok: false, reason: `Ollama HTTP ${res.status}` };
      const model = process.env.OLLAMA_MODEL ?? 'qwen2.5:3b-instruct';
      const data = (await res.json()) as { models?: { name: string }[] };
      const names = (data.models ?? []).map((m) => m.name);
      const found = names.some((n) => n === model || n.startsWith(`${model}:`) || n.startsWith(model.split(':')[0]));
      if (!found) {
        return {
          ok: false,
          reason: `Ollama running but missing model "${model}". Run: ollama pull ${model}`,
        };
      }
      return { ok: true };
    } catch {
      return { ok: false, reason: 'Ollama not reachable (install + ollama serve)' };
    }
  }

  // Gateway: key may live in env or the local Cursor/Vercel login — try a probe
  // only when explicitly required. Absence of AI_GATEWAY_API_KEY is a soft skip.
  if (!process.env.AI_GATEWAY_API_KEY && !process.env.VERCEL_OIDC_TOKEN) {
    return {
      ok: false,
      reason: 'No AI_GATEWAY_API_KEY (set in web/.env.local to include gateway)',
    };
  }
  return { ok: true };
}
