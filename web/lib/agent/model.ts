import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { gateway, type LanguageModel } from 'ai';

/**
 * Two interchangeable LLM backends behind one call.
 *
 * `ollama`  - our QLoRA fine-tuned Qwen2.5-3B, merged and quantised, served
 *             locally. No external API, nothing to time out mid-demo.
 * `gateway` - a hosted frontier model, used as the baseline.
 *
 * Keeping both live is not indecision. The thesis needs a head-to-head on
 * tool-call accuracy and latency, and that benchmark only exists if the same
 * fixture suite can point at either backend by flipping one env var.
 */

export type Backend = 'ollama' | 'gateway';

export function activeBackend(): Backend {
  return (process.env.LLM_BACKEND as Backend) ?? 'gateway';
}

const ollama = createOpenAICompatible({
  name: 'ollama',
  baseURL: process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434/v1',
});

export function getModel(backend: Backend = activeBackend()): LanguageModel {
  if (backend === 'ollama') {
    return ollama(process.env.OLLAMA_MODEL ?? 'qwen2.5:3b-instruct');
  }
  return gateway(process.env.GATEWAY_MODEL ?? 'anthropic/claude-sonnet-4.5');
}
