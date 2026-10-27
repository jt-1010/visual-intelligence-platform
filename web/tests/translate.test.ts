import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  cleanTranslation,
  isLang,
  translationInstructions,
  translationPrompt,
} from '@/lib/translate/languages';

describe('isLang', () => {
  it('accepts the supported languages only', () => {
    assert.equal(isLang('es'), true);
    assert.equal(isLang('zh'), true);
    assert.equal(isLang('fr'), false);
    assert.equal(isLang('toString'), false);
    assert.equal(isLang(undefined), false);
  });
});

describe('translation prompt', () => {
  it('names the target language and ends on its label for the model to complete', () => {
    assert.match(translationInstructions('zh'), /Simplified Chinese/);
    assert.equal(translationPrompt(' WANT Coffee ', 'es'), 'Gloss: want coffee\nSpanish:');
  });
});

describe('cleanTranslation', () => {
  it('strips echoed labels and quotes', () => {
    assert.equal(cleanTranslation('Spanish: "Quiero un café."', 'es'), 'Quiero un café.');
    assert.equal(cleanTranslation('「我要两个汉堡。」', 'zh'), '我要两个汉堡。');
    assert.equal(cleanTranslation('Simplified Chinese：我要咖啡。', 'zh'), '我要咖啡。');
  });

  it('keeps only the first sentence line when the model adds notes', () => {
    assert.equal(
      cleanTranslation('\nQuiero dos hamburguesas.\n(Note: "two" was signed after the noun.)', 'es'),
      'Quiero dos hamburguesas.',
    );
  });
});
