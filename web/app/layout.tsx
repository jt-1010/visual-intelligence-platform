import type { Metadata } from 'next';
import { Atkinson_Hyperlegible } from 'next/font/google';
import './globals.css';

/*
  Atkinson Hyperlegible, from the Braille Institute. Its letterforms are drawn
  so that characters people most often confuse -- I l 1, O 0, c e -- stay
  distinct at small sizes and low acuity. On an ordering terminal whose whole
  purpose is access, the typeface is part of the accessibility work rather
  than a style preference.
*/
const atkinson = Atkinson_Hyperlegible({
  weight: ['400', '700'],
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-atkinson',
});

export const metadata: Metadata = {
  title: 'SignOrder',
  description: 'Order food by sign language, voice, typing, or touch.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={atkinson.variable}>
      {/*
        suppressHydrationWarning is here for browser extensions, not for our own
        markup. Grammarly and similar tools write attributes onto <body>
        (data-gr-ext-installed, data-new-gr-c-s-check-loaded) before React
        hydrates, which React then reports as a server/client mismatch we cannot
        fix and did not cause.

        It suppresses mismatches on THIS element only, one level deep, so a real
        mismatch anywhere inside the app is still reported.
      */}
      <body suppressHydrationWarning className="min-h-screen bg-paper text-ink">
        {children}
      </body>
    </html>
  );
}
