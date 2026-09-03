// Streaming Markdown renderer config (markstream-vue + Shiki + KaTeX + Mermaid).
import { enableKatex, enableMermaid } from 'markstream-vue'
import 'markstream-vue/index.css'
import 'katex/dist/katex.min.css'
// Our dark-theme overrides for markstream's hardcoded-light blocks (after its
// stylesheet so they win).
import '@/styles/markstream-overrides.css'

// Grammars preloaded for code fences; anything else lazy-loads or falls back.
export const MD_LANGS = [
  'text', 'bash', 'shell', 'powershell', 'json', 'yaml', 'toml', 'ini',
  'python', 'rust', 'c', 'cpp', 'go', 'javascript', 'typescript', 'tsx', 'jsx',
  'vue', 'html', 'css', 'scss', 'sql', 'markdown', 'diff', 'dockerfile', 'make',
  'xml', 'java', 'csharp', 'kotlin', 'swift', 'ruby', 'php', 'lua', 'r',
]

let inited = false

/** Enable math + mermaid once, before the first render. Mermaid is lazy-loaded
 *  (a separate chunk + worker) only when a ```mermaid fence actually appears. */
export function initMarkstream(): void {
  if (inited) return
  inited = true
  enableKatex()
  enableMermaid(() => import('mermaid'))
}
