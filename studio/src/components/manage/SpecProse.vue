<script setup lang="ts">
// Renders the tiny markdown subset the runner's OpenAPI descriptions use -
// paragraphs, "- " bullet lists, `code`, **bold** - as real elements. No
// v-html anywhere in the Studio, and this page introduces no exception: the
// document crosses a relay, so it gets parsed, never injected.
import { computed } from 'vue'

interface Seg {
  t: 'text' | 'code' | 'strong'
  s: string
}
interface Block {
  kind: 'p' | 'ul'
  items: Seg[][]
}

const props = defineProps<{
  text?: string | null
  /** render as one <span> (for table cells) instead of block elements. */
  inline?: boolean
}>()

function segs(line: string): Seg[] {
  const out: Seg[] = []
  const re = /`([^`]+)`|\*\*([^*]+)\*\*/g
  let last = 0
  for (let m = re.exec(line); m; m = re.exec(line)) {
    if (m.index > last) out.push({ t: 'text', s: line.slice(last, m.index) })
    out.push(m[1] !== undefined ? { t: 'code', s: m[1] } : { t: 'strong', s: m[2] ?? '' })
    last = m.index + m[0].length
  }
  if (last < line.length) out.push({ t: 'text', s: line.slice(last) })
  return out
}

const blocks = computed<Block[]>(() => {
  const text = props.text?.trim()
  if (!text) return []
  if (props.inline) return [{ kind: 'p', items: [segs(text)] }]
  return text.split(/\n{2,}/).map((b) => {
    const lines = b
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
    if (lines.length && lines.every((l) => l.startsWith('- ')))
      return { kind: 'ul' as const, items: lines.map((l) => segs(l.slice(2))) }
    return { kind: 'p' as const, items: [segs(lines.join(' '))] }
  })
})
</script>

<template>
  <template v-for="(b, i) in blocks" :key="i">
    <ul v-if="b.kind === 'ul'">
      <li v-for="(li, j) in b.items" :key="j">
        <template v-for="(s, k) in li" :key="k">
          <code v-if="s.t === 'code'">{{ s.s }}</code>
          <strong v-else-if="s.t === 'strong'">{{ s.s }}</strong>
          <template v-else>{{ s.s }}</template>
        </template>
      </li>
    </ul>
    <component :is="inline ? 'span' : 'p'" v-else>
      <template v-for="(s, k) in b.items[0]" :key="k">
        <code v-if="s.t === 'code'">{{ s.s }}</code>
        <strong v-else-if="s.t === 'strong'">{{ s.s }}</strong>
        <template v-else>{{ s.s }}</template>
      </template>
    </component>
  </template>
</template>
