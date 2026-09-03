<script setup lang="ts">
// One web search in the assistant turn: the query the model searched for, and
// (expanded) the sources the provider returned.
import { computed, ref } from 'vue'
import type { WebSearchCall } from '@/types/chat'
import Icon from '@/components/Icon.vue'
import SearchLogo from '@/components/manage/SearchLogo.vue'
import { searchLabel, searchProvider } from '@/lib/websearch'

const props = defineProps<{ call: WebSearchCall }>()

const open = ref(false)
const isRunning = computed(
  () => props.call.status === 'in_progress' || props.call.status === 'searching',
)
const failed = computed(() => props.call.status === 'failed')
/** The engine's own mark replaces the generic globe once we know which one
 *  answered - but only for a provider this build actually has a mark for. A
 *  cloud lane's built-in search, or a provider added after this Studio was
 *  built, keeps the globe rather than rendering an empty box. */
const mark = computed(() =>
  props.call.provider && searchProvider(props.call.provider) ? props.call.provider : '',
)

function host(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}
</script>

<template>
  <div class="ws" :class="`ws--${call.status}`">
    <button class="ws__head" type="button" @click="open = !open">
      <span class="ws__icon">
        <Icon v-if="isRunning" name="spinner" :size="14" class="ws__spin" />
        <Icon v-else-if="failed" name="alert-triangle" :size="14" />
        <SearchLogo v-else-if="mark" :provider="mark" :size="14" />
        <Icon v-else name="globe" :size="14" />
      </span>
      <span class="ws__title">
        <span class="ws__verb">{{ isRunning ? 'Searching the web for' : 'Searched the web for' }}</span>
        <span class="ws__query">{{ call.query || '...' }}</span>
      </span>
      <span v-if="failed" class="ws__tag">error</span>
      <span v-else-if="call.sources.length" class="ws__count">
        {{ call.sources.length }} source{{ call.sources.length > 1 ? 's' : '' }}
      </span>
      <Icon name="chevron-down" :size="14" class="ws__chev" :class="{ 'ws__chev--open': open }" />
    </button>

    <div v-if="open" class="ws__body">
      <div v-if="failed" class="ws__err">{{ call.error || 'The search failed.' }}</div>
      <ol v-else-if="call.sources.length" class="ws__list">
        <li v-for="(s, i) in call.sources" :key="s.url" class="ws__item">
          <span class="ws__num">{{ i + 1 }}</span>
          <a :href="s.url" target="_blank" rel="noopener noreferrer" class="ws__link">
            <span class="ws__link-title">{{ s.title || host(s.url) }}</span>
            <span class="ws__link-host">{{ host(s.url) }}</span>
          </a>
        </li>
      </ol>
      <div v-else class="ws__none">No results.</div>
      <p v-if="mark" class="ws__via">
        <SearchLogo :provider="mark" :size="12" />
        <span>Searched with {{ searchLabel(mark) }}</span>
      </p>
    </div>
  </div>
</template>

<style scoped>
.ws {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.ws--failed {
  border-color: var(--pk-border-subtle);
}
/* Gutter scale, shared with ThinkingFold and ToolCall - see the note in
   ToolCall.vue. These stack in one strip above the answer and have to read as
   one family, so a change here is a change there. */
.ws__head {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 7px 10px;
  border: none;
  background: transparent;
  color: var(--pk-text-primary);
  cursor: pointer;
  text-align: left;
  font: inherit;
}
.ws__head:hover {
  background: var(--pk-bg-hover);
}
.ws__icon {
  display: inline-flex;
  color: var(--pk-text-muted);
  flex-shrink: 0;
}
.ws--completed .ws__icon {
  color: var(--pk-status-success);
}
.ws--failed .ws__icon {
  color: var(--pk-text-danger);
}
.ws__title {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  flex: 1;
}
.ws__verb {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  flex-shrink: 0;
}
.ws__query {
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ws__count {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  flex-shrink: 0;
}
.ws__tag {
  font-size: 10px;
  line-height: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 2px 6px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-danger-subtle);
  color: var(--pk-text-danger);
  flex-shrink: 0;
}
.ws__chev {
  color: var(--pk-text-muted);
  flex-shrink: 0;
  transition: transform 0.15s ease;
}
.ws__chev--open {
  transform: rotate(180deg);
}
.ws__spin {
  animation: ws-spin 0.8s linear infinite;
}
.ws__body {
  padding: 0 10px 10px;
}
/* Browser list markers render in the page's default font and clash with the
   styled rows - number the sources ourselves in the same mono/muted style as
   the host chips. */
.ws__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ws__item {
  display: flex;
  align-items: baseline;
  gap: 9px;
  min-width: 0;
}
.ws__num {
  flex: none;
  min-width: 12px;
  text-align: right;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-muted);
}
.ws__link {
  display: inline-flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
  text-decoration: none;
}
.ws__link:hover .ws__link-title {
  text-decoration: underline;
}
.ws__link-title {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-accent-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ws__link-host {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  flex-shrink: 0;
}
.ws__err {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-danger);
}
.ws__none {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* Attribution under the sources: which engine found them. Quiet enough to
   stay out of the way of the links, present enough that the results are never
   anonymous - the provider picked the ranking AND charged for it. */
.ws__via {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 10px 0 0;
  padding-top: 8px;
  border-top: 1px solid var(--pk-border-subtle);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
@keyframes ws-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
