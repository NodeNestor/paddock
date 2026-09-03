<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref , watch } from 'vue'
import Icon from '@/components/Icon.vue'
import { useUpdatesStore } from '@/stores/updates'

const updates = useUpdatesStore()
const err = ref<string | null>(null)
let timer: number | undefined

// The 2s cadence exists to animate a DOWNLOAD - running it unconditionally
// polled the update endpoint every 2s for the lifetime of the manage page,
// one of a stack of overlapping timers that kept an idle dashboard at ~12%
// of a core. Fast only while a download is in flight;
// a quiet card re-checks on the page's own cadence (mount + the header's
// 15-minute refresh).
watch(
  () => updates.info?.download?.phase === 'running',
  (running) => {
    clearInterval(timer)
    timer = running ? window.setInterval(() => void updates.refresh(), 2000) : undefined
  },
  { immediate: true },
)
onMounted(() => {
  void updates.refresh()
})
onBeforeUnmount(() => clearInterval(timer))

const info = computed(() => updates.info)
const dl = computed(() => updates.info?.download ?? null)
const available = computed(() => info.value?.state === 'available')
const running = computed(() => dl.value?.phase === 'running')
const ready = computed(() => dl.value?.phase === 'ready')

/** What we are running, whatever state the check is in. */
const current = computed(() => info.value?.current ?? info.value?.version ?? '')

const pct = computed(() => {
  const d = dl.value
  if (!d?.total) return null
  return Math.min(100, Math.round((d.received / d.total) * 100))
})

function mb(n: number | null | undefined): string {
  if (!n) return ''
  return `${(n / 1048576).toFixed(1)} MB`
}

async function start(): Promise<void> {
  err.value = await updates.download()
}
</script>

<template>
  <section class="uc">
    <div class="uc__head"><h2>Version</h2></div>

    <p class="uc__sub">
      <template v-if="available">
        Paddock <strong>{{ info?.latest }}</strong> is available. You are running
        {{ current }}.
      </template>
      <template v-else-if="info?.state === 'current'">
        Paddock {{ current }} is the newest release.
      </template>
      <template v-else-if="info?.state === 'unknown'">
        Paddock {{ current }}. The release server could not be reached, so there may or
        may not be a newer version.
      </template>
      <template v-else>Checking...</template>
    </p>

    <!-- Release notes, verbatim from the publisher. Only when there is
         something to decide about. -->
    <pre v-if="available && info?.notes" class="uc__notes">{{ info.notes }}</pre>

    <!-- The one thing that must never be quiet: a release with no sha256 can be
         downloaded but not checked, and the person clicking deserves to know
         that BEFORE they click, not from a log file afterwards. -->
    <p v-if="available && info?.verifiable === false" class="uc__warn">
      <Icon name="warning" :size="14" />
      This release was published without a checksum, so the download cannot be verified
      beyond the transport.
    </p>

    <div v-if="running" class="uc__prog">
      <div class="uc__bar"><div class="uc__fill" :style="{ width: `${pct ?? 0}%` }" /></div>
      <span class="uc__count">
        {{ mb(dl?.received) }}<template v-if="dl?.total"> / {{ mb(dl.total) }}</template>
      </span>
      <button class="pk-btn pk-btn--sm" type="button" @click="updates.cancel">Cancel</button>
    </div>

    <template v-else-if="ready">
      <p class="uc__ready">
        <Icon name="check" :size="14" />
        {{ dl?.version }} downloaded and verified.
      </p>
      <!-- Deliberately not an "install" button. On Windows a running exe cannot
           be replaced, and the manager must never restart itself behind the
           user's back - so we say where the file is and let them choose. -->
      <p class="uc__path">
        <code>{{ dl?.path }}</code>
      </p>
      <p class="uc__sub">
        Close paddock, replace the folder contents with the ones in that archive, and
        start it again. Your data folder is untouched by this.
      </p>
    </template>

    <p v-else-if="dl?.phase === 'failed'" class="uc__warn">
      <Icon name="warning" :size="14" /> {{ dl.error }}
    </p>

    <button
      v-if="available && info?.downloadable && !running && !ready"
      class="pk-btn pk-btn--primary"
      type="button"
      :disabled="updates.busy"
      @click="start"
    >
      <Icon name="arrow-down" :size="15" />
      Download {{ info?.latest }}<template v-if="info?.size"> ({{ mb(info.size) }})</template>
    </button>

    <p v-if="err" class="uc__warn"><Icon name="warning" :size="14" /> {{ err }}</p>
  </section>
</template>

<style scoped>
.uc {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: flex-start;
}
.uc__head h2 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.uc__sub {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  margin: 0;
}
.uc__notes {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  background: var(--pk-bg-base);
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  padding: 12px;
  margin: 0;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  width: 100%;
}
.uc__warn,
.uc__ready {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-sm);
  margin: 0;
}
.uc__warn {
  color: var(--pk-status-warning);
}
.uc__ready {
  color: var(--pk-status-success, var(--pk-text-primary));
}
.uc__path {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  word-break: break-all;
}
.uc__prog {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}
.uc__bar {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: var(--pk-bg-base);
  overflow: hidden;
}
.uc__fill {
  height: 100%;
  background: var(--pk-accent);
  transition: width 0.2s ease;
}
.uc__count {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
</style>
