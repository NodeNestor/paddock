<script setup lang="ts">
// Reka-backed meter (the only fill bar in the app - reka-ui reuse rule). Every
// bar in the Studio was a bare width-% div before this, so none of them
// announced anything: ProgressRoot carries role="progressbar" plus
// valuemin/max/now/text, which is the whole reason to route through it.
// `label` becomes aria-valuetext - pass the sentence a screen reader should
// hear ("12.4 GB of 48 GB"), not the number it can already read.
import { computed } from 'vue'
import { ProgressIndicator, ProgressRoot } from 'reka-ui'

const props = withDefaults(
  defineProps<{
    value: number
    max?: number
    /** accent by default; warning/error are the "nearing the ceiling" tones. */
    tone?: 'accent' | 'warning' | 'error'
    /** track height: 4 / 6 / 8 / 24px. lg is the one that hosts a slot. */
    size?: 'xs' | 'sm' | 'md' | 'lg'
    /** draw the empty track's outline, so an idle gauge still reads as a gauge
     *  rather than a missing element. */
    bordered?: boolean
    /** spoken value; falls back to Reka's "N percent". */
    label?: string | undefined
      /** Live-telemetry mode: the value streams several times a second, so the
     *  width TRANSITION never finishes - a layout property perpetually
     *  animating held a compositor at ~10% of a core per open dock. Live
     *  bars snap; eased fills stay for one-shot progress (downloads). */
    live?: boolean
}>(),
  { max: 100, tone: 'accent', size: 'sm', bordered: false },
)

// A ceiling can genuinely be unknown - `power_limit_w ?? 0`, `mem_total ?? 0` -
// and ProgressRoot warns and silently substitutes 100 when max isn't positive.
// Treat "no ceiling" as an empty bar instead: it's the honest reading, and it
// keeps the console quiet.
const known = computed(() => props.max > 0)
const safeMax = computed(() => (known.value ? props.max : 100))
const pct = computed(() =>
  known.value ? Math.min(100, Math.max(0, (props.value / props.max) * 100)) : 0,
)
// Clamped for aria-valuenow: Reka warns past max, and an over-budget context
// meter legitimately reports more used than it has.
const clamped = computed(() =>
  known.value ? Math.min(props.max, Math.max(0, props.value)) : 0,
)
</script>

<template>
  <ProgressRoot
    :model-value="clamped"
    :max="safeMax"
    :get-value-text="label ? () => label : undefined"
    :class="['pk-progress', `pk-progress--${size}`, bordered && 'pk-progress--bordered']"
  >
    <ProgressIndicator
      :class="['pk-progress__fill', `pk-progress__fill--${tone}`, live && 'pk-progress__fill--live']"
      :style="{ width: `${pct}%`, minWidth: pct > 0 ? '4px' : '0' }"
    />
    <slot />
  </ProgressRoot>
</template>

<style scoped>
.pk-progress {
  position: relative;
  width: 100%;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  overflow: hidden;
}
.pk-progress--bordered {
  border: 1px solid var(--pk-border-default);
}
.pk-progress--xs {
  height: 4px;
  background: var(--pk-border-default);
}
.pk-progress--sm {
  height: 6px;
}
.pk-progress--md {
  height: 8px;
  background: var(--pk-bg-elevated);
}
.pk-progress--lg {
  height: 24px;
  border-radius: var(--pk-radius-sm);
}

.pk-progress__fill {
  height: 100%;
  border-radius: inherit;
  background: var(--pk-accent);
  transition: width 0.4s ease, background 0.3s ease;
}
.pk-progress__fill--warning {
  background: var(--pk-status-warning);
}
.pk-progress__fill--error {
  background: var(--pk-status-error);
}
.pk-progress__fill--live {
  transition: none;
}
@media (prefers-reduced-motion: reduce) {
  .pk-progress__fill {
    transition: none;
  }
}
</style>
