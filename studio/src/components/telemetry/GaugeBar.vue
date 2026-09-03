<script setup lang="ts">
// Thin glanceable fill bar for a live metric. The track and the aria wiring are
// ui/Progress; what stays here is the only thing that is telemetry's own - the
// thresholds that shift colour as a value nears its ceiling (temp warms -> red;
// power ambers near the limit).
import { computed } from 'vue'
import Progress from '@/components/ui/Progress.vue'

const props = withDefaults(
  defineProps<{
    value: number
    max: number
    tone?: 'accent' | 'temp' | 'power'
    /** spoken value with its unit - "62 degrees", not "62 percent". */
    label?: string | undefined
  }>(),
  { tone: 'accent' },
)

const pct = computed(() =>
  props.max > 0 ? Math.min(100, Math.max(0, (props.value / props.max) * 100)) : 0,
)

const tone = computed<'accent' | 'warning' | 'error'>(() => {
  if (props.tone === 'temp') {
    if (pct.value >= 85) return 'error'
    if (pct.value >= 65) return 'warning'
  } else if (props.tone === 'power' && pct.value >= 90) {
    return 'warning'
  }
  return 'accent'
})
</script>

<template>
  <Progress :value="value" :max="max" :tone="tone" :label="label" size="sm" bordered live />
</template>
