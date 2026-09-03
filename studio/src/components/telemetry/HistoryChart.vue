<script setup lang="ts">
// Time-series history chart (Apache ECharts, Apache-2.0). Smooth area line with
// a gradient fill + hover tooltip, theme-aware off the design tokens. Tree-shaken
// ECharts core so only the line chart + grid/tooltip ship.
import { computed, shallowRef } from 'vue'
import { watchThrottled } from '@vueuse/core'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useTheme } from '@/composables/useTheme'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent])

const props = withDefaults(
  defineProps<{ times: number[]; values: number[]; unit: string; max?: number }>(),
  { max: 0 },
)

const { theme } = useTheme()

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
function withAlpha(c: string, a: number): string {
  const s = c.trim()
  if (s.startsWith('#')) {
    let h = s.slice(1)
    if (h.length === 3)
      h = h
        .split('')
        .map((x) => x + x)
        .join('')
    const r = parseInt(h.slice(0, 2), 16)
    const g = parseInt(h.slice(2, 4), 16)
    const b = parseInt(h.slice(4, 6), 16)
    return `rgba(${r},${g},${b},${a})`
  }
  if (s.startsWith('rgb(')) return s.replace('rgb(', 'rgba(').replace(')', `,${a})`)
  return s
}

// The WS feed pushes every 400ms; redrawing a 900-point smoothed area per
// push held a renderer at ~20% of a core. The chart is a
// TREND view - 1 update/second loses nothing a human can see, and the
// gauges/engine strip stay on the live cadence. shallowRef: the array is
// replaced wholesale, never mutated.
const data = shallowRef<[number, number][]>([])
watchThrottled(
  () => props.times.length,
  () => {
    const n = Math.min(props.times.length, props.values.length)
    const t = props.times.slice(-n)
    const v = props.values.slice(-n)
    data.value = t.map((tt, i) => [Math.round(tt * 1000), v[i]])
  },
  { throttle: 1000, leading: true, trailing: true },
)

// Colours keyed on the THEME, not on every option rebuild: getComputedStyle
// is a forced style recalc, and six of them per data tick added layout work
// to every redraw.
const colors = computed(() => {
  void theme.value // recompute on theme toggle
  return {
    accent: cssVar('--pk-accent') || '#6366f1',
    grid: cssVar('--pk-border-subtle') || 'rgba(128,128,128,0.16)',
    muted: cssVar('--pk-text-muted') || '#888',
    elevated: cssVar('--pk-bg-elevated') || '#1b1b1b',
    strong: cssVar('--pk-border-strong') || '#333',
    primary: cssVar('--pk-text-primary') || '#fff',
  }
})

const option = computed(() => {
  const { accent, grid, muted, elevated, strong, primary } = colors.value
  const unit = props.unit
  return {
    animation: false,
    grid: { left: 4, right: 10, top: 10, bottom: 4, containLabel: true },
    tooltip: {
      trigger: 'axis',
      backgroundColor: elevated,
      borderColor: strong,
      borderWidth: 1,
      padding: [6, 10],
      textStyle: { color: primary, fontSize: 12 },
      axisPointer: { type: 'line', lineStyle: { color: muted, width: 1, type: 'dashed' } },
      formatter: (params: { value: [number, number] }[]) => {
        const p = params[0]
        const d = new Date(p.value[0])
        const hms = d.toLocaleTimeString()
        const val = Math.round(p.value[1] * 10) / 10
        return `<span style="color:${muted};font-size:11px">${hms}</span><br/><b>${val}</b> ${unit}`
      },
    },
    xAxis: {
      type: 'time',
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { color: muted, fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: props.max && props.max > 0 ? props.max : undefined,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 10 },
      splitLine: { lineStyle: { color: grid, type: 'dashed' } },
    },
    series: [
      {
        type: 'line',
        smooth: true,
        showSymbol: false,
        // downsample to pixel density before the smooth pass - 900 points
        // into a ~300px panel is 3x wasted bezier work
        sampling: 'lttb',
        data: data.value,
        lineStyle: { color: accent, width: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(accent, 0.35) },
              { offset: 1, color: withAlpha(accent, 0.02) },
            ],
          },
        },
      },
    ],
  }
})
</script>

<template>
  <VChart class="hchart" :option="option" autoresize />
</template>

<style scoped>
.hchart {
  width: 100%;
  height: 150px;
}
</style>
