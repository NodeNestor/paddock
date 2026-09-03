<script setup lang="ts">
// Where a photo was taken, drawn from an outline that ships with the app
// (layer 2).
//
// The obvious way to show a coordinate is a tile server, and the obvious way
// is wrong here: every tile request tells a third party where the user's
// photos were taken, for the crime of glancing at their own file. So the
// DEFAULT view contacts nothing - coastlines and borders come from a bundled
// Natural Earth outline, the place name from the offline table in the
// manager, and the whole thing works on a plane. The real slippy map is a
// deliberate act, one button away, and it names the host it will call.
//
// The map answers a different question from the text under it. "in Arezzo
// (Tuscany, Italy)" is the fact; the pin is the glance - which continent,
// how far inland, near which coast - and neither substitutes for the other.
//
// LAYER 3 lives here too: "Open map" swaps the drawing for a real MapLibre
// slippy map on OpenStreetMap tiles. It is one click, never automatic, the
// host is named next to the button before it is called, and the tile URL is
// a setting - see the `mapTiles` note in stores/settings.ts for why that is
// a user's decision rather than ours.
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { copyText } from '@/lib/clipboard'
import { attributionFor, isRasterTemplate, tileHost, tileTemplate } from '@/lib/maptiles'
import { useSettingsStore } from '@/stores/settings'
import type { FileLocation } from '@/lib/api'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import ToggleGroup from '@/components/ui/ToggleGroup.vue'
import ToggleGroupItem from '@/components/ui/ToggleGroupItem.vue'

const props = defineProps<{
  location: FileLocation
  /** The thread's minimap: a small locator under an attached photo, with the
   *  place name and nothing else. Everything the compact form drops is
   *  available one click away in the document pane, and a chat message is not
   *  the place for a scale bar and a coordinate readout. */
  compact?: boolean
}>()

// ~37 KB gzipped of path data that only a geotagged photo ever needs, so it
// arrives as its own chunk on the first one instead of in everyone's bundle.
const outline = shallowRef<{ LAND: readonly string[]; BORDERS: readonly string[] } | null>(null)
void import('@/assets/world-outline').then((m) => {
  outline.value = m
})

const scope = ref<'near' | 'world'>('near')

/** North-south degrees in the close view: ~1,100 km, which is a region rather
 *  than a street and is the honest limit of a 1:110M outline. */
const NEAR_SPAN = 10
/** How wide the drawn box is against its height - and it must match the CSS
 *  aspect-ratio, because preserveAspectRatio="none" stretches the viewBox to
 *  fill it. A mismatch reintroduces exactly the longitude stretch the
 *  compensation below exists to remove. */
const boxAspect = computed(() => (props.compact ? 1.5 : 2))

const lat = computed(() => props.location.latitude)
const lon = computed(() => props.location.longitude)

/** The window into the world, in degrees. The paths are plate carrée with y
 *  flipped, so the viewBox is the geography and no projection code exists. */
const view = computed(() => {
  if (scope.value === 'world') return { x: -180, y: -90, w: 360, h: 180 }
  // Plate carrée stretches longitude by 1/cos(lat). Uncompensated, a Nordic
  // photo gets a Norway smeared to nearly three times its width; the clamp
  // stops the correction running away at the poles.
  //
  // This is why the SVG says preserveAspectRatio="none". The correction is an
  // anisotropic scale - more degrees of longitude across the same pixels -
  // and any uniform fit (meet/slice) simply undoes it, silently, by rescaling
  // both axes together. Measured on a rendered page before believing it:
  // under `slice` the Tromsø view came out at the raw 2.9x stretch. Strokes
  // survive `none` because every path carries non-scaling-stroke.
  const squash = Math.max(Math.cos((lat.value * Math.PI) / 180), 0.15)
  const w = (NEAR_SPAN * boxAspect.value) / squash
  return { x: lon.value - w / 2, y: -lat.value - NEAR_SPAN / 2, w, h: NEAR_SPAN }
})

/** Copies of the outline shifted a full turn, for a window that straddles the
 *  antimeridian - otherwise a photo from Fiji sits alone in a blank sea. */
const wraps = computed(() => {
  const v = view.value
  const out = [0]
  if (v.x < -180) out.push(-360)
  if (v.x + v.w > 180) out.push(360)
  return out
})

const viewBox = computed(() => {
  const v = view.value
  return `${v.x} ${v.y} ${v.w} ${v.h}`
})

/** The marker sits in an overlay in PIXELS, not in the SVG: inside a viewBox
 *  measured in degrees it would be re-scaled by every zoom. */
const pin = computed(() => {
  const v = view.value
  return {
    left: `${((lon.value - v.x) / v.w) * 100}%`,
    top: `${((-lat.value - v.y) / v.h) * 100}%`,
  }
})

/** A round distance about a quarter of the box wide. Only in the near view:
 *  at world scale a horizontal bar means a different distance at every
 *  latitude, so drawing one would be a lie of convenience. */
const STEPS_KM = [10, 20, 50, 100, 200, 500, 1000]
const scale = computed(() => {
  if (scope.value === 'world') return null
  const kmPerDeg = 111.32 * Math.max(Math.cos((lat.value * Math.PI) / 180), 0.15)
  const target = (view.value.w * kmPerDeg) / 4
  const km = STEPS_KM.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a))
  return { km, width: `${(km / kmPerDeg / view.value.w) * 100}%` }
})

const coords = computed(
  () =>
    `${Math.abs(lat.value).toFixed(6)}° ${lat.value < 0 ? 'S' : 'N'}, ` +
    `${Math.abs(lon.value).toFixed(6)}° ${lon.value < 0 ? 'W' : 'E'}`,
)

const altitude = computed(() => {
  const a = props.location.altitude
  if (a == null) return ''
  const m = Math.round(Math.abs(a))
  return `${m.toLocaleString()} m ${a < 0 ? 'below' : 'above'} sea level`
})

// Decimal degrees, comma-separated: what every map and every search box takes.
// Unlabelled here, unlike the display above, because that is the format they
// parse - a pasted "43.467448° N" is a coin flip.
const copied = ref(false)
let copiedTimer: ReturnType<typeof setTimeout> | undefined
async function copyCoords(): Promise<void> {
  try {
    await copyText(`${lat.value}, ${lon.value}`)
    copied.value = true
    clearTimeout(copiedTimer)
    copiedTimer = setTimeout(() => (copied.value = false), 1500)
  } catch {
    /* clipboard denied - the button just doesn't confirm */
  }
}
onBeforeUnmount(() => clearTimeout(copiedTimer))

// ── the live map (layer 3) ───────────────────────────────────────────────────

const settings = useSettingsStore()
// The default basemap follows the Studio's theme, so a map opened in a dark
// pane is a dark map. A custom URL overrides both.
const tileUrl = computed(() => tileTemplate(settings.mapTiles, settings.theme))
/** Said out loud beside the button, before anything is called. */
const host = computed(() => tileHost(tileUrl.value))
const attribution = computed(() => attributionFor(host.value))

const live = ref(false)
const liveError = ref('')
const mapEl = ref<HTMLElement | null>(null)
// Not a ref: MapLibre's Map is a big mutable object with its own event loop,
// and making it reactive buys nothing and costs a proxy on every frame. The
// TYPES come from a type-only import, which the compiler erases - the runtime
// import stays inside openMap() so the bundle stays lazy.
let map: import('maplibre-gl').Map | null = null

async function openMap(): Promise<void> {
  live.value = true
  liveError.value = ''
  try {
    // Both lazy: maplibre is ~250 KB gzipped that a user who never opens a map
    // must not pay for, and its CSS rides the same chunk.
    const [maplibre, workerUrl] = await Promise.all([
      import('maplibre-gl'),
      import('maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url').then((m) => m.default),
      import('maplibre-gl/dist/maplibre-gl.css'),
    ])
    // Without this the VECTOR MAP is BLANK, and blank in a way that looks like
    // a dead tile server rather than a bug: maplibre v6 resolves its worker as
    // `new URL('./maplibre-gl-worker.mjs', import.meta.url)`, i.e. a file it
    // expects to sit beside its own bundle. Vite emits no such file, so the
    // request 404s, the worker never starts, and every VECTOR source stays
    // unloaded forever - no error event, no console message, just a map that
    // never paints. Raster tiles are decoded on the main thread, which is why
    // the raster style tested fine and hid this. Measured over CDP: the built
    // bundle asked for /assets/maplibre-gl-worker.mjs and got a 404, and not
    // one .pbf or glyph was ever requested.
    //
    // `?worker&url` and not plain `?url`: the shipped worker is an ES module
    // that imports a sibling `maplibre-gl-shared.mjs`, and a plain `?url` copies
    // the one file without it - which fails the same way one step later. This
    // asks Vite to BUNDLE the worker (imports resolved, emitted as .js), which
    // also sidesteps whether a given static server types .mjs as JavaScript.
    maplibre.setWorkerUrl(workerUrl)
    await nextTick()
    if (!mapEl.value || !live.value) return
    const accent = getComputedStyle(document.documentElement)
      .getPropertyValue('--pk-accent')
      .trim()
    // Two kinds of address, one map. A style URL (the default) is handed over
    // whole; a raster {z}/{x}/{y} template gets a minimal style built around
    // it, which is what lets someone paste either into the setting.
    const style: import('maplibre-gl').StyleSpecification | string = isRasterTemplate(tileUrl.value)
      ? {
          version: 8 as const,
          sources: {
            tiles: {
              type: 'raster' as const,
              tiles: [tileUrl.value],
              tileSize: 256,
              maxzoom: 19,
            },
          },
          layers: [{ id: 'tiles', type: 'raster' as const, source: 'tiles' }],
        }
      : tileUrl.value
    const m = new maplibre.Map({
      container: mapEl.value,
      style,
      center: [lon.value, lat.value],
      // close enough to see the street the photo was taken on, far enough that
      // a coordinate a few hundred metres out still shows its surroundings
      zoom: 14,
      // In an ANSWER the map sits inside a scrolling thread, where a wheel
      // that zooms is a wheel that traps you: you scroll past the picture and
      // the page stays put. Cooperative gestures ask for ctrl+wheel (two
      // fingers on a trackpad) and say so on screen. The document pane has a
      // column of its own, so there it stays a plain wheel-zoom.
      cooperativeGestures: props.compact === true,
      // Ours only when we built the style. A style URL's sources carry their
      // own credit (OpenFreeMap's arrives with the TileJSON), and adding ours
      // on top printed it twice - measured on the first render, and a doubled
      // credit reads as a bug in the thing whose whole job is being careful
      // about credit.
      attributionControl: {
        compact: false,
        ...(typeof style === 'string' ? {} : { customAttribution: attribution.value }),
      },
    })
    map = m
    m.addControl(new maplibre.NavigationControl({ showCompass: false }), 'top-right')
    new maplibre.Marker({ color: accent || '#e0703a' }).setLngLat([lon.value, lat.value]).addTo(m)
    // No silent failures: an unreachable tile host paints an empty grey box,
    // which reads as "the map is broken" rather than "you are offline".
    m.on('error', (e: { error?: { message?: string } }) => {
      if (!liveError.value) {
        liveError.value = `Could not load tiles from ${host.value}. ${
          e.error?.message ?? 'The host did not answer.'
        }`
      }
    })
  } catch (e) {
    liveError.value = e instanceof Error ? e.message : String(e)
  }
}

function closeMap(): void {
  map?.remove()
  map = null
  live.value = false
  liveError.value = ''
}

// Flipping the theme with the map open swaps the basemap under it rather than
// tearing the map down: the user is looking at a place, not at a widget, and
// losing their pan and zoom to a colour change would be its own bug. setStyle
// keeps the camera; a raster template only needs its source's URLs changed.
watch(tileUrl, (url) => {
  if (!map) return
  if (isRasterTemplate(url)) {
    const src = map.getSource('tiles') as import('maplibre-gl').RasterTileSource | undefined
    src?.setTiles([url])
    return
  }
  map.setStyle(url)
})

// A different photo is a different place: never leave the previous one's map
// running under new coordinates.
watch(() => [props.location.latitude, props.location.longitude], closeMap)
onBeforeUnmount(closeMap)
</script>

<template>
  <section class="ph" :class="{ 'ph--compact': compact }">
    <div v-if="live" class="ph__map ph__map--live">
      <div ref="mapEl" class="ph__gl"></div>
      <p v-if="liveError" class="ph__err">{{ liveError }}</p>
    </div>
    <div v-else class="ph__map">
      <svg
        v-if="outline"
        class="ph__svg"
        :viewBox="viewBox"
        preserveAspectRatio="none"
        role="img"
        :aria-label="`Map: ${location.place?.description ?? coords}`"
      >
        <g v-for="dx in wraps" :key="dx" :transform="`translate(${dx} 0)`">
          <path v-for="(d, i) in outline.LAND" :key="`l${i}`" :d="d" class="ph__land" />
          <path v-for="(d, i) in outline.BORDERS" :key="`b${i}`" :d="d" class="ph__border" />
        </g>
      </svg>
      <span class="ph__pin" :style="pin" aria-hidden="true" />
      <div v-if="scale && !compact" class="ph__scale" aria-hidden="true">
        <span class="ph__scale-bar" :style="{ width: scale.width }" />
        <span class="ph__scale-km">{{ scale.km }} km</span>
      </div>
      <ToggleGroup v-if="!compact" v-model="scope" class="ph__scope" label="Map extent">
        <ToggleGroupItem value="near" class="ph__scope-btn">Nearby</ToggleGroupItem>
        <ToggleGroupItem value="world" class="ph__scope-btn">World</ToggleGroupItem>
      </ToggleGroup>
    </div>

    <p v-if="location.place" class="ph__place">
      <Icon name="pin" :size="14" />
      <span>{{ location.place.description }}</span>
    </p>
    <p v-if="!compact" class="ph__coords">
      <span class="ph__nums">{{ coords }}</span>
      <Tooltip :label="copied ? 'Copied' : 'Copy coordinates'">
        <button type="button" class="pk-icon-btn ph__copy" @click="copyCoords">
          <Icon :name="copied ? 'check' : 'copy'" :size="13" />
        </button>
      </Tooltip>
    </p>
    <p v-if="altitude && !compact" class="ph__alt">{{ altitude }}</p>

    <div class="ph__foot">
      <button v-if="!live" type="button" class="pk-btn pk-btn--sm" @click="openMap">
        <Icon name="globe" :size="14" />
        Open map
      </button>
      <button v-else type="button" class="pk-btn pk-btn--sm" @click="closeMap">Close map</button>
      <span class="ph__wire">{{ live ? `Tiles from ${host}.` : `Loads map tiles from ${host}.` }}</span>
    </div>
  </section>
</template>

<style scoped>
.ph {
  margin: 0 0 16px;
}
/* The map inside an answer. Not a strip: 3:2 gives the pin somewhere to sit
   and the surrounding country room to be recognised, which a 3:1 letterbox
   did not. */
.ph--compact {
  margin: 0;
}
.ph--compact .ph__map {
  aspect-ratio: 3 / 2;
}
.ph--compact .ph__place {
  margin-top: 6px;
  font-size: var(--pk-font-size-xs);
}
.ph--compact .ph__foot {
  margin-top: 6px;
}
.ph__map {
  position: relative;
  aspect-ratio: 2 / 1;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  overflow: hidden;
}
.ph__svg {
  display: block;
  width: 100%;
  height: 100%;
}
/* Land must read LIGHTER than water in both themes, and no pair of surface
   tokens guarantees that - bg-elevated is darker than bg-inset in the dark
   theme, which drew a black Italy in a pale sea. Mixing toward the text colour
   is ordered by construction: text always contrasts with the ground. */
.ph__land {
  fill: color-mix(in srgb, var(--pk-text-muted) 30%, var(--pk-bg-inset));
  stroke: var(--pk-border-strong);
  /* the viewBox is degrees, so every stroke would scale with the zoom */
  stroke-width: 1px;
  vector-effect: non-scaling-stroke;
}
.ph__border {
  fill: none;
  stroke: var(--pk-border-default);
  stroke-width: 1px;
  stroke-dasharray: 2 2;
  vector-effect: non-scaling-stroke;
}
/* The marker: a dot with a halo, so it reads on land and on water alike, and
   a slow pulse ringing out of it - at this scale a static 11px dot is easy to
   lose against a coastline, and the eye finds movement before it finds
   contrast. Slow and once every two seconds: a chat is not a dashboard. */
.ph__pin {
  position: absolute;
  width: 11px;
  height: 11px;
  margin: -5.5px 0 0 -5.5px;
  border: 2px solid var(--pk-bg-surface);
  border-radius: var(--pk-radius-full);
  background: var(--pk-accent);
  box-shadow: 0 0 0 1px var(--pk-accent);
  pointer-events: none;
}
.ph__pin::after {
  content: '';
  position: absolute;
  inset: -3px;
  border: 2px solid var(--pk-accent);
  border-radius: var(--pk-radius-full);
  animation: ph-pulse 2.4s ease-out infinite;
}
@keyframes ph-pulse {
  0% {
    transform: scale(0.7);
    opacity: 0.85;
  }
  70%,
  100% {
    transform: scale(2.8);
    opacity: 0;
  }
}
/* Someone who asked the OS for less motion gets a plain dot. */
@media (prefers-reduced-motion: reduce) {
  .ph__pin::after {
    animation: none;
    opacity: 0.35;
  }
}
/* Full-bleed so the bar's percentage resolves against the MAP's width - the
   number under it is a distance, so the bar has to be that fraction of the
   view and not of whatever a shrink-wrapped box happened to be. */
.ph__scale {
  position: absolute;
  inset: auto 0 6px 0;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  color: var(--pk-text-secondary);
}
.ph__scale-bar {
  flex: none;
  margin-left: 8px;
  height: 5px;
  border: 1px solid var(--pk-text-secondary);
  border-top: 0;
}
.ph__scale-km {
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  text-shadow: 0 0 3px var(--pk-bg-inset);
}
.ph__scope {
  position: absolute;
  top: 6px;
  right: 6px;
  display: flex;
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  overflow: hidden;
}
.ph__scope :deep(.ph__scope-btn) {
  padding: 3px 8px;
  border: 0;
  background: transparent;
  color: var(--pk-text-muted);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.ph__scope :deep(.ph__scope-btn:hover) {
  background: var(--pk-bg-hover);
  color: var(--pk-text-primary);
}
.ph__scope :deep(.ph__scope-btn[data-state='on']) {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.ph__place {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 10px 0 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.ph__place svg {
  flex-shrink: 0;
  color: var(--pk-text-muted);
}
.ph__coords {
  display: flex;
  align-items: center;
  gap: 4px;
  margin: 3px 0 0;
}
.ph__nums {
  font-family: var(--pk-font-mono, ui-monospace, monospace);
  font-size: 12px;
  color: var(--pk-text-secondary);
}
/* the shared icon button is sized for a toolbar; this one sits in a line of
   12px type */
.ph__copy {
  width: 22px;
  height: 22px;
}
.ph__alt {
  margin: 3px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.ph__foot {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
}
.ph__wire {
  flex: 1;
  min-width: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
/* A real map wants room the 2:1 locator does not; the pane itself resizes. */
.ph__map--live {
  aspect-ratio: 4 / 3;
}
.ph__gl {
  width: 100%;
  height: 100%;
}
.ph__err {
  position: absolute;
  inset: auto 8px 8px;
  margin: 0;
  padding: 8px 10px;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
/* MapLibre draws its own chrome with hard-coded light colours; these are the
   few that would otherwise sit on the pane looking like someone else's app.
   The zoom buttons are the loudest - a white rectangle on a dark map. */
.ph__map--live :deep(.maplibregl-ctrl-group) {
  border: 1px solid var(--pk-border-subtle);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-surface);
  box-shadow: none;
}
.ph__map--live :deep(.maplibregl-ctrl-group button) {
  background: transparent;
}
.ph__map--live :deep(.maplibregl-ctrl-group button:hover) {
  background: var(--pk-bg-hover);
}
.ph__map--live :deep(.maplibregl-ctrl-group button + button) {
  border-top-color: var(--pk-border-subtle);
}
/* The +/- glyphs are background SVGs drawn in near-black, so on a dark ground
   they vanish. Inverting the icon is the whole fix and touches nothing else. */
[data-theme='dark'] .ph__map--live :deep(.maplibregl-ctrl-icon) {
  filter: invert(1) brightness(0.85);
}
.ph__map--live :deep(.maplibregl-ctrl-attrib) {
  background: color-mix(in srgb, var(--pk-bg-surface) 82%, transparent);
  color: var(--pk-text-secondary);
  font-size: 10px;
}
.ph__map--live :deep(.maplibregl-ctrl-attrib a) {
  color: var(--pk-text-secondary);
}
</style>
