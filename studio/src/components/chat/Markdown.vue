<script setup lang="ts">
// Assistant markdown, plus the one block that is not text: a ```map fence the
// model emits becomes a real map, live, as the answer streams.
// Everything else goes to markstream untouched - the split happens at block
// boundaries, so a map between two paragraphs costs the renderer nothing but
// a second instance, and a message without one (nearly all of them) takes a
// fast path that allocates nothing.
import { computed } from 'vue'
import { MarkdownRender } from 'markstream-vue'
import { useSettingsStore } from '@/stores/settings'
import { MD_LANGS } from '@/lib/markstream'
import { splitMapBlocks, type MapBlock } from '@/lib/mapblock'
import PhotoLocation from '@/components/chat/PhotoLocation.vue'
import type { FileLocation } from '@/lib/api'

const props = defineProps<{ content: string; streaming?: boolean }>()
const settings = useSettingsStore()
const isDark = computed(() => settings.theme === 'dark')

const segments = computed(() => splitMapBlocks(props.content))

/** The map component speaks the metadata shape, because that is where this
 *  started; a model-emitted block fills the same fields. `description` is what
 *  renders under the map, so an unlabelled block simply shows none. */
function asLocation(m: MapBlock): FileLocation {
  return {
    latitude: m.lat,
    longitude: m.lon,
    place: m.label
      ? { city: m.label, region: '', country: '', distance_km: 0, bearing: 'N', description: m.label }
      : undefined,
  }
}

/** Only the last markdown segment is still arriving; the ones before a closed
 *  map block are final, and telling markstream so lets it stop re-measuring
 *  them on every chunk. */
function isLive(i: number): boolean {
  return !!props.streaming && i === segments.value.length - 1
}
</script>

<template>
  <template v-for="(seg, i) in segments" :key="i">
    <MarkdownRender
      v-if="seg.kind === 'md'"
      class="pk-md"
      :content="seg.text"
      :is-dark="isDark"
      code-renderer="shiki"
      :langs="MD_LANGS"
      :final="!isLive(i)"
      :typewriter="false"
      :smooth-streaming="false"
      :batch-rendering="true"
      mode="chat"
    />
    <div v-else class="pk-md__map">
      <PhotoLocation :location="asLocation(seg.map)" compact />
    </div>
  </template>
</template>

<style scoped>
.pk-md__map {
  max-width: 420px;
  margin: 10px 0;
}
</style>
