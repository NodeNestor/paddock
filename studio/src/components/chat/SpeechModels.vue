<script setup lang="ts">
// The speech models this box has configured, started and stopped in place
// Lives in both mic menus: the one shown when nothing is
// listening (where every row is a Start) and the one behind the running mic's
// chevron (which is the only place a Stop could ever be reached).
//
// Two buttons per row, not a clickable row. A row that acts on click cannot
// say which way it will act, and the state it depends on is the thing you
// opened the menu to find out.
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useFleetStore } from '@/stores/fleet'
import { useModelsStore } from '@/stores/models'
import { friendlyModelName } from '@/lib/model-caps'
import Icon from '@/components/Icon.vue'
import VendorLogo from '@/components/manage/VendorLogo.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'

const fleet = useFleetStore()
const models = useModelsStore()
const router = useRouter()

const rows = computed(() => fleet.speechEndpoints)
/** The port this menu is waiting on. The store's own `busy` covers a start
 *  begun anywhere; this is what keeps the button labelled while our own call
 *  is still in the air, before the store has caught up. */
const pending = ref<number | null>(null)

async function act(port: number, model: string | null, start: boolean): Promise<void> {
  if (pending.value !== null) return
  pending.value = port
  try {
    if (start) await fleet.startConfigured(port, model ?? String(port))
    else await fleet.stop(port)
    // Caps come from the RUNNER, so the model list has to re-read before the
    // composer's `transcribers` can see a newly started one.
    await models.refresh()
  } finally {
    pending.value = null
  }
}
</script>

<template>
  <div v-for="r in rows" :key="r.port" class="spm">
    <VendorLogo v-if="r.vendor" :vendor="r.vendor" :size="14" class="spm__mark" />
    <Icon v-else name="microphone" :size="14" class="spm__mark" />
    <span class="spm__id">
      <span class="spm__name">{{ r.display ?? friendlyModelName(r.model ?? '') }}</span>
      <span class="spm__sub">
        {{ pending === r.port || r.busy ? 'working...' : r.running ? `running · port ${r.port}` : `port ${r.port}` }}
      </span>
    </span>
    <span class="spm__acts">
      <button
        class="pk-btn pk-btn--sm"
        type="button"
        :disabled="r.running || r.busy || pending !== null"
        @click="act(r.port, r.model, true)"
      >
        Start
      </button>
      <button
        class="pk-btn pk-btn--sm"
        type="button"
        :disabled="!r.running || r.busy || pending !== null"
        @click="act(r.port, r.model, false)"
      >
        Stop
      </button>
    </span>
  </div>
  <!-- Getting a new one rides with the list rather than being placed by each
       host: it went missing from the running-mic menu when only the other one
       had it. -->
  <MenuSeparator />
  <MenuItem @select="router.push({ name: 'server-new' })">
    <Icon name="plus" :size="14" />
    <span>Start a new speech model</span>
  </MenuItem>
</template>

<style scoped>
.spm {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: var(--pk-radius-md);
}
.spm__mark {
  flex: none;
}
/* takes the slack so the buttons sit hard right at every menu width */
.spm__id {
  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
}
.spm__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.spm__sub {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.spm__acts {
  display: flex;
  flex: none;
  gap: 4px;
}
</style>
