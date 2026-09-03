<script setup lang="ts">
// One TooltipProvider at the root feeds every <Tooltip> in the app: a shared
// open delay and the "skip the delay when hopping between adjacent tooltips" UX.
// Toaster is the app's single toast outlet (outcome notices, e.g. deploys).
// EvictConfirm is the one "Stop X and start Y?" dialog - any start path can
// raise it via the fleet store, so it mounts once here like the toasts.
import { TooltipProvider } from 'reka-ui'
import Toaster from '@/components/ui/Toaster.vue'
import EvictConfirm from '@/components/manage/EvictConfirm.vue'
import KeyGate from '@/components/KeyGate.vue'
import { useModelsStore } from '@/stores/models'

const models = useModelsStore()
</script>

<template>
  <TooltipProvider :delay-duration="400" :skip-delay-duration="300">
    <router-view />
    <KeyGate v-if="models.needsKey" />
    <Toaster />
    <EvictConfirm />
  </TooltipProvider>
</template>
