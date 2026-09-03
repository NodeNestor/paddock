import { defineStore } from 'pinia'
import { uuid } from '@/lib/uuid'
import { ref } from 'vue'
import { promptsApi, type SavedPrompt } from '@/lib/api'

function uid(): string {
  return uuid()
}

/** The reusable system-prompt library, backed by the server store (/api/prompts).
 *  Separate from a conversation's own `systemPrompt`: these are named, saved
 *  prompts the user can apply to any chat. */
export const usePromptsStore = defineStore('prompts', () => {
  const prompts = ref<SavedPrompt[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function refresh(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      prompts.value = await promptsApi.list()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  /** Create or update a named prompt; returns the saved record. */
  async function save(name: string, body: string, id?: string): Promise<SavedPrompt> {
    const rec: SavedPrompt = { id: id ?? uid(), name: name.trim() || 'Untitled', body }
    await promptsApi.save(rec)
    await refresh()
    return prompts.value.find((p) => p.id === rec.id) ?? rec
  }

  async function remove(id: string): Promise<void> {
    await promptsApi.remove(id)
    prompts.value = prompts.value.filter((p) => p.id !== id)
  }

  return { prompts, loading, error, refresh, save, remove }
})
