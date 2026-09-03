import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'

type Theme = 'light' | 'dark'

/**
 * Light/dark theme: the setting is stored in the `settings` Pinia store
 * (persisted to localStorage as `pk_theme`); applying it writes `data-theme`
 * on <html>. An anti-FOUC script in index.html applies the stored value
 * before this bundle loads, so `initTheme` just re-syncs the DOM to the store.
 */
export function useTheme() {
  const settings = useSettingsStore()

  const theme = computed<Theme>({
    get: () => settings.theme,
    set: (v) => {
      settings.theme = v
    },
  })

  const setTheme = (newTheme: Theme) => {
    if (newTheme !== 'light' && newTheme !== 'dark') newTheme = 'dark'
    settings.theme = newTheme
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const toggleTheme = (): Theme => {
    const next: Theme = settings.theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    return next
  }

  const initTheme = () => {
    document.documentElement.setAttribute('data-theme', settings.theme)
  }

  return { theme, setTheme, toggleTheme, initTheme }
}
