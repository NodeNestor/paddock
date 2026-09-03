import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'

import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import './styles/base.css'
import './styles/components.css'

createApp(App).use(createPinia()).use(router).mount('#app')
