<script setup lang="ts">
// Install this computer's certificate on the device you browse from.
//
// Its own route, not a fold-out: this is a page you open on the other device -
// the phone or the laptop that needs the certificate - so it has to be a URL
// someone can type or send. Same reasoning as the graphics-card sheet.
import { computed, onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import Tabs from '@/components/ui/Tabs.vue'
import { copyText } from '@/lib/clipboard'

interface TlsInfo {
  enabled: boolean
  fingerprint?: string
  names?: string[]
  root_url?: string
}

const info = ref<TlsInfo | null>(null)
const copied = ref(false)
const platform = ref('windows')

const PLATFORMS = [
  { value: 'windows', label: 'Windows' },
  { value: 'macos', label: 'macOS' },
  { value: 'ios', label: 'iPhone & iPad' },
  { value: 'android', label: 'Android' },
  { value: 'firefox', label: 'Firefox' },
]

// One command where the platform has one. It does the import AND the trust in
// a single step, which the click-through paths do not - on macOS especially,
// where double-clicking a certificate can fail with "Unable to import ... Error:
// -25294" (the keychain it wanted could not be resolved) before you ever reach
// the trust setting. That has been hit for real.
const COMMANDS: Record<string, string> = {
  macos:
    'sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ~/Downloads/paddock-root.crt',
  windows: 'certutil -user -f -addstore Root %USERPROFILE%\\Downloads\\paddock-root.crt',
}

const STEPS: Record<string, string[]> = {
  windows: [
    'Download the certificate, then double-click the file.',
    'Choose Install Certificate, then Current User.',
    'Choose Place all certificates in the following store, then Browse, then Trusted Root Certification Authorities.',
    'Finish, confirm the security warning, and restart your browser.',
  ],
  macos: [
    'Download the certificate, then open Keychain Access.',
    'Choose File, then Import Items, and pick the login keychain by name - double-clicking the file instead is what fails with Error: -25294.',
    'Find Paddock in the list, double-click it, and open Trust.',
    'Set When using this certificate to Always Trust, close the window and enter your password.',
  ],
  ios: [
    'Open this page in Safari - other browsers cannot install a certificate.',
    'Download the certificate, then open Settings. Profile Downloaded appears near the top.',
    'Install it, entering your passcode.',
    'Then open Settings, General, About, Certificate Trust Settings, and turn Paddock on.',
  ],
  android: [
    'Download the certificate.',
    'Open Settings and search for Install a certificate.',
    'Choose CA certificate, then Install anyway.',
    'Pick the downloaded file.',
  ],
  firefox: [
    'Firefox keeps its own list, so it needs this even if your system already trusts the certificate.',
    'Download the certificate.',
    'Open Settings, Privacy & Security, and scroll to Certificates. Choose View Certificates.',
    'On the Authorities tab choose Import, pick the file, and tick Trust this CA to identify websites.',
  ],
}

// The addresses worth showing are the ones another device can reach. Loopback
// is in the certificate too, but a phone typing 127.0.0.1 reaches itself.
const reachable = computed(() =>
  (info.value?.names ?? []).filter((n) => n !== 'localhost' && n !== '127.0.0.1' && n !== '::1'),
)

function url(name: string): string {
  const host = name.includes(':') ? `[${name}]` : name
  const port = location.port ? `:${location.port}` : ''
  return `https://${host}${port}`
}

async function copyFingerprint(): Promise<void> {
  if (!info.value?.fingerprint) return
  await copyText(info.value.fingerprint)
  copied.value = true
  setTimeout(() => (copied.value = false), 1600)
}

const copiedCmd = ref(false)
async function copyCommand(): Promise<void> {
  const cmd = COMMANDS[platform.value]
  if (!cmd) return
  await copyText(cmd)
  copiedCmd.value = true
  setTimeout(() => (copiedCmd.value = false), 1600)
}

onMounted(async () => {
  try {
    const r = await fetch('/tls/info')
    if (r.ok) info.value = (await r.json()) as TlsInfo
  } catch {
    // A manager we cannot reach is the shell's problem to report.
  }
})
</script>

<template>
  <div class="tr">
    <div class="tr__head">
      <h1 class="tr__title">Trust this computer</h1>
      <p class="tr__lead">
        Browsers hand the microphone, the clipboard and other private things only to pages they
        trust. Install this computer's certificate once on each device you open the Studio from,
        and it stops warning you.
      </p>
    </div>

    <section v-if="info && !info.enabled" class="tr__card tr__card--warn">
      <h2 class="tr__h2"><Icon name="alert-triangle" :size="16" /> No certificate</h2>
      <p class="tr__body">
        This computer could not set up a certificate, so the Studio is being served without
        encryption. Browsers on other devices will refuse the microphone, and there is nothing to
        install here. The manager's log says why.
      </p>
    </section>

    <template v-else-if="info">
      <section class="tr__card">
        <h2 class="tr__h2"><Icon name="download" :size="16" /> The certificate</h2>
        <div class="tr__row">
          <a class="pk-btn pk-btn--primary" href="/tls/root.crt" download="paddock-root.crt">
            <Icon name="download" :size="15" /> Download certificate
          </a>
          <button class="pk-btn" type="button" @click="copyFingerprint">
            <Icon :name="copied ? 'check' : 'copy'" :size="15" />
            {{ copied ? 'Copied' : 'Copy fingerprint' }}
          </button>
        </div>
        <p class="tr__body">
          Your computer will show you a fingerprint before it trusts anything. It must match this
          one exactly:
        </p>
        <code class="tr__fp">{{ info.fingerprint }}</code>
      </section>

      <section class="tr__card">
        <h2 class="tr__h2"><Icon name="terminal" :size="16" /> How to install it</h2>
        <Tabs v-model="platform" :tabs="PLATFORMS" />
        <template v-if="COMMANDS[platform]">
          <p class="tr__body tr__label">One command, which also sets the trust:</p>
          <div class="tr__cmdrow">
            <code class="tr__cmd">{{ COMMANDS[platform] }}</code>
            <button class="pk-btn pk-btn--sm" type="button" @click="copyCommand">
              <Icon :name="copiedCmd ? 'check' : 'copy'" :size="14" />
              {{ copiedCmd ? 'Copied' : 'Copy' }}
            </button>
          </div>
          <p class="tr__body tr__label">Or by hand:</p>
        </template>
        <ol class="tr__steps">
          <li v-for="(s, i) in STEPS[platform]" :key="i">{{ s }}</li>
        </ol>
        <p class="tr__body tr__body--muted">
          Skipping this still works. Your browser will warn you every time and let you continue,
          and the microphone works once you do.
        </p>
      </section>

      <section v-if="reachable.length" class="tr__card">
        <h2 class="tr__h2"><Icon name="globe" :size="16" /> Where to open the Studio</h2>
        <ul class="tr__addrs">
          <li v-for="n in reachable" :key="n">
            <a :href="url(n)">{{ url(n) }}</a>
          </li>
        </ul>
        <p class="tr__body tr__body--muted">
          The certificate covers these addresses. Any other address will warn even after you
          install it.
        </p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.tr {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.tr__title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.tr__lead {
  margin: 6px 0 0;
  max-width: 70ch;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.tr__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
}
.tr__card--warn {
  border-color: var(--pk-status-warning);
}
.tr__h2 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 10px;
  font-size: var(--pk-font-size-md);
  font-weight: 650;
  color: var(--pk-text-primary);
}
.tr__row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}
.tr__body {
  margin: 0;
  max-width: 70ch;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary, var(--pk-text-primary));
}
.tr__body--muted {
  margin-top: 14px;
  color: var(--pk-text-muted);
}
/* The fingerprint is 95 characters of hex that someone reads off a system
   dialog character by character. It gets its own line, monospace, and must
   wrap rather than scroll - a comparison you have to scroll is one nobody
   finishes. */
.tr__fp {
  display: block;
  margin-top: 8px;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  line-height: 1.7;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.tr__label {
  margin-top: 16px;
  color: var(--pk-text-muted);
}
.tr__cmdrow {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-top: 6px;
}
/* The command is long and must WRAP rather than scroll: a line you have to
   scroll is a line people copy half of. */
.tr__cmd {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  line-height: 1.6;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.tr__steps {
  margin: 14px 0 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 70ch;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary, var(--pk-text-primary));
}
.tr__addrs {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
}
.tr__addrs a {
  color: var(--pk-accent);
}
</style>
