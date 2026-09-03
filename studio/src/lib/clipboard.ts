// navigator.clipboard is SECURE-CONTEXT-ONLY: present on https and on
// http://localhost, absent on a LAN origin like http://10.10.0.189 - where
// every copy button in the Studio silently threw. The execCommand path is
// deprecated but works in every context, which is exactly the job here.
export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text)
    return
  }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  ta.focus()
  ta.select()
  try {
    document.execCommand('copy')
  } finally {
    ta.remove()
  }
}
