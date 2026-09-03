// Where the interactive map's tiles come from (layer 3).
//
// Two places need to agree about this - the Settings field that sets it and
// the photo pane that calls it - and the thing they must agree on is a host
// NAME shown to the user. A second copy of this logic would eventually name
// one host in the settings and call another.

/** The default pair, one per theme: OpenFreeMap's Positron and Dark styles.
 *
 *  Why this SOURCE AND not the OBVIOUS ONES. Paddock is a product people pay
 *  for, so its default map service has to be one we are allowed to point paying
 *  users at:
 *
 *  - CARTO's Positron/Dark Matter look right and were the first choice, until
 *    their own basemaps page settled it: "For commercial purposes, you will
 *    need an Enterprise license." Shipping them as the default would make every
 *    user an unlicensed one (the maintainer caught this).
 *  - OpenStreetMap's own tile server asks that applications not pull from it
 *    systematically for their users, and has no dark rendering at all.
 *  - OpenFreeMap is MIT, explicitly free for commercial use, needs no API key
 *    and sets no request limit; tiles, glyphs and sprites all come from the
 *    one host, so "we contact tiles.openfreemap.org" stays a true sentence.
 *
 *  It is a donation-funded public instance, which is exactly why the address
 *  stays a setting: a heavy user should point at their own server, and anyone
 *  with a provider contract should use it. */
export const DEFAULT_TILES = {
  light: 'https://tiles.openfreemap.org/styles/positron',
  dark: 'https://tiles.openfreemap.org/styles/dark',
} as const

/** OpenStreetMap's own tile server, named in the settings copy as the obvious
 *  alternative for personal use. Not the default: see above. */
export const OSM_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

/** The address to fetch. A custom setting wins for both themes - someone who
 *  named a server means that server, and silently swapping it at dusk would be
 *  a request they did not make. */
export function tileTemplate(custom: string, theme: 'light' | 'dark'): string {
  return custom.trim() || DEFAULT_TILES[theme]
}

/** Two kinds of address answer "where do tiles come from", and MapLibre takes
 *  them differently: a raster TEMPLATE has `{z}/{x}/{y}` in it and needs a
 *  style built around it, a STYLE URL is handed over whole. Deciding by the
 *  placeholders means a user can paste either and neither needs explaining. */
export function isRasterTemplate(url: string): boolean {
  return /\{z\}/i.test(url)
}

/** The host an address will contact, for saying so before calling it. `{z}`
 *  and friends are not legal URL characters, so they come out first; anything
 *  unparseable is echoed back rather than swallowed, since a user who typed a
 *  broken URL needs to see what they typed. */
export function tileHost(url: string): string {
  try {
    return new URL(url.replace(/\{[^}]*\}/g, '0')).host
  } catch {
    return url
  }
}

const OSM_CREDIT =
  '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors'

/** Who to credit, decided by the host that actually served the pixels.
 *
 *  Every source here is OpenStreetMap data, and ODbL requires that credit;
 *  the renderers ask for their own on top. None of it is safe to assume about
 *  a host we have never heard of, so an unknown one is NAMED instead of being
 *  credited to someone who had nothing to do with it. */
export function attributionFor(host: string): string {
  if (/(^|\.)openfreemap\.org$/i.test(host)) {
    return `<a href="https://openfreemap.org" target="_blank" rel="noreferrer">OpenFreeMap</a> © <a href="https://openmaptiles.org/" target="_blank" rel="noreferrer">OpenMapTiles</a> - data from ${OSM_CREDIT}`
  }
  if (/(^|\.)openstreetmap\.org$/i.test(host)) return OSM_CREDIT
  if (/(^|\.)cartocdn\.com$/i.test(host)) {
    return `${OSM_CREDIT}, © <a href="https://carto.com/attributions" target="_blank" rel="noreferrer">CARTO</a>`
  }
  return `Tiles from ${host}`
}
