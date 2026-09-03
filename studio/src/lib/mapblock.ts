// A map the MODEL asked for.
//
// The runner offers the capability only when a photo actually carries GPS -
// the injected line ends with a ready-to-copy ```map block - and the answer
// includes it or does not. That is the whole point of the redesign: a map
// under every geotagged photo is us deciding, and we were deciding wrong
// ("now you will dump a map without the model wanting to
// show it").
//
// A fenced block rather than an inline marker, for two reasons: it is what
// mermaid and friends already do so models emit it reliably, and it can only
// appear at a block boundary, which means splitting the markdown around it
// cannot cut a sentence - or a table - in half.

export interface MapBlock {
  lat: number
  lon: number
  /** What to call the place. Empty is fine; the map still draws. */
  label: string
}

export type MdSegment = { kind: 'md'; text: string } | { kind: 'map'; map: MapBlock }

/** Body -> coordinates, forgiving deliberately.
 *
 *  A 9B model writing JSON gets it right most of the time, and "most" is not a
 *  rendering strategy. Both of these work:
 *
 *      43.467448, 11.885127, Arezzo
 *      {"lat": 43.467448, "lon": 11.885127, "label": "Arezzo"}
 *
 *  Anything else returns null and the block stays a visible code block, so a
 *  malformed map reads as what the model actually said rather than vanishing.
 */
export function parseMapBody(body: string): MapBlock | null {
  const text = body.trim()
  if (!text) return null

  if (text.startsWith('{')) {
    try {
      const o = JSON.parse(text) as Record<string, unknown>
      const lat = Number(o.lat ?? o.latitude)
      const lon = Number(o.lon ?? o.lng ?? o.longitude)
      const label = String(o.label ?? o.name ?? o.place ?? '')
      return valid(lat, lon) ? { lat, lon, label: label.slice(0, 120) } : null
    } catch {
      return null
    }
  }

  // "lat, lon" with anything after it treated as the name - a place with a
  // comma in it ("Arezzo, Tuscany") survives, which splitting on every comma
  // would not.
  const m = /^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*(.*))?$/s.exec(
    text.split('\n')[0] ?? '',
  )
  if (!m) return null
  const lat = Number(m[1])
  const lon = Number(m[2])
  return valid(lat, lon) ? { lat, lon, label: (m[3] ?? '').trim().slice(0, 120) } : null
}

function valid(lat: number, lon: number): boolean {
  return (
    Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    Math.abs(lat) <= 90 &&
    Math.abs(lon) <= 180 &&
    // 0,0 is in the Gulf of Guinea and is what a defaulted or half-parsed pair
    // looks like. A real photo taken there is a loss we accept.
    !(lat === 0 && lon === 0)
  )
}

/** Markdown split around its map blocks, in order.
 *
 *  An UNCLOSED trailing block is dropped rather than shown: mid-stream it is
 *  three tokens of coordinates the reader does not want to watch arrive, and
 *  it becomes the map a moment later when the fence closes.
 */
export function splitMapBlocks(content: string): MdSegment[] {
  // The overwhelmingly common case, and it must cost nothing: this runs on
  // every streamed chunk.
  if (!content.includes('```map')) return [{ kind: 'md', text: content }]

  const out: MdSegment[] = []
  const lines = content.split('\n')
  let md: string[] = []
  let i = 0
  const flush = () => {
    if (md.length) out.push({ kind: 'md', text: md.join('\n') })
    md = []
  }
  while (i < lines.length) {
    const line = lines[i] ?? ''
    if (line.trim() !== '```map') {
      md.push(line)
      i++
      continue
    }
    // collect to the closing fence
    let j = i + 1
    const body: string[] = []
    while (j < lines.length && (lines[j] ?? '').trim() !== '```') {
      body.push(lines[j] ?? '')
      j++
    }
    if (j >= lines.length) {
      // still streaming - hide the half-written block, keep the text above it
      flush()
      return out
    }
    const map = parseMapBody(body.join('\n'))
    if (map) {
      flush()
      out.push({ kind: 'map', map })
    } else {
      // unparseable: leave it exactly as the model wrote it
      md.push(line, ...body, '```')
    }
    i = j + 1
  }
  flush()
  return out
}
