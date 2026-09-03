// Turn a picked/dropped/pasted File into an ImagePart with three deliberate
// sizes: a faithful VIEW copy (uploaded as a blob to the attachments table, for
// the lightbox), a MODEL copy sized to the vision budget (kept inline, sent to
// the model), and a tiny THUMB (inline, renders the chat bubble). The heavy
// original never lives in the conversation doc.

import type { AudioPart, FilePart, GraphPart, ImagePart } from '@/types/chat'
import { uuid } from '@/lib/uuid'
import { attachmentsApi } from './api'
import { pdfPageCount } from './pdf'
import { audioDuration } from './transcribe'
import { type ImageDetail } from './vision'

// Thumb: tiny, for the bubble. Originals go to the attachments table whole -
// the server owns resizing (vision budget), orientation and metadata.
const THUMB_EDGE = 160

// Per-file upload ceiling. Ollama defaults to 10 MB; we allow more headroom.
export const MAX_FILE_MB = 100
export const MAX_FILE_SIZE = MAX_FILE_MB * 1024 * 1024

/** Formats a browser may hand us with an empty `File.type`, because Windows has
 *  no registered MIME type for them. Extension is then the only clue there is -
 *  and getting it wrong is what sent the maintainers' IMG_5195.HEIC down the document
 *  road, where it came back as "looks like binary data, not text". */
const IMAGE_EXT = /\.(heic|heif|avif)$/i

export function isImageFile(f: File): boolean {
  // Type first when the browser has one: it is the better answer, and an
  // extension can lie. The extension test only fills a hole.
  return f.type.startsWith('image/') || (!f.type && IMAGE_EXT.test(f.name))
}

/** An image no model here can be shown, because nothing in Paddock decodes it.
 *
 *  HEIC only, and not because it is exotic - it is what an iPhone writes by
 *  default. It is HEVC, and every HEVC decoder is (L)GPL, so none can be
 * embedded in a closed binary. AVIF is deliberately not in this
 *  list: it decodes fine.
 *
 *  Knowing this in the COMPOSER is the whole point. The server refuses a HEIC
 *  with a clear message, but finding out there costs an upload and a turn -
 *  and the advice ("convert it") has to be acted on outside the app anyway,
 *  so the round trip buys nothing. */
const UNREADABLE_EXT = /\.(heic|heif)$/i
export function isUnreadableImage(f: File): boolean {
  return /^image\/hei[cf]$/i.test(f.type) || UNREADABLE_EXT.test(f.name)
}
export function isPdfFile(f: File): boolean {
  return f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
}

/** A Traverse graph database image. Routed to the graph session, never to a
 *  model - the extension is the whole format signal a File object gives us. */
export function isGraphFile(f: File): boolean {
  return /\.tvdb$/i.test(f.name)
}
/** TIFFs ride the image lane but can be multi-page documents - the server
 *  expands extra pages into per-page images, so the page-range menu applies. */
export function isTiffFile(f: File): boolean {
  return f.type === 'image/tiff' || /\.tiff?$/.test(f.name.toLowerCase())
}

/** Per-file document settings the composer's chip menu edits - the settings
 *  belong to the file, not the prompt, and ride the wire as part-level
 *  `pdf_mode` / `pages` fields on that attachment only. */
export interface DocOpts {
  /** PDF only: force the extracted-text route for this file. */
  text?: boolean
  /** page range (1-based inclusive) - absent bound = open end. */
  from?: number
  to?: number
}

/** The `pages` wire string for a file's range ("2-4" / "3" / "2-"), or
 *  undefined for all pages. */
export function pagesParam(o: DocOpts): string | undefined {
  if (!o.from && !o.to) return undefined
  const a = Math.max(1, o.from ?? 1)
  if (o.to && o.to >= a) return a === o.to ? String(a) : `${a}-${o.to}`
  if (o.to && o.to < a) return String(a)
  return `${a}-`
}
/** Anything the composer will stage - which is everything under the size cap.
 *  The server owns format truth: it extracts what it can (PDF today; text,
 * spreadsheets and Word docs) and refuses the rest with an
 *  honest message naming the file, which the chat surfaces. A client-side
 *  allowlist just gets stale as the server learns formats. */
export function isAttachableFile(f: File): boolean {
  void f
  return true
}
export function isTooLarge(f: File): boolean {
  return f.size > MAX_FILE_SIZE
}

function readAsDataUrl(f: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(r.result as string)
    r.onerror = () => reject(r.error ?? new Error('read failed'))
    r.readAsDataURL(f)
  })
}
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('decode failed'))
    img.src = src
  })
}
function scaled(w: number, h: number, maxEdge: number): [number, number] {
  const m = Math.max(w, h)
  if (m <= maxEdge) return [w, h]
  const s = maxEdge / m
  return [Math.round(w * s), Math.round(h * s)]
}
function draw(img: HTMLImageElement, w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  c.getContext('2d')?.drawImage(img, 0, 0, w, h)
  return c
}
export async function readImagePart(
  file: File,
  conversationId?: string,
  detail: ImageDetail = 'auto',
): Promise<ImagePart> {
  const raw = await readAsDataUrl(file)
  const id = uuid()
  const name = file.name || 'image'
  const mime = file.type || 'image/png'

  // The ORIGINAL bytes go to the attachments table and are what the send
  // path ships: the server reads EXIF metadata (GPS, capture time, camera),
  // applies orientation and fits to the vision budget off the true file.
  // The old pipeline canvas-re-encoded a "model copy" here, which silently
  // stripped every byte of metadata (the EXIF bug) - the only
  // canvas job left is the thumbnail. Uploads are localhost, so shipping
  // originals costs nothing that matters.
  let attachmentId = ''
  try {
    await attachmentsApi.put(id, file, mime, { name, conv: conversationId })
    attachmentId = id
  } catch {
    /* store unavailable - fall back to inline raw below */
  }

  // Thumbnail + dimensions (dimensions drive the size/cost menu). Browsers
  // decode <img> with EXIF orientation applied, so the thumb is upright.
  // GIFs skip the canvas (it flattens animation). Formats Chrome can't
  // decode (TIFF ...) get no thumb - the bubble renders a labeled tile
  // instead. The raw data URL used to stand in here, which both drew a
  // broken-image icon AND persisted the whole undecodable file into the
  // conversation doc on every save.
  let thumbUrl = ''
  let width: number | undefined
  let height: number | undefined
  try {
    const img = await loadImage(raw)
    width = img.naturalWidth
    height = img.naturalHeight
    if (file.type === 'image/gif') {
      thumbUrl = raw
    } else {
      const [tw, th] = scaled(width, height, THUMB_EDGE)
      thumbUrl = draw(img, tw, th).toDataURL('image/jpeg', 0.7)
    }
  } catch {
    // The browser cannot decode this one. HEIC is why: it is HEVC, which sits
    // in a patent pool, so no browser but Safari touches it - while an iPhone
    // writes it by default. Ask the manager for a viewable JPEG instead.
    //
    // Dimensions stay UNKNOWN rather than being taken from the rendition: a
    // rendition is fitted to a viewing size, so its size is not the photo's,
    // and the size/cost menu would quote a number that is simply wrong. The
    // menu already handles not knowing (TIFF has always arrived here).
    if (attachmentId) {
      try {
        thumbUrl = attachmentsApi.renditionUrl(attachmentId, THUMB_EDGE)
        // Confirm it actually renders before putting it in the doc - a 501
        // from an install with no decoder must leave a labeled tile, not a
        // broken-image icon.
        await loadImage(thumbUrl)
      } catch {
        thumbUrl = ''
      }
    }
  }

  return {
    type: 'image',
    attachmentId,
    mime,
    name,
    detail,
    width,
    height,
    thumbUrl,
    modelUrl: attachmentId ? undefined : raw,
    // Recorded on the PART, not re-derived at send: the file object is gone by
    // then, and a conversation reopened months later must still know this one
    // was never shown to the model.
    unreadable: isUnreadableImage(file) || undefined,
  }
}

// Document attachment (PDF, or any other file): the raw bytes go to the
// attachments table (viewer + the send-time fetch that builds the `input_file`
// part the server extracts). We deliberately do not inline the bytes in the
// conversation doc - a multi-MB file would bloat every debounced persist.
// `pages` is PDF-only, filled here (client-side pdfium/lector) for the chip;
// a fresh part just carries the reference + size.
export async function readFilePart(file: File, conversationId?: string): Promise<FilePart> {
  const id = uuid()
  const name = file.name || 'file'
  const mime = file.type || 'application/octet-stream'
  await attachmentsApi.put(id, file, mime, { name, conv: conversationId })
  let pages: number | undefined
  if (isPdfFile(file)) {
    try {
      const n = await pdfPageCount(await file.arrayBuffer())
      if (n > 0) pages = n
    } catch {
      /* page count unknown */
    }
  }
  return { type: 'file', attachmentId: id, mime, name, size: file.size, pages }
}

/** Store a .tvdb and return the part that records it on the message. The
 *  bytes go only to the attachments table - the graph session loads them into
 *  the WASM engine; nothing graph-shaped is ever inlined for a model. */
export async function readGraphPart(file: File, conversationId?: string): Promise<GraphPart> {
  const id = uuid()
  const name = file.name || 'graph.tvdb'
  await attachmentsApi.put(id, file, 'application/x-traverse-tvdb', { name, conv: conversationId })
  return { type: 'graph', attachmentId: id, name, size: file.size }
}

// An audio clip: the USER half of a transcription turn. Same shape
// of decision as a document - bytes to the attachments table, a reference in
// the conversation doc - because the clip is re-read at send time and must
// still be there for the player when the chat is revisited months later.
//
// `name` is deliberately allowed to be '': a microphone recording has no file
// name, and inventing one ("recording.webm") would make the sidebar title a
// fiction. The duration is read here rather than taken from the transcription
// so the chip and the scrub bar are honest before any model has answered.
export async function readAudioPart(
  file: Blob & { name?: string; type?: string },
  conversationId?: string,
  language?: string,
): Promise<AudioPart> {
  const id = uuid()
  const name = file.name ?? ''
  const mime = file.type || 'audio/webm'
  await attachmentsApi.put(id, file, mime, { name: name || 'recording', conv: conversationId })
  return {
    type: 'audio',
    attachmentId: id,
    mime,
    name,
    size: file.size,
    durationS: await audioDuration(file),
    language: language || undefined,
  }
}
