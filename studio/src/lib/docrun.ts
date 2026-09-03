// The document context of a conversation: which documents
// it holds, which one is SELECTED, and which turn's run drives the pane.
//
// SELECTION = TARGET: the selected document is both what
// the DocumentPane shows and what the mode chips read - "what you see is
// what you ask". Selection moves in exactly two ways: a newly sent document
// selects itself, and the user clicks a tab in the pane. Never by scroll,
// never by navigation.
import { reactive } from 'vue'
import type { Conversation, FilePart, ImagePart, Message } from '@/types/chat'
import type { ModelCaps } from '@/stores/models'
import { activeMessages } from '@/lib/tree'

export interface DocContext {
  images: ImagePart[]
  pdf?: FilePart
  /** a Word document - scriptor renders it in the pane, and unlike a PDF we
   *  never rasterize it, so it is not a `isRasterDoc` document */
  docx?: FilePart
  /** the user turn the document rode in on - its id is the selection key */
  source: Message
  /** the latest assistant run on this document (states/regions): after its
   *  source, before the next document's turn */
  run?: Message
}

/** True when this model only reads documents (capability; absent
 *  caps - stopped server, old chat - fall back to the turns' own evidence). */
export function isDocParserConv(
  conv: Conversation | null | undefined,
  caps: Record<string, ModelCaps | undefined>,
): boolean {
  if (!conv) return false
  if (caps[conv.model]?.docParser) return true
  // Every branch deliberately, unlike docContexts below: "has this conversation
  // ever parsed a document" is a fact about the chat, and branching away from
  // a run does not un-make it. Same question the manager answers in its `kind`
  // column, and the two must not disagree.
  return conv.messages.some((m) => m.role === 'assistant' && (m.docRun ?? m.ocr) !== undefined)
}

/** What counts as a PDF, in one place. The pane's document list and the chat
 *  chip that opens the pane must agree: if the chip is more generous, a click
 *  selects a document the pane does not have and nothing happens at all - a
 *  silent dead control. Name as well as mime, because an attachment that
 *  arrived without a content type is still plainly a PDF. */
export function isPdfPart(p: { mime?: string; name?: string }): boolean {
  return (p.mime ?? '').includes('pdf') || (p.name ?? '').toLowerCase().endsWith('.pdf')
}

/** What counts as a Word document - the same one-place contract as
 *  `isPdfPart`, and for the same reason: the pane's document list and the chip
 *  that opens the pane have to agree or the click selects an id the pane can't
 *  resolve. Mime as well as name, because an attachment that arrived without a
 *  content type is still plainly a .docx.
 *
 *  Only the OOXML .docx: scriptor reads an OPC package, and legacy .doc is a
 *  different (binary) format it cannot open. Those keep the dialog, which
 *  offers the download. */
export function isDocxPart(p: { mime?: string; name?: string }): boolean {
  return (
    (p.mime ?? '').includes('wordprocessingml') ||
    (p.name ?? '').toLowerCase().endsWith('.docx')
  )
}

/** Documents the parser lane can actually fan out: page images, or a PDF we
 *  rasterize page by page. A Word document is a pane document but not one of
 *  these - nothing renders it to pixels for a decoder, and its text reaches
 *  the model through the ordinary attachment path (sift extracts it server
 *  side). Without this split a .docx in a PaddleOCR chat would look to
 *  `maybeDocRun` like a document with zero pages. */
export function isRasterDoc(c: DocContext): boolean {
  return !!c.pdf || c.images.length > 0
}

/** Every document the conversation holds, in drop order, each with its own
 *  latest run. A run between two document turns belongs to the earlier one. */
export function docContexts(conv: Conversation | null | undefined): DocContext[] {
  if (!conv) return []
  const out: DocContext[] = []
  // The branch on screen: the pane shows the documents of the conversation you
  // are actually reading, and the "run between two documents" span below only
  // means anything along a single path.
  const msgs = activeMessages(conv)
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i]
    if (m.role !== 'user') continue
    const images = m.content.filter((p): p is ImagePart => p.type === 'image')
    const pdf = m.content.find((p): p is FilePart => p.type === 'file' && isPdfPart(p))
    const docx = m.content.find((p): p is FilePart => p.type === 'file' && isDocxPart(p))
    if (!images.length && !pdf && !docx) continue
    // close the previous document's span
    out.push({ images, pdf, docx, source: m })
  }
  // attach each document's latest run: the newest docRun/ocr assistant turn
  // between its source and the next document's source
  for (let d = 0; d < out.length; d++) {
    const from = msgs.indexOf(out[d].source)
    const to = d + 1 < out.length ? msgs.indexOf(out[d + 1].source) : msgs.length
    for (let j = to - 1; j > from; j--) {
      const a = msgs[j]
      if (a.role === 'assistant' && (a.docRun ?? a.ocr)) {
        out[d].run = a
        break
      }
    }
  }
  return out
}

/** The SELECTED document: the conversation's `activeDocId` when it still
 *  resolves, else the newest. Undefined only when no document exists. */
export function docContext(conv: Conversation | null | undefined): DocContext | undefined {
  return select(docContexts(conv), conv)
}

/** The selected document among the ones a parser lane can fan out - what the
 *  mode chips read and what a re-run re-reads. The pane uses `docContext`
 *  instead, because it can show a Word document the decoders can't. */
export function rasterContext(conv: Conversation | null | undefined): DocContext | undefined {
  return select(docContexts(conv).filter(isRasterDoc), conv)
}

function select(all: DocContext[], conv: Conversation | null | undefined): DocContext | undefined {
  if (!all.length) return undefined
  return all.find((c) => c.source.id === conv?.activeDocId) ?? all[all.length - 1]
}

/** A file part's page range ("5-12" / "7" / "5-", 1-based; pagesParam's
 *  vocabulary) as 0-based [start, endExclusive] against the real page count.
 *  Malformed or out-of-range falls back to the whole document. */
export function pageRangeBounds(range: string | undefined, count: number): [number, number] {
  if (!range) return [0, count]
  const m = /^(\d+)(-(\d+)?)?$/.exec(range.trim())
  if (!m) return [0, count]
  const a = Math.max(1, parseInt(m[1], 10))
  if (a > count) return [0, count]
  const b = m[3] ? Math.min(count, parseInt(m[3], 10)) : m[2] ? count : a
  return [a - 1, Math.max(a, b)]
}

/** The page images the pane has rendered, keyed by the document's
 *  source-message id - the result column crops figure/image regions out of
 *  these (the official demo places the document's own pictures inline).
 *  Dimensions ride along because a crop's aspect needs the page's. */
export interface PageImage {
  src: string
  w: number
  h: number
}
export const pageImages = reactive(new Map<string, (PageImage | null)[]>())
