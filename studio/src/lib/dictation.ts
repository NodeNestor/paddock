// Dictation into the composer, without fighting the caret.
//
// The session finalises an utterance every time you pause, so dictation is a
// stream of small commits rather than one paragraph at the end. Two rules make
// that pleasant instead of hostile:
//
// FINALISED text is inserted for real, at the END of the document, and the
// selection is left exactly where it was. So you can put the cursor three
// sentences back and fix a word while still speaking, and the next utterance
// lands after the text rather than on top of your edit.
//
// The open utterance is a widget DECORATION, not content. It is drawn in the
// document and is not in it: no undo step, no selection to land inside, no
// getText() contribution, nothing to clean up if the socket dies mid-word.
// Provisional text that lives in the document is provisional text you have to
// remember to remove, and the one time you forget it gets sent to a model.
// This is how Google Docs voice typing and macOS dictation behave too.
import { Extension } from '@tiptap/core'
import type { Editor } from '@tiptap/core'
import { Plugin, PluginKey, Selection } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

const key = new PluginKey<string>('dictation')

export const Dictation = Extension.create({
  name: 'dictation',
  addProseMirrorPlugins() {
    return [
      new Plugin({
        key,
        state: {
          init: () => '',
          // A meta-only transaction: the doc does not change, so tiptap does
          // not emit `update` and the draft/undo history never see it.
          apply: (tr, prev) => tr.getMeta(key) ?? prev,
        },
        props: {
          // Flag the editor while a ghost is up, so the placeholder can get out
          // of its way. Placeholder draws with `content: attr(data-placeholder)`
          // on a `float: left; height: 0` pseudo-element, which does not take
          // part in layout - so an empty composer drew "Send a message..." and
          // the incoming sentence on TOP of each other until the first
          // utterance finalised and the document stopped being empty.
          attributes: (state) => ({ class: key.getState(state) ? 'is-dictating' : '' }),
          decorations(state) {
            const text = key.getState(state)?.replace(/^\s+/, '')
            if (!text) return null
            const at = Selection.atEnd(state.doc).to
            const span = document.createElement('span')
            span.className = 'dictation-ghost'
            // The word boundary is decided here, against the live document,
            // because the document keeps changing under the ghost - you can
            // type while it is showing - and a space baked in when it was
            // rendered would be the wrong answer a keystroke later.
            const before = state.doc.textBetween(Math.max(0, at - 1), at)
            span.textContent = before && !/\s$/.test(before) ? ` ${text}` : text
            // `side: 1` puts the widget after anything else at this position,
            // which is what keeps a caret parked at the end of the line in
            // front of the ghost rather than behind it.
            return DecorationSet.create(state.doc, [
              Decoration.widget(at, span, { side: 1 }),
            ])
          },
        },
      }),
    ]
  },
})

/** Show (or clear, with '') the utterance still being spoken. */
export function setGhost(editor: Editor | null | undefined, text: string): void {
  if (!editor || editor.isDestroyed) return
  if (key.getState(editor.state) === text) return
  editor.view.dispatch(editor.state.tr.setMeta(key, text))
}

/** Append a finalised utterance, leaving the selection alone. */
export function appendDictated(editor: Editor | null | undefined, text: string): void {
  const said = text.trim()
  if (!editor || editor.isDestroyed || !said) return
  const at = Selection.atEnd(editor.state.doc).to
  // The word boundary is ours to add: an utterance is a sentence, not a
  // continuation, and `textBetween` on an empty doc gives '' rather than
  // undefined so the first one gets no stray leading space.
  const before = editor.state.doc.textBetween(Math.max(0, at - 1), at)
  const lead = before && !/\s$/.test(before) ? ' ' : ''
  editor.commands.insertContentAt(at, `${lead}${said}`, { updateSelection: false })
}
