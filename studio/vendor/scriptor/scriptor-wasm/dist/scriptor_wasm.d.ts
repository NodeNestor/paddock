/* tslint:disable */
/* eslint-disable */

/**
 * The result of a [`ScriptorDoc::relayout`]: the page dimensions (device px) so the browser can
 * size + lay out the page stack, plus the per-page fingerprints it diffs to decide which pages to
 * re-rasterize via [`ScriptorDoc::paint_page`].
 */
export class LayoutInfo {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The per-page content fingerprints (one per page). The caller compares against the previous
     * set and re-rasterizes only the pages whose value changed.
     */
    readonly fingerprints: BigUint64Array;
    readonly gap: number;
    readonly pageCount: number;
    readonly pageHeight: number;
    readonly pageWidth: number;
    readonly totalHeight: number;
}

/**
 * A paragraph's formatting, for the toolbar's Paragraph group. `align` is "" when unset;
 * `lineSpacing` (240ths) + indents are 0 when unset.
 */
export class ParaFmt {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly align: string;
    readonly indentFirst: number;
    readonly indentLeft: number;
    readonly indentRight: number;
    readonly lineSpacing: number;
}

/**
 * A live document held across the FFI boundary. Owns the CRDT replica and the canvas renderer;
 * the TS shell holds an opaque handle to it.
 */
export class ScriptorDoc {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Accept every tracked change in the document - body, header, and footer. Returns the total
     * count resolved.
     */
    acceptAll(): number;
    /**
     * Accept the tracked change under the caret `(para, off)` (insertion -> keep text, deletion ->
     * remove text). Returns whether one was resolved. Re-layout + re-paint after.
     */
    acceptChange(para: number, off: number): boolean;
    /**
     * Accept a specific revision id in the region of `para` (the inline click popup carries both the
     * click's paragraph and the id from [`track_at`]; revision ids are per-story, so the region picks
     * the right child document).
     */
    acceptRevision(para: number, id: number): boolean;
    /**
     * Add a named bookmark over codepoint `[start, end)` in paragraph `para`. The name should already be
     * a valid Word bookmark name (letters/digits/underscore, letter-initial); the caller sanitizes.
     * Re-paint after (bookmarks are invisible but become hyperlink targets).
     */
    addBookmark(para: number, start: number, end: number, name: string): void;
    /**
     * Add a comment over the selection `(start_para,start_off)..(end_para,end_off)` (one story) with
     * `text` as the body, attributed to the current author + last timestamp. Returns the new comment
     * id, or `-1` if the endpoints are in different stories / the story doesn't exist.
     */
    addComment(start_para: number, start_off: number, end_para: number, end_off: number, text: string): number;
    /**
     * Add a hyperlink over codepoint `[start, end)` in paragraph `para`, targeting `target` (an
     * external URL, or `#bookmarkName` for an internal jump). Re-layout + re-paint after.
     */
    addHyperlink(para: number, start: number, end: number, target: string): void;
    /**
     * Insert `text` at codepoint `off` in paragraph `para` as the **destination** of move `id`
     * (`w:moveTo`), pairing with a prior [`mark_move_source`](Self::mark_move_source). Re-paint after.
     */
    addMoveDest(para: number, off: number, text: string, id: number): void;
    /**
     * Create a new paragraph style (Word's New-Style / Save-Selection-as-a-Style) named `name`, based
     * on `based_on` (empty = no parent), with the given formatting (same per-field sentinels as
     * `setStyleProps`). Mints a unique style id from `name`, registers it (gallery + persistence), and
     * returns the id so the caller can apply it to the selected paragraph(s). Body story only.
     */
    addStyle(name: string, based_on: string, size: number, bold: number, italic: number, color: string, font: string, line_spacing: number, space_before: number, space_after: number, align: string, line_rule: string): string;
    /**
     * Encode an edit-stable anchor for a SELECTED RANGE `[(p1,o1), (p2,o2))`
     * (body codepoint offsets). Send it with an inline select->ask so the agent
     * edits exactly that span via the anchored `document_propose_edit`. The head
     * biases left, the tail right, so the range doesn't grow/shrink spuriously
     * when text is inserted at either edge.
     */
    anchorRange(p1: number, o1: number, p2: number, o2: number): Uint8Array;
    /**
     * Apply a bullet (`bullet = true`) / decimal numbered list to paragraph `para` (body only). The
     * Bullets / Numbering buttons. Re-layout + re-paint after.
     */
    applyList(para: number, bullet: boolean): void;
    /**
     * Apply a numbered list with a specific number format to paragraph `para` (body only): `numFmt` is
     * an OOXML token (`decimal` / `lowerLetter` / `upperLetter` / `lowerRoman` / `upperRoman`). The
     * Numbering button's format picker. Tracked as a `w:pPrChange` when Track-Changes is on.
     */
    applyListFormat(para: number, num_fmt: string): void;
    /**
     * Whether revision balloons are on.
     */
    balloonsOn(): boolean;
    /**
     * The body paragraph index where bookmark `name` begins, or `-1` - lets an internal hyperlink
     * (`#name`) jump to its target. Body only.
     */
    bookmarkParagraph(name: string): number;
    canRedo(): boolean;
    /**
     * Whether there is anything to undo / redo in the active story (for greying the toolbar buttons).
     */
    canUndo(): boolean;
    /**
     * Encode an edit-stable anchor at a body caret position (codepoint offset).
     * Send it as your cursor for presence, or capture it before a merge and
     * resolve it after to remap the local caret.
     */
    caretAnchor(para: number, off: number): Uint8Array;
    /**
     * The caret one visual line above/below `(para, off)`, keeping goal column `x` (device px) -
     * Word's ArrowUp/Down. Returns `[para, off]` (codepoints), or empty at the document edge. (The
     * shell's old one-pixel hit-test probe snapped back inside inter-paragraph spacing, so Up/Down
     * never crossed a paragraph boundary.)
     */
    caretLineStep(para: number, off: number, x: number, down: boolean): Uint32Array;
    /**
     * The caret rectangle for `(para, codepoint_offset)` as `[x, y, height]` (device px).
     */
    caretRect(para: number, off: number): Float32Array;
    /**
     * The caret cell's shading fill (RGB hex, no `#`), or `""` if not in a cell / no shading - lets the
     * UI pre-select the current colour.
     */
    cellShading(para: number): string;
    /**
     * The body paragraph index of the first paragraph of the cell one step forward (`forward=true`) /
     * backward from the caret's cell, or `-1` when `para` isn't in a cell / is at the table's last /
     * first cell. Drives Tab / Shift+Tab cell navigation. Body only (header/footer have no tables in v1).
     */
    cellStep(para: number, forward: boolean): number;
    /**
     * The margin change-bar rectangles from the last layout, flattened as `[page, x, y, w, h, para]`
     * per bar (device px; `y` is page-local within `page`). The editor hit-tests a click in the left
     * margin against these to drive Simple-Markup click-to-expand; `para` is the namespaced paragraph
     * the bar belongs to. Body + table-cell bars only (header/footer bars paint inline, not here).
     */
    changeBars(): Float32Array;
    /**
     * Clear every click-to-expand paragraph (on a display-mode change or a new document).
     */
    clearExpandedParagraphs(): void;
    /**
     * Clear all inline run formatting over `[start, end)` (the Clear Formatting eraser). Re-paint after.
     */
    clearFormatting(para: number, start: number, end: number): void;
    /**
     * Highlight rectangles (device px, flattened `[x,y,w,h,...]`) behind every text range anchored by
     * at least one *unresolved* comment, across body + header/footer. Editor chrome (not exported) -
     * the view paints these on its overlay like the selection.
     */
    commentRects(): Float32Array;
    /**
     * The comment ids anchored at `(para, off)` (the run under the caret, or one ending at it).
     * Region-routed; empty when the caret isn't inside a comment anchor.
     */
    commentsAt(para: number, off: number): Uint32Array;
    /**
     * Delete comment `id` and its replies (clearing the anchor in whichever story holds it).
     */
    deleteComment(id: number): boolean;
    /**
     * Delete codepoint range `[start, end)` in paragraph `para` as a direct human edit. No-op when
     * the range is empty. Call [`paint`] after.
     */
    deleteRange(para: number, start: number, end: number): void;
    /**
     * Delete the caret's table column. With Track-Changes on it's *marked* (`w:tcPr/w:cellDel` on the
     * column's cells, retained until accepted); otherwise removed (and the table if it empties).
     * Returns the caret paragraph after the edit, or `-1` if not in a table. Re-layout + re-paint.
     */
    deleteTableColumn(para: number): number;
    /**
     * Delete the caret's table row. With Track-Changes on it's *marked* (`w:trPr/w:del`, the row
     * survives until accepted); otherwise it's removed (and the table if it was the last row). Returns
     * the caret paragraph after the edit, or `-1` if `para` isn't in a table. Re-layout + re-paint.
     */
    deleteTableRow(para: number): number;
    /**
     * Ensure a default footer story exists (see [`ensure_header`](Self::ensure_header)) and return
     * the namespaced index of its first paragraph.
     */
    ensureFooter(): number;
    /**
     * Ensure a default header story exists - creating an empty one (a single blank paragraph) if the
     * document has none - and return the namespaced paragraph index of its first paragraph, so the
     * shell can drop the caret into the header to edit it (Word's Insert > Header). Idempotent: an
     * existing header is left untouched (its content preserved). Re-layout + paint after.
     */
    ensureHeader(): number;
    /**
     * Export the ops committed since `version` (from `oplogVersion`) as a loro
     * update delta to send to peers.
     */
    exportUpdatesSince(version: Uint8Array): Uint8Array;
    /**
     * The default footer as plain text.
     */
    footerText(): string;
    /**
     * Toggle/set bold over codepoint `[start, end)` in paragraph `para`. Re-paint after.
     */
    formatBold(para: number, start: number, end: number, on: boolean): void;
    /**
     * Set text color (RRGGBB hex) over `[start, end)`. Re-paint after.
     */
    formatColor(para: number, start: number, end: number, hex: string): void;
    /**
     * Set font family over `[start, end)`. Re-paint after.
     */
    formatFont(para: number, start: number, end: number, family: string): void;
    /**
     * Set / clear the highlight color over `[start, end)` (`""` clears it). Re-paint after.
     */
    formatHighlight(para: number, start: number, end: number, color: string): void;
    /**
     * Toggle/set italic over `[start, end)`. Re-paint after.
     */
    formatItalic(para: number, start: number, end: number, on: boolean): void;
    /**
     * Set font size (half-points, OOXML `w:sz`) over `[start, end)`. Re-paint after.
     */
    formatSize(para: number, start: number, end: number, half_points: number): void;
    /**
     * Toggle/set strikethrough over `[start, end)`. Re-paint after.
     */
    formatStrike(para: number, start: number, end: number, on: boolean): void;
    /**
     * Toggle/set underline over `[start, end)`. Re-paint after.
     */
    formatUnderline(para: number, start: number, end: number, on: boolean): void;
    /**
     * Set / clear vertical alignment over `[start, end)` ("superscript" / "subscript", or `""` to
     * clear back to baseline). Re-paint after.
     */
    formatVertAlign(para: number, start: number, end: number, value: string): void;
    /**
     * Build a document from a loro snapshot - the collaboration join message
     * from the server. Unlike the `constructor`, it seeds NO empty paragraph:
     * the snapshot is the authoritative content, and a seed would union a stray
     * blank paragraph into the merged document.
     */
    static fromSnapshot(snapshot: Uint8Array): ScriptorDoc;
    /**
     * The default header as plain text (one line per paragraph).
     */
    headerText(): string;
    /**
     * Map a canvas point (device px) to a caret position, returned as `[paragraph, codepoint]`.
     * Uses the geometry from the most recent [`paint`].
     */
    hitTest(x: number, y: number): Uint32Array;
    /**
     * The editable picture id whose hit-rect contains the canvas point `(x, y)` (absolute px), or
     * `None`. The topmost match wins (floats over inline) - the view uses this for click-to-select.
     */
    imageAtPoint(x: number, y: number): bigint | undefined;
    /**
     * Picture `id`'s crop as `[l, t, r, b]` (`<a:srcRect>`, thousandths of a percent - the share of
     * each edge cut), or `None`. The view seeds the crop window from this.
     */
    imageCrop(id: bigint): Int32Array | undefined;
    /**
     * The raw (encoded, uncropped) media bytes of picture `id`, or `None`. The view decodes these to
     * show the full image behind the crop window in crop mode (the page canvas only has the cropped
     * bitmap).
     */
    imageMedia(id: bigint): Uint8Array | undefined;
    /**
     * Picture `id`'s rect on the canvas as `[x, y, w, h]` (absolute px), or `None` if it isn't placed
     * (e.g. off-page). The view draws the selection box + resize handles from this.
     */
    imageRect(id: bigint): Float32Array | undefined;
    /**
     * Picture `id`'s wrap state as a single token for the Wrap Text menu + drag logic: `inline`
     * (in the flow), `square` / `tight` / `through` / `topAndBottom` (floating, text wraps), `behind`
     * (floating, behind the text), or `front` (floating, in front of the text). `None` if `id` is
     * unknown.
     */
    imageWrapState(id: bigint): string | undefined;
    /**
     * Insert a picture at codepoint `off` in **body** paragraph `para`: `bytes` (MIME `mime`, e.g.
     * `image/png`) ship as a fresh `word/media` part on save, shown at `w_emu` x `h_emu` (EMU; the
     * caller derives these from the decoded natural size via [`px_to_emu`]). Returns the new picture
     * id. Re-layout + repaint after. Images live on the body story only (header/footer not supported).
     */
    insertImage(para: number, off: number, bytes: Uint8Array, mime: string, w_emu: number, h_emu: number): bigint;
    /**
     * Insert a table column left (`right=false`) / right (`right=true`) of the caret's cell. With
     * Track-Changes on it's a tracked insertion (`w:tcPr/w:cellIns` on each new cell); otherwise
     * direct. Returns the new caret paragraph, or `-1` if not in a table. Re-layout + re-paint after.
     */
    insertTableColumn(para: number, right: boolean): number;
    /**
     * Insert a table row above (`below=false`) / below (`below=true`) the caret's row. With
     * Track-Changes on it's recorded as a tracked insertion (`w:trPr/w:ins`); otherwise direct.
     * Returns the new caret paragraph, or `-1` if `para` isn't in a table. Re-layout + re-paint after.
     */
    insertTableRow(para: number, below: boolean): number;
    /**
     * Insert `text` at codepoint `off` in paragraph `para` as a direct human edit, routed through
     * the shared `scriptor_edit::apply` path (the same one the agent uses). Call [`paint`] after.
     */
    insertText(para: number, off: number, text: string): void;
    /**
     * Insert a table of contents at body paragraph `at` (the caret), built from the document's current
     * headings (`Heading1`..`Heading9`): one `TOC{level}` line per heading - "{heading}\t{page}" -
     * wrapped as a real `TOC` field so Word can update it (F9). Page numbers come from a fresh layout
     * that already includes the inserted lines. Returns whether a TOC was inserted (`false` when there
     * are no headings, or `at` isn't a body paragraph). Re-layout + re-paint after.
     */
    insertToc(at: number): boolean;
    /**
     * Whether paragraph `para` is currently expanded (click-to-expand).
     */
    isParagraphExpanded(para: number): boolean;
    /**
     * Join paragraph `para` into the previous one (Backspace at paragraph start / Delete at end).
     * Returns the codepoint offset in the previous paragraph where the two met (the merged caret),
     * or `-1` when the join is refused because it would cross a table-cell boundary (the caller
     * should leave the caret where it is). Routed through the shared `scriptor_edit::apply` path.
     */
    joinParagraph(para: number): number;
    /**
     * The hyperlink target at codepoint `off` in paragraph `para` (external URL or `#bookmarkName`),
     * or `""` when the caret isn't on a link - lets the toolbar reflect / the caret follow it.
     */
    linkAt(para: number, off: number): string;
    /**
     * Every tracked change across stories as a JSON array (for the reviewing pane): each object has
     * `id`, `kind` (`"ins"` / `"del"` / `"fmt"` / `"movefrom"` / `"moveto"` for run changes;
     * `"rowins"` / `"rowdel"` / `"colins"` / `"coldel"` for table-structure changes), `author`,
     * `date`, `text`, and the caret `para` (namespaced) / `off`. Run changes come from each story's
     * `change_carets` + `track_at`; table changes from `table_changes()`. Sorted in document order.
     * (Comments come from [`list_comments`](Self::list_comments) - the pane merges the two.)
     */
    listChanges(): string;
    /**
     * Every comment as a JSON array string (for the popover / reviewing list): each object has
     * `id`, `author`, `initials`, `date`, `text`, `parent` (id or null), `resolved`, and the anchor
     * caret `para` / `off` (namespaced; `-1` when un-anchored). Replies inherit the parent's anchor.
     */
    listComments(): string;
    /**
     * Mark codepoint range `[start, end)` in paragraph `para` as the **source** of a move
     * (`w:moveFrom`, text retained), returning the move's revision id (or `-1` for an empty range /
     * missing story). The matching destination is added with [`add_move_dest`](Self::add_move_dest)
     * using this id - the two-step path the editor's cut-then-paste move uses. Re-paint after.
     */
    markMoveSource(para: number, start: number, end: number): number;
    /**
     * Merge a remote loro blob (a snapshot or an update delta) into this replica.
     * Loro merges are commutative + idempotent, so order does not matter and a
     * re-merge is a no-op. The caller re-renders afterward (`relayout` re-reads
     * the model); pair with `caretAnchor` + `resolveAnchor` to keep the local
     * caret on the same character when a concurrent edit shifts offsets.
     */
    merge(bytes: Uint8Array): void;
    /**
     * Merge the caret's cell with the `count - 1` cells below it (vertical `w:vMerge` merge). Returns
     * the caret after the merge, or `-1` if not in a table / not enough rows. Re-layout after.
     */
    mergeCellsDown(para: number, count: number): number;
    /**
     * Merge the caret's cell with the `count - 1` cells to its right (horizontal `w:gridSpan` merge).
     * Returns the caret after the merge, or `-1` if not in a table / not enough cells. Re-layout after.
     */
    mergeCellsRight(para: number, count: number): number;
    /**
     * Move codepoint range `[from_start, from_end)` in paragraph `from_para` to codepoint `to_off` in
     * paragraph `to_para` as a tracked move (`w:moveFrom` source + `w:moveTo` destination, one shared
     * revision id). Both endpoints must be in the same story (body / header / footer); the destination
     * must lie outside the source range. Returns the move's revision id, or `-1` when the endpoints
     * span stories / the range is empty / the move is into itself. Re-paint after.
     */
    moveRange(from_para: number, from_start: number, from_end: number, to_para: number, to_off: number): number;
    /**
     * Move the caret's table column left (`left=true`) / right one position. Returns the caret
     * paragraph after the move, or `-1` if not in a table / the move runs off the edge.
     */
    moveTableColumn(para: number, left: boolean): number;
    /**
     * Move the caret's table row up (`up=true`) / down one position (a direct structural reorder).
     * Returns the caret paragraph after the move, or `-1` if not in a table / the move runs off the
     * edge. Re-layout + re-paint after.
     */
    moveTableRow(para: number, up: boolean): number;
    /**
     * Start an empty document (no file) - used to author a fresh doc in the editor. Seeds one
     * empty paragraph so the caret has a block to type into (an editor never has zero paragraphs).
     */
    constructor();
    /**
     * The caret `[para, off]` of the next tracked change after `(para, off)`, searched **across all
     * stories** (body + header + footer) and wrapping, or an empty array when the document has no
     * tracked changes. For Review > Next.
     */
    nextChange(para: number, off: number): Uint32Array;
    /**
     * Caret `[para, off]` of the next (`forward`) / previous comment anchor across stories (wraps),
     * or an empty array when the document has no comments. For Review > Next/Previous comment.
     */
    nextComment(para: number, off: number): Uint32Array;
    /**
     * Open a `.docx` (the raw OPC zip bytes, e.g. from a `File`) into the CRDT model.
     */
    static openDocx(bytes: Uint8Array): ScriptorDoc;
    /**
     * The current oplog version, encoded. Hold it, then `exportUpdatesSince` to
     * ship only the ops committed since - the efficient wire delta.
     */
    oplogVersion(): Uint8Array;
    /**
     * Page geometry in twips: `[width, height, marginTop, marginRight, marginBottom, marginLeft,
     * headerDist, footerDist]`. For the ruler + the Layout tab's page-size / margin controls.
     */
    pageGeometry(): Uint32Array;
    /**
     * Rasterize a single page (0-based) of the current layout: an opaque white sheet with that
     * page's text. Returns RGBA8 (`page_width*page_height*4`); the browser blits it at
     * `y = index*(page_height+gap)`. Call after [`relayout`], only for pages whose fingerprint
     * changed. Also refreshes the page's raster cache, so a later [`Self::paint_page_band`] diffs
     * against exactly what the canvas shows.
     */
    paintPage(index: number): Uint8Array;
    /**
     * [`Self::paint_page`], returning only the vertical band of rows that actually CHANGED since
     * the page's last raster: an 8-byte little-endian header `[y0, y1)` followed by `(y1-y0)` rows
     * of RGBA. Typing edits one paragraph, so shipping the whole ~3-4MB page across the wasm->JS
     * boundary per keystroke was mostly wasted transfer + GC; the band is pixel-diffed against the
     * cached previous raster, so it can never miss a visual change (no raster cached, or a size
     * change, degrades to the full page; nothing changed returns an empty `[0, 0)` band). Only
     * valid when the caller's canvas still shows this page's previous raster.
     */
    paintPageBand(index: number): Uint8Array;
    /**
     * Number of paragraphs in the **body** (back-compat for callers that don't track regions). For
     * region-aware caret bounds use [`paragraph_range`]. Served from the cached texts (O(1)) with a
     * fallback to materializing the tree before the first render.
     */
    paragraphCount(): number;
    /**
     * The paragraph-level formatting of paragraph `para` (for the Paragraph-group toolbar state).
     */
    paragraphFormat(para: number): ParaFmt;
    /**
     * The codepoint length of paragraph `para` (for caret clamping + cross-paragraph movement).
     */
    paragraphLength(para: number): number;
    /**
     * Paragraph `para`'s list level-0 number format (`"decimal"` / `"lowerRoman"` / `"bullet"` / ...),
     * or `""` when it isn't in a list - lets the Numbering format picker check the active format.
     */
    paragraphListFormat(para: number): string;
    /**
     * The kind of list paragraph `para` is in: `"bullet"`, `"number"`, or `""` (not a list) - lets the
     * toolbar toggle the Bullets / Numbering buttons like Word.
     */
    paragraphListKind(para: number): string;
    /**
     * Paragraph `para`'s list level (`w:numPr/w:ilvl`, 0-8), or `-1` when it isn't in a list - lets
     * Tab / Shift+Tab demote / promote a list item to the next / previous level.
     */
    paragraphListLevel(para: number): number;
    /**
     * Paragraph `para`'s current list id (`w:numPr/w:numId`), or `-1` when it isn't in a list - lets
     * the toolbar reflect / toggle the numbering state.
     */
    paragraphNumId(para: number): number;
    /**
     * The 0-based page index a paragraph sits on (from the last layout) - for "Page X of N". Body
     * paragraphs come from `placements`; table-cell paragraphs (which have no placement, only caret
     * geometry) are found by scanning the placed cells.
     */
    paragraphPage(para: number): number;
    /**
     * The `[firstIndex, count]` of the story (body / header / footer) that `para` belongs to - so the
     * JS shell can clamp caret movement to one story (a header caret can't arrow into the body). The
     * first index is the region's namespace base; `count` is its paragraph count.
     */
    paragraphRange(para: number): Uint32Array;
    /**
     * Paragraph `para`'s current named style id (`w:pStyle`), or `""` for the default (Normal) - lets
     * the Styles dropdown reflect the caret's paragraph.
     */
    paragraphStyle(para: number): string;
    /**
     * The concatenated plain text of paragraph `index` (namespaced). Served from the cached texts
     * when available (avoids re-materializing the tree per call).
     */
    paragraphText(index: number): string;
    /**
     * The caret `[para, off]` of the previous tracked change before `(para, off)`, across all
     * stories (wraps).
     */
    prevChange(para: number, off: number): Uint32Array;
    prevComment(para: number, off: number): Uint32Array;
    /**
     * Redo the last undone edit (Ctrl+Y / Ctrl+Shift+Z) in the active story. Returns whether anything
     * changed.
     */
    redo(): boolean;
    /**
     * Reject every tracked change in the document - body, header, and footer. Returns the total count.
     */
    rejectAll(): number;
    /**
     * Reject the tracked change under the caret (insertion -> remove text, deletion -> keep text).
     */
    rejectChange(para: number, off: number): boolean;
    /**
     * Reject a specific revision id in the region of `para`.
     */
    rejectRevision(para: number, id: number): boolean;
    /**
     * Render the whole document to a [`PaintResult`] (RGBA8 + dimensions) at the document's real
     * Re-resolve + lay out the whole document at the document's real page geometry (`w:sectPr` size
     * + margins), WITHOUT rasterizing - the cheap pass run on every edit. `scale` is the device-
     * pixel ratio. Each run's size / bold / italic / color is resolved from inline run formatting
     * over the paragraph's `styles.xml` style (so headings / title get their real sizing). Returns
     * a [`LayoutInfo`] (page dimensions + per-page fingerprints); the caller diffs the fingerprints
     * and calls [`paint_page`] only for the pages that changed.
     */
    relayout(scale: number): LayoutInfo;
    /**
     * Remove the hyperlink at codepoint `off` in paragraph `para`. Returns whether one was removed.
     */
    removeHyperlink(para: number, off: number): boolean;
    /**
     * Remove picture `id`. Under Track Changes this is a tracked deletion (the run is marked `w:del`,
     * retained until accepted); otherwise the run + placement are dropped outright. Returns whether it
     * existed. Re-layout after.
     */
    removeImage(id: bigint): boolean;
    /**
     * Reply to comment `parent` (a threaded child sharing the parent's anchor). Returns the new id.
     */
    replyComment(parent: number, text: string): number;
    /**
     * Reset picture `id`'s crop (Word's "Reset Crop"): clear `<a:srcRect>` and restore the display
     * extent so the whole image reappears at the same scale. Returns whether it was cropped.
     * Re-layout + repaint after.
     */
    resetImageCrop(id: bigint): boolean;
    /**
     * Resolve an anchor (from `caretAnchor`) to a current body `[para, off]`, or
     * `undefined` if the anchored block was deleted. Both a live and a shifted
     * anchor return a position (the caret follows the content); only a deleted
     * block returns nothing.
     */
    resolveAnchor(anchor: Uint8Array): Uint32Array | undefined;
    /**
     * Mark comment `id`'s thread resolved / unresolved. Returns whether it existed.
     */
    resolveComment(id: number, resolved: boolean): boolean;
    /**
     * Resolve an anchored range (from `anchorRange`) to current body coordinates
     * `[p1, o1, p2, o2]`, or `undefined` if it no longer resolves.
     */
    resolveRange(range: Uint8Array): Uint32Array | undefined;
    /**
     * The *resolved* definition of paragraph style `id` as JSON, for prefilling the Modify-Style
     * dialog: `{"size"(half-pts,0=inherit),"bold","italic","color"(hex,""=inherit),"font"(""=inherit),
     * "lineSpacing"(240ths,0=inherit),"spaceBefore"(twips,-1=inherit),"spaceAfter"(twips,-1=inherit)}`.
     * Resolved through the style's `basedOn` chain over docDefaults, with any runtime edit folded in -
     * so the dialog opens showing what the style currently renders at.
     */
    resolveStyleProps(id: string): string;
    /**
     * Every reviewer who authored a tracked change or comment, as a JSON array (the "Show Markup"
     * legend): each object has `name`, `color` (the author's hue as `#rrggbb`), and `hidden`. Sorted
     * by name.
     */
    reviewers(): string;
    /**
     * The resolved formatting of codepoint `[start, end)` in paragraph `para`, for driving toolbar
     * state. Boolean getters are tri-state via `*IsMixed` (true = the selection spans both).
     */
    selectionFormat(para: number, start: number, end: number): SelFormat;
    /**
     * Selection highlight rectangles between two caret positions (codepoint offsets), flattened
     * `[x, y, w, h, ...]` (device px). Empty when the selection is collapsed.
     */
    selectionRects(p1: number, o1: number, p2: number, o2: number): Float32Array;
    /**
     * Set which story the caret is in (from a namespaced paragraph index), so undo/redo route to the
     * right child document. The JS shell calls this on every selection change.
     */
    setActiveStory(para: number): void;
    /**
     * Set paragraph alignment ("left" | "center" | "right" | "justify"). Re-paint after.
     */
    setAlignment(para: number, align: string): void;
    /**
     * Set the current author: a stable `id` (audit trail) + a display `name` (stamped as `w:author`
     * on tracked changes, and shown in the hover tooltip).
     */
    setAuthor(id: string, name: string): void;
    /**
     * Turn revision balloons on/off (Word's "Show Revisions in Balloons"). When on, tracked deletions
     * move from the line into right-margin bubbles; it only takes visible effect in the markup display
     * modes (All / Simple). Re-layout + paint after.
     */
    setBalloons(on: boolean): void;
    /**
     * Set the caret cell's shading fill (`fill` = RGB hex without `#`; `""` clears it). With
     * Track-Changes on it's a tracked cell-property change (`w:tcPrChange`); otherwise direct. Returns
     * whether the caret was in a table cell (so the caller re-layouts). Re-layout + re-paint after.
     */
    setCellShading(para: number, fill: string): boolean;
    /**
     * Replace the default footer with plain `text`. Re-paint after.
     */
    setFooterText(text: string): void;
    /**
     * Set the page whose header/footer instance the caret is on (the JS shell computes it from the
     * click on a multi-page document). Lets the caret resolve to that instance, not always page 1.
     */
    setHeaderFooterPage(page: number): void;
    /**
     * Replace the default header with plain `text`. Re-paint after.
     */
    setHeaderText(text: string): void;
    /**
     * Set picture `id`'s crop (`<a:srcRect>` l/t/r/b, thousandths of a percent, 0..100000 - the share
     * of each edge to cut). Returns whether it existed. Re-layout + repaint after.
     */
    setImageCrop(id: bigint, l: number, t: number, r: number, b: number): boolean;
    /**
     * Make picture `id` floating (positioned + text-wrapped) or inline (in the flow). `wrap` is the
     * wrap type (`square` / `tight` / `topAndBottom` / `through` / `none`); `behind` paints it under
     * the text. Returns whether it existed. Re-layout after.
     */
    setImageFloating(id: bigint, floating: boolean, wrap: string, behind: boolean): boolean;
    /**
     * Position floating picture `id`: `h_from`/`v_from` are the `relativeFrom` origins
     * (`column`/`page`/`margin`/...) and `x_emu`/`y_emu` the offset from it (EMU). Clears any
     * alignment (an explicit offset wins, as in Word's drag-to-move). No-op on an inline picture.
     * Returns whether it existed and was floating. Re-layout after.
     */
    setImagePosition(id: bigint, h_from: string, x_emu: number, v_from: string, y_emu: number): boolean;
    /**
     * Resize picture `id` to `w_emu` x `h_emu` (EMU). Returns whether it existed. Re-layout after.
     */
    setImageSize(id: bigint, w_emu: number, h_emu: number): boolean;
    /**
     * Set the first-line indent (twips; negative = hanging). Re-paint after.
     */
    setIndentFirst(para: number, twips: number): void;
    /**
     * Set the left indent (twips). Re-paint after.
     */
    setIndentLeft(para: number, twips: number): void;
    /**
     * Set the right indent (twips). Re-paint after.
     */
    setIndentRight(para: number, twips: number): void;
    /**
     * Set page orientation (true = landscape); swaps the page dimensions if needed.
     */
    setLandscape(landscape: boolean): void;
    /**
     * Set line spacing in 240ths (240 = single, 360 = 1.5, 480 = double). Re-paint after.
     */
    setLineSpacing(para: number, x240: number): void;
    /**
     * Set the page margins in twips (1 inch = 1440).
     */
    setMargins(top: number, right: number, bottom: number, left: number): void;
    /**
     * Hand the engine the current wall-clock time (ISO-8601) to stamp on the next tracked change.
     * The engine never invents time; the JS shell calls this with `new Date().toISOString()` before
     * a tracked edit.
     */
    setNow(iso: string): void;
    /**
     * Set (or clear) paragraph `para`'s list numbering (`w:numPr`): `num_id < 0` removes it from any
     * list; otherwise it joins list `num_id` at level `ilvl` (a negative `ilvl` defaults to 0). With
     * Track-Changes on this records a `w:pPrChange` (a numbering change is a paragraph-property
     * change); otherwise it applies directly. Routed through the shared edit path. Re-layout + re-paint.
     */
    setNumbering(para: number, num_id: number, ilvl: number): void;
    /**
     * Set the page size in twips (Letter = 12240x15840, A4 = 11906x16838).
     */
    setPageSize(width: number, height: number): void;
    /**
     * Set (or clear, when `style` is empty -> Normal) paragraph `para`'s named style (`w:pStyle`).
     * With Track-Changes on this records a `w:pPrChange` (a style change is a paragraph-property
     * change); otherwise it applies directly. Routed through the shared edit path. Re-layout + re-paint.
     */
    setParagraphStyle(para: number, style: string): void;
    /**
     * Filter a reviewer's markup in / out of the display by `w:author` name (display-only; the model
     * is untouched). Hidden reviewers' tracked changes + comments are suppressed on the next
     * [`relayout`]. Re-layout + re-paint after.
     */
    setReviewerHidden(author: string, hidden: boolean): void;
    /**
     * Set the caret row's height in twips (`twips = 0` clears it; `exact` = exact rule, else at-least).
     * Tracked as `w:trPrChange` when Track-Changes is on. Returns whether the caret was in a table row.
     */
    setRowHeight(para: number, twips: number, exact: boolean): boolean;
    /**
     * Edit style `id`'s *definition* (Word's Modify-Style): every paragraph resolving through `id`
     * re-renders with the new properties. Per-field merge - each argument is a sentinel meaning
     * "leave this field unchanged" so the dialog can write only what the user touched:
     * `size`/`line_spacing`/`space_before`/`space_after` < 0 = unchanged (else the value);
     * `bold`/`italic` < 0 = unchanged, 0 = off, 1 = on; `color`/`font` empty = unchanged. Direct, not
     * a tracked revision (Word doesn't redline a style-definition change). Body story only. Re-layout.
     */
    setStyleProps(id: string, size: number, bold: number, italic: number, color: string, font: string, line_spacing: number, space_before: number, space_after: number, align: string, line_rule: string): void;
    /**
     * Set a uniform single-line border on every edge of the caret's table (`size_eighths` = line
     * weight in eighths of a point, `0` removes all borders; `color` = RGB hex without `#`). Tracked as
     * `w:tblPrChange` when Track-Changes is on. Returns whether the caret was in a table.
     */
    setTableBorders(para: number, size_eighths: number, color: string): boolean;
    /**
     * Turn Track-Changes (suggesting) mode on/off. While on, typing / deleting author tracked
     * changes attributed to the current author instead of editing the document directly. Ignored when
     * tracking is **locked** (see [`set_track_locked`](Self::set_track_locked)) - it stays on.
     */
    setTrackChanges(on: boolean): void;
    /**
     * Set how tracked changes are displayed: `all` (insertions underlined + deletions struck, in
     * author colours), `simple`/`none` (deletions hidden - the Final view), or `original`
     * (insertions hidden). Unknown values are ignored. Call [`relayout`] + re-paint after. The
     * non-`all` modes are render/preview only: the caret geometry still indexes the full
     * (All-Markup) text, so edit in `all`.
     */
    setTrackDisplay(mode: string): void;
    /**
     * Lock / unlock Track-Changes (Review > Lock Tracking): while locked, tracking can't be turned
     * off (and is forced on). v1 is session state, not yet persisted to `settings.xml`.
     */
    setTrackLocked(locked: boolean): void;
    /**
     * A full, self-contained snapshot of the document (history + state) - what a
     * joining client ships, and the merge unit the server sends on join.
     */
    snapshot(): Uint8Array;
    /**
     * Split (unmerge) the caret's horizontally-merged cell back into single columns. Returns the caret,
     * or `-1` if not in a table / the cell isn't merged.
     */
    splitCellHorizontal(para: number): number;
    /**
     * Split (unmerge) the caret's vertically-merged cell. Returns the caret, or `-1` if not in a table /
     * the cell isn't a vertical-merge anchor.
     */
    splitCellVertical(para: number): number;
    /**
     * Split paragraph `para` at codepoint `off` (the Enter key) - text from `off` onward moves to a
     * new paragraph after it. Routed through the shared `scriptor_edit::apply` path. Re-paint after.
     */
    splitParagraph(para: number, off: number): void;
    /**
     * The Styles gallery as a JSON array (Title / Subtitle / Heading N / Normal / ... - the document's
     * quick styles), for the Home tab's Styles gallery. Each entry carries the style's resolved preview
     * formatting so the gallery can render each name in its own look:
     * `{"id","name","size"(half-points,0=inherit),"bold","italic","color"(hex,""=inherit),"font"}`.
     */
    styleGallery(): string;
    /**
     * Table context for paragraph `para`: `[row, col, rowCount, colCount]` (cell indices), or an
     * empty array when the paragraph isn't inside a table. Drives the table context menu.
     */
    tableContext(para: number): Uint32Array;
    /**
     * The current document serialized to OOXML `word/document.xml` (the edited body). Hook for
     * "save"; full `.docx` re-packaging (re-zip with the source's other parts) is a follow-up.
     */
    toDocumentXml(): string;
    /**
     * Save the whole document to `.docx` bytes - the original package re-zipped with the edited
     * body + header/footer parts (or a minimal package for a from-scratch document).
     */
    toDocx(): Uint8Array;
    /**
     * Toggle whether paragraph `para` is expanded to inline All-Markup while the document is in
     * Simple Markup (click-to-expand). Returns the new state. Re-layout + paint after. The override is
     * only consulted in Simple Markup, but the toggle is always recorded.
     */
    toggleParagraphExpanded(para: number): boolean;
    /**
     * The tracked change under `(para, off)` for the hover tooltip + click popup, or `undefined`
     * when the point isn't over a change.
     */
    trackAt(para: number, off: number): TrackHit | undefined;
    /**
     * Whether Track-Changes mode is on.
     */
    trackChangesOn(): boolean;
    /**
     * Whether Track-Changes is locked on.
     */
    trackLocked(): boolean;
    /**
     * Undo the last local edit (Ctrl+Z) in the **active story** (body / header / footer - each child
     * owns its own undo history). Returns whether anything changed. Re-paint after. The active story
     * is set from the caret via [`set_active_story`](Self::set_active_story).
     */
    undo(): boolean;
    /**
     * Update (regenerate) the document's TOC in place: delete the old field block, then rebuild it from
     * the current headings + page numbers (Word's F9). When there's no existing TOC, insert one at the
     * caret `at` instead. Returns whether a TOC was written. Re-layout + re-paint after.
     */
    updateToc(at: number): boolean;
    /**
     * Total word count across the document, computed in a single pass. (The TS shell previously
     * looped `paragraphText` per paragraph, which re-materialized the whole tree each time - O(n^2),
     * and a UI freeze once tables put 100+ paragraphs in the flow.)
     */
    wordCount(): number;
}

/**
 * The selection's resolved formatting, for the toolbar. Each boolean has a companion `*Mixed`
 * getter (true when the selection spans both states); `size` is 0 when mixed/unset; `color` /
 * `font` are empty strings when mixed/unset.
 */
export class SelFormat {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly bold: boolean;
    readonly boldMixed: boolean;
    /**
     * Text color `RRGGBB`; empty when mixed or unset.
     */
    readonly color: string;
    /**
     * Font family; empty when mixed or unset.
     */
    readonly font: string;
    /**
     * Highlight color name; empty when none or mixed.
     */
    readonly highlight: string;
    readonly italic: boolean;
    readonly italicMixed: boolean;
    /**
     * Font size in half-points (OOXML `w:sz`); 0 when mixed or unset.
     */
    readonly size: number;
    readonly strike: boolean;
    readonly strikeMixed: boolean;
    readonly underline: boolean;
    readonly underlineMixed: boolean;
    /**
     * Vertical alignment ("superscript" / "subscript"); empty when baseline or mixed.
     */
    readonly vertAlign: string;
}

/**
 * A tracked change under a point (hover tooltip + click popup): the revision id, its kind (`"ins"`
 * or `"del"`), the author, the ISO date, and the change's text.
 */
export class TrackHit {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly author: string;
    readonly date: string;
    readonly id: number;
    /**
     * `"ins"` (insertion) or `"del"` (deletion).
     */
    readonly kind: string;
    readonly text: string;
}

/**
 * Compare two `.docx` documents (blacklining): produce a **redline** - `original` with every
 * difference as an author-attributed tracked change - plus the change manifest. Returns a
 * `{ redline: Uint8Array, manifest: string }` object: `redline` is a Word-openable tracked-changes
 * `.docx` the view can open like any document (its changes then appear in the reviewing pane);
 * `manifest` is the deterministic change set as JSON (`{"changes":[…]}`) the UI parses for a
 * summary / change-list. The redline is attributed to `author` and dated `date` (a parameter, so the
 * result is deterministic).
 */
export function compareDocx(original: Uint8Array, revised: Uint8Array, author: string, date: string, detect_formatting: boolean, detect_moves: boolean, ignore_whitespace: boolean, ignore_case: boolean): any;

/**
 * EMU (English Metric Units, 914400/inch - the unit the image model + `.docx` speak) -> canvas px at
 * zoom `scale` (1.0 = 96 px/in). The view sizes selection handles + draws crop overlays in px, so it
 * converts at this boundary rather than hard-coding the magic numbers. Natural-image-size conversion
 * (DPI-independent, 96 px/in) passes `scale = 1.0`; on-screen handle math passes the current zoom.
 */
export function emuToPx(emu: number, scale: number): number;

/**
 * Every bundled substitute face, so the DOM chrome can register `@font-face` rules and preview a
 * font / style menu in the SAME clone the canvas renders (true WYSIWYG - the OS has none of these MS
 * fonts installed). One entry per face: `family` is the MS name it substitutes for (so a CSS
 * `font-family:'Cambria'` label draws in Caladea, matching what the shaper paints), `bold`/`italic`
 * are the style flags, and `bytes` is the raw font data (the exact bytes embedded in this module -
 * no second copy shipped as a web asset). The DejaVu broad-Unicode fallback is skipped: it stands in
 * for no MS family (`substitute_family` never returns it), so it is never a menu entry.
 */
export function fontFaces(): Array<any>;

/**
 * Canvas px at zoom `scale` -> EMU (the inverse of [`emu_to_px`]). The view turns a resize-handle
 * drag (px at the current zoom) or a decoded natural size (px at `scale = 1.0`) into the EMU the
 * edit ops want. Returns 0 for a non-positive scale (no sensible inverse).
 */
export function pxToEmu(px: number, scale: number): number;

/**
 * Route Rust panics to the browser console (dev ergonomics). Runs on module init.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_layoutinfo_free: (a: number, b: number) => void;
    readonly __wbg_parafmt_free: (a: number, b: number) => void;
    readonly __wbg_scriptordoc_free: (a: number, b: number) => void;
    readonly __wbg_selformat_free: (a: number, b: number) => void;
    readonly __wbg_trackhit_free: (a: number, b: number) => void;
    readonly compareDocx: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number, number];
    readonly emuToPx: (a: number, b: number) => number;
    readonly fontFaces: () => any;
    readonly layoutinfo_fingerprints: (a: number) => [number, number];
    readonly layoutinfo_gap: (a: number) => number;
    readonly layoutinfo_pageCount: (a: number) => number;
    readonly layoutinfo_pageHeight: (a: number) => number;
    readonly layoutinfo_pageWidth: (a: number) => number;
    readonly layoutinfo_totalHeight: (a: number) => number;
    readonly parafmt_align: (a: number) => [number, number];
    readonly parafmt_lineSpacing: (a: number) => number;
    readonly pxToEmu: (a: number, b: number) => number;
    readonly scriptordoc_anchorRange: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly scriptordoc_caretAnchor: (a: number, b: number, c: number) => [number, number, number, number];
    readonly scriptordoc_exportUpdatesSince: (a: number, b: number, c: number) => [number, number, number, number];
    readonly scriptordoc_fromSnapshot: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_merge: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_new: () => number;
    readonly scriptordoc_openDocx: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_oplogVersion: (a: number) => [number, number];
    readonly scriptordoc_resolveAnchor: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_resolveRange: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_snapshot: (a: number) => [number, number, number, number];
    readonly selformat_bold: (a: number) => number;
    readonly selformat_boldMixed: (a: number) => number;
    readonly selformat_color: (a: number) => [number, number];
    readonly selformat_font: (a: number) => [number, number];
    readonly selformat_highlight: (a: number) => [number, number];
    readonly selformat_italic: (a: number) => number;
    readonly selformat_italicMixed: (a: number) => number;
    readonly selformat_size: (a: number) => number;
    readonly selformat_strike: (a: number) => number;
    readonly selformat_strikeMixed: (a: number) => number;
    readonly selformat_underline: (a: number) => number;
    readonly selformat_underlineMixed: (a: number) => number;
    readonly selformat_vertAlign: (a: number) => [number, number];
    readonly trackhit_author: (a: number) => [number, number];
    readonly trackhit_date: (a: number) => [number, number];
    readonly trackhit_id: (a: number) => number;
    readonly trackhit_kind: (a: number) => [number, number];
    readonly trackhit_text: (a: number) => [number, number];
    readonly start: () => void;
    readonly parafmt_indentFirst: (a: number) => number;
    readonly parafmt_indentLeft: (a: number) => number;
    readonly parafmt_indentRight: (a: number) => number;
    readonly scriptordoc_imageAtPoint: (a: number, b: number, c: number) => [number, bigint];
    readonly scriptordoc_imageCrop: (a: number, b: bigint) => [number, number];
    readonly scriptordoc_imageMedia: (a: number, b: bigint) => [number, number];
    readonly scriptordoc_imageRect: (a: number, b: bigint) => [number, number];
    readonly scriptordoc_imageWrapState: (a: number, b: bigint) => [number, number];
    readonly scriptordoc_insertImage: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [bigint, number, number];
    readonly scriptordoc_removeImage: (a: number, b: bigint) => [number, number, number];
    readonly scriptordoc_resetImageCrop: (a: number, b: bigint) => [number, number, number];
    readonly scriptordoc_setImageCrop: (a: number, b: bigint, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly scriptordoc_setImageFloating: (a: number, b: bigint, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly scriptordoc_setImagePosition: (a: number, b: bigint, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly scriptordoc_setImageSize: (a: number, b: bigint, c: number, d: number) => [number, number, number];
    readonly scriptordoc_addBookmark: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_addHyperlink: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_addMoveDest: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_addStyle: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => [number, number, number, number];
    readonly scriptordoc_bookmarkParagraph: (a: number, b: number, c: number) => number;
    readonly scriptordoc_canRedo: (a: number) => number;
    readonly scriptordoc_canUndo: (a: number) => number;
    readonly scriptordoc_cellShading: (a: number, b: number) => [number, number];
    readonly scriptordoc_cellStep: (a: number, b: number, c: number) => number;
    readonly scriptordoc_clearFormatting: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_deleteRange: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_deleteTableColumn: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_deleteTableRow: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_formatBold: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_formatColor: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_formatFont: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_formatHighlight: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_formatItalic: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_formatSize: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_formatStrike: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_formatUnderline: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_formatVertAlign: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly scriptordoc_insertTableColumn: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_insertTableRow: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_insertText: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_insertToc: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_joinParagraph: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_linkAt: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_markMoveSource: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scriptordoc_mergeCellsDown: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_mergeCellsRight: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_moveRange: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly scriptordoc_moveTableColumn: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_moveTableRow: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_paragraphFormat: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_paragraphStyle: (a: number, b: number) => [number, number];
    readonly scriptordoc_redo: (a: number) => [number, number, number];
    readonly scriptordoc_removeHyperlink: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_resolveStyleProps: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_selectionFormat: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scriptordoc_setAlignment: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_setCellShading: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scriptordoc_setIndentFirst: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_setIndentLeft: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_setIndentRight: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_setLineSpacing: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_setParagraphStyle: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_setRowHeight: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scriptordoc_setStyleProps: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly scriptordoc_setTableBorders: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly scriptordoc_splitCellHorizontal: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_splitCellVertical: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_splitParagraph: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_styleGallery: (a: number) => [number, number];
    readonly scriptordoc_tableContext: (a: number, b: number) => [number, number];
    readonly scriptordoc_undo: (a: number) => [number, number, number];
    readonly scriptordoc_updateToc: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_caretLineStep: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_caretRect: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_changeBars: (a: number) => [number, number];
    readonly scriptordoc_hitTest: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_paintPage: (a: number, b: number) => [number, number];
    readonly scriptordoc_paintPageBand: (a: number, b: number) => [number, number];
    readonly scriptordoc_paragraphCount: (a: number) => [number, number, number];
    readonly scriptordoc_paragraphLength: (a: number, b: number) => number;
    readonly scriptordoc_paragraphPage: (a: number, b: number) => number;
    readonly scriptordoc_paragraphRange: (a: number, b: number) => [number, number];
    readonly scriptordoc_paragraphText: (a: number, b: number) => [number, number, number, number];
    readonly scriptordoc_selectionRects: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly scriptordoc_wordCount: (a: number) => number;
    readonly scriptordoc_addComment: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly scriptordoc_commentRects: (a: number) => [number, number];
    readonly scriptordoc_commentsAt: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_deleteComment: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_listComments: (a: number) => [number, number];
    readonly scriptordoc_nextComment: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_pageGeometry: (a: number) => [number, number];
    readonly scriptordoc_prevComment: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_replyComment: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly scriptordoc_resolveComment: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_setLandscape: (a: number, b: number) => void;
    readonly scriptordoc_setMargins: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly scriptordoc_setPageSize: (a: number, b: number, c: number) => void;
    readonly scriptordoc_toDocumentXml: (a: number) => [number, number, number, number];
    readonly scriptordoc_toDocx: (a: number) => [number, number, number, number];
    readonly scriptordoc_applyList: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_applyListFormat: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_paragraphListFormat: (a: number, b: number) => [number, number];
    readonly scriptordoc_paragraphListKind: (a: number, b: number) => [number, number];
    readonly scriptordoc_paragraphListLevel: (a: number, b: number) => number;
    readonly scriptordoc_paragraphNumId: (a: number, b: number) => number;
    readonly scriptordoc_relayout: (a: number, b: number) => [number, number, number];
    readonly scriptordoc_setNumbering: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scriptordoc_acceptAll: (a: number) => [number, number, number];
    readonly scriptordoc_acceptChange: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_acceptRevision: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_balloonsOn: (a: number) => number;
    readonly scriptordoc_clearExpandedParagraphs: (a: number) => void;
    readonly scriptordoc_ensureFooter: (a: number) => number;
    readonly scriptordoc_ensureHeader: (a: number) => number;
    readonly scriptordoc_footerText: (a: number) => [number, number];
    readonly scriptordoc_headerText: (a: number) => [number, number];
    readonly scriptordoc_isParagraphExpanded: (a: number, b: number) => number;
    readonly scriptordoc_listChanges: (a: number) => [number, number];
    readonly scriptordoc_nextChange: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_prevChange: (a: number, b: number, c: number) => [number, number];
    readonly scriptordoc_rejectAll: (a: number) => [number, number, number];
    readonly scriptordoc_rejectChange: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_rejectRevision: (a: number, b: number, c: number) => [number, number, number];
    readonly scriptordoc_reviewers: (a: number) => [number, number];
    readonly scriptordoc_setActiveStory: (a: number, b: number) => void;
    readonly scriptordoc_setAuthor: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly scriptordoc_setBalloons: (a: number, b: number) => void;
    readonly scriptordoc_setFooterText: (a: number, b: number, c: number) => void;
    readonly scriptordoc_setHeaderFooterPage: (a: number, b: number) => void;
    readonly scriptordoc_setHeaderText: (a: number, b: number, c: number) => void;
    readonly scriptordoc_setNow: (a: number, b: number, c: number) => void;
    readonly scriptordoc_setReviewerHidden: (a: number, b: number, c: number, d: number) => void;
    readonly scriptordoc_setTrackChanges: (a: number, b: number) => void;
    readonly scriptordoc_setTrackDisplay: (a: number, b: number, c: number) => void;
    readonly scriptordoc_setTrackLocked: (a: number, b: number) => void;
    readonly scriptordoc_toggleParagraphExpanded: (a: number, b: number) => number;
    readonly scriptordoc_trackAt: (a: number, b: number, c: number) => number;
    readonly scriptordoc_trackChangesOn: (a: number) => number;
    readonly scriptordoc_trackLocked: (a: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
