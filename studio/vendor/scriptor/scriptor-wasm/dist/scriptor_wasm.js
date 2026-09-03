/* @ts-self-types="./scriptor_wasm.d.ts" */

/**
 * The result of a [`ScriptorDoc::relayout`]: the page dimensions (device px) so the browser can
 * size + lay out the page stack, plus the per-page fingerprints it diffs to decide which pages to
 * re-rasterize via [`ScriptorDoc::paint_page`].
 */
export class LayoutInfo {
    static __wrap(ptr) {
        const obj = Object.create(LayoutInfo.prototype);
        obj.__wbg_ptr = ptr;
        LayoutInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LayoutInfoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_layoutinfo_free(ptr, 0);
    }
    /**
     * The per-page content fingerprints (one per page). The caller compares against the previous
     * set and re-rasterizes only the pages whose value changed.
     * @returns {BigUint64Array}
     */
    get fingerprints() {
        const ret = wasm.layoutinfo_fingerprints(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get gap() {
        const ret = wasm.layoutinfo_gap(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get pageCount() {
        const ret = wasm.layoutinfo_pageCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get pageHeight() {
        const ret = wasm.layoutinfo_pageHeight(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get pageWidth() {
        const ret = wasm.layoutinfo_pageWidth(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get totalHeight() {
        const ret = wasm.layoutinfo_totalHeight(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) LayoutInfo.prototype[Symbol.dispose] = LayoutInfo.prototype.free;

/**
 * A paragraph's formatting, for the toolbar's Paragraph group. `align` is "" when unset;
 * `lineSpacing` (240ths) + indents are 0 when unset.
 */
export class ParaFmt {
    static __wrap(ptr) {
        const obj = Object.create(ParaFmt.prototype);
        obj.__wbg_ptr = ptr;
        ParaFmtFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ParaFmtFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_parafmt_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get align() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.parafmt_align(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get indentFirst() {
        const ret = wasm.parafmt_indentFirst(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get indentLeft() {
        const ret = wasm.parafmt_indentLeft(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get indentRight() {
        const ret = wasm.parafmt_indentRight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get lineSpacing() {
        const ret = wasm.parafmt_lineSpacing(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) ParaFmt.prototype[Symbol.dispose] = ParaFmt.prototype.free;

/**
 * A live document held across the FFI boundary. Owns the CRDT replica and the canvas renderer;
 * the TS shell holds an opaque handle to it.
 */
export class ScriptorDoc {
    static __wrap(ptr) {
        const obj = Object.create(ScriptorDoc.prototype);
        obj.__wbg_ptr = ptr;
        ScriptorDocFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ScriptorDocFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_scriptordoc_free(ptr, 0);
    }
    /**
     * Accept every tracked change in the document - body, header, and footer. Returns the total
     * count resolved.
     * @returns {number}
     */
    acceptAll() {
        const ret = wasm.scriptordoc_acceptAll(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Accept the tracked change under the caret `(para, off)` (insertion -> keep text, deletion ->
     * remove text). Returns whether one was resolved. Re-layout + re-paint after.
     * @param {number} para
     * @param {number} off
     * @returns {boolean}
     */
    acceptChange(para, off) {
        const ret = wasm.scriptordoc_acceptChange(this.__wbg_ptr, para, off);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Accept a specific revision id in the region of `para` (the inline click popup carries both the
     * click's paragraph and the id from [`track_at`]; revision ids are per-story, so the region picks
     * the right child document).
     * @param {number} para
     * @param {number} id
     * @returns {boolean}
     */
    acceptRevision(para, id) {
        const ret = wasm.scriptordoc_acceptRevision(this.__wbg_ptr, para, id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Add a named bookmark over codepoint `[start, end)` in paragraph `para`. The name should already be
     * a valid Word bookmark name (letters/digits/underscore, letter-initial); the caller sanitizes.
     * Re-paint after (bookmarks are invisible but become hyperlink targets).
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} name
     */
    addBookmark(para, start, end, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_addBookmark(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Add a comment over the selection `(start_para,start_off)..(end_para,end_off)` (one story) with
     * `text` as the body, attributed to the current author + last timestamp. Returns the new comment
     * id, or `-1` if the endpoints are in different stories / the story doesn't exist.
     * @param {number} start_para
     * @param {number} start_off
     * @param {number} end_para
     * @param {number} end_off
     * @param {string} text
     * @returns {number}
     */
    addComment(start_para, start_off, end_para, end_off, text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_addComment(this.__wbg_ptr, start_para, start_off, end_para, end_off, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Add a hyperlink over codepoint `[start, end)` in paragraph `para`, targeting `target` (an
     * external URL, or `#bookmarkName` for an internal jump). Re-layout + re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} target
     */
    addHyperlink(para, start, end, target) {
        const ptr0 = passStringToWasm0(target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_addHyperlink(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Insert `text` at codepoint `off` in paragraph `para` as the **destination** of move `id`
     * (`w:moveTo`), pairing with a prior [`mark_move_source`](Self::mark_move_source). Re-paint after.
     * @param {number} para
     * @param {number} off
     * @param {string} text
     * @param {number} id
     */
    addMoveDest(para, off, text, id) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_addMoveDest(this.__wbg_ptr, para, off, ptr0, len0, id);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Create a new paragraph style (Word's New-Style / Save-Selection-as-a-Style) named `name`, based
     * on `based_on` (empty = no parent), with the given formatting (same per-field sentinels as
     * `setStyleProps`). Mints a unique style id from `name`, registers it (gallery + persistence), and
     * returns the id so the caller can apply it to the selected paragraph(s). Body story only.
     * @param {string} name
     * @param {string} based_on
     * @param {number} size
     * @param {number} bold
     * @param {number} italic
     * @param {string} color
     * @param {string} font
     * @param {number} line_spacing
     * @param {number} space_before
     * @param {number} space_after
     * @param {string} align
     * @param {string} line_rule
     * @returns {string}
     */
    addStyle(name, based_on, size, bold, italic, color, font, line_spacing, space_before, space_after, align, line_rule) {
        let deferred8_0;
        let deferred8_1;
        try {
            const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(based_on, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len2 = WASM_VECTOR_LEN;
            const ptr3 = passStringToWasm0(font, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len3 = WASM_VECTOR_LEN;
            const ptr4 = passStringToWasm0(align, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len4 = WASM_VECTOR_LEN;
            const ptr5 = passStringToWasm0(line_rule, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len5 = WASM_VECTOR_LEN;
            const ret = wasm.scriptordoc_addStyle(this.__wbg_ptr, ptr0, len0, ptr1, len1, size, bold, italic, ptr2, len2, ptr3, len3, line_spacing, space_before, space_after, ptr4, len4, ptr5, len5);
            var ptr7 = ret[0];
            var len7 = ret[1];
            if (ret[3]) {
                ptr7 = 0; len7 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred8_0 = ptr7;
            deferred8_1 = len7;
            return getStringFromWasm0(ptr7, len7);
        } finally {
            wasm.__wbindgen_free(deferred8_0, deferred8_1, 1);
        }
    }
    /**
     * Encode an edit-stable anchor for a SELECTED RANGE `[(p1,o1), (p2,o2))`
     * (body codepoint offsets). Send it with an inline select->ask so the agent
     * edits exactly that span via the anchored `document_propose_edit`. The head
     * biases left, the tail right, so the range doesn't grow/shrink spuriously
     * when text is inserted at either edge.
     * @param {number} p1
     * @param {number} o1
     * @param {number} p2
     * @param {number} o2
     * @returns {Uint8Array}
     */
    anchorRange(p1, o1, p2, o2) {
        const ret = wasm.scriptordoc_anchorRange(this.__wbg_ptr, p1, o1, p2, o2);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Apply a bullet (`bullet = true`) / decimal numbered list to paragraph `para` (body only). The
     * Bullets / Numbering buttons. Re-layout + re-paint after.
     * @param {number} para
     * @param {boolean} bullet
     */
    applyList(para, bullet) {
        const ret = wasm.scriptordoc_applyList(this.__wbg_ptr, para, bullet);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Apply a numbered list with a specific number format to paragraph `para` (body only): `numFmt` is
     * an OOXML token (`decimal` / `lowerLetter` / `upperLetter` / `lowerRoman` / `upperRoman`). The
     * Numbering button's format picker. Tracked as a `w:pPrChange` when Track-Changes is on.
     * @param {number} para
     * @param {string} num_fmt
     */
    applyListFormat(para, num_fmt) {
        const ptr0 = passStringToWasm0(num_fmt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_applyListFormat(this.__wbg_ptr, para, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether revision balloons are on.
     * @returns {boolean}
     */
    balloonsOn() {
        const ret = wasm.scriptordoc_balloonsOn(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The body paragraph index where bookmark `name` begins, or `-1` - lets an internal hyperlink
     * (`#name`) jump to its target. Body only.
     * @param {string} name
     * @returns {number}
     */
    bookmarkParagraph(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_bookmarkParagraph(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    canRedo() {
        const ret = wasm.scriptordoc_canRedo(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether there is anything to undo / redo in the active story (for greying the toolbar buttons).
     * @returns {boolean}
     */
    canUndo() {
        const ret = wasm.scriptordoc_canUndo(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Encode an edit-stable anchor at a body caret position (codepoint offset).
     * Send it as your cursor for presence, or capture it before a merge and
     * resolve it after to remap the local caret.
     * @param {number} para
     * @param {number} off
     * @returns {Uint8Array}
     */
    caretAnchor(para, off) {
        const ret = wasm.scriptordoc_caretAnchor(this.__wbg_ptr, para, off);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * The caret one visual line above/below `(para, off)`, keeping goal column `x` (device px) -
     * Word's ArrowUp/Down. Returns `[para, off]` (codepoints), or empty at the document edge. (The
     * shell's old one-pixel hit-test probe snapped back inside inter-paragraph spacing, so Up/Down
     * never crossed a paragraph boundary.)
     * @param {number} para
     * @param {number} off
     * @param {number} x
     * @param {boolean} down
     * @returns {Uint32Array}
     */
    caretLineStep(para, off, x, down) {
        const ret = wasm.scriptordoc_caretLineStep(this.__wbg_ptr, para, off, x, down);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The caret rectangle for `(para, codepoint_offset)` as `[x, y, height]` (device px).
     * @param {number} para
     * @param {number} off
     * @returns {Float32Array}
     */
    caretRect(para, off) {
        const ret = wasm.scriptordoc_caretRect(this.__wbg_ptr, para, off);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The caret cell's shading fill (RGB hex, no `#`), or `""` if not in a cell / no shading - lets the
     * UI pre-select the current colour.
     * @param {number} para
     * @returns {string}
     */
    cellShading(para) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_cellShading(this.__wbg_ptr, para);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The body paragraph index of the first paragraph of the cell one step forward (`forward=true`) /
     * backward from the caret's cell, or `-1` when `para` isn't in a cell / is at the table's last /
     * first cell. Drives Tab / Shift+Tab cell navigation. Body only (header/footer have no tables in v1).
     * @param {number} para
     * @param {boolean} forward
     * @returns {number}
     */
    cellStep(para, forward) {
        const ret = wasm.scriptordoc_cellStep(this.__wbg_ptr, para, forward);
        return ret;
    }
    /**
     * The margin change-bar rectangles from the last layout, flattened as `[page, x, y, w, h, para]`
     * per bar (device px; `y` is page-local within `page`). The editor hit-tests a click in the left
     * margin against these to drive Simple-Markup click-to-expand; `para` is the namespaced paragraph
     * the bar belongs to. Body + table-cell bars only (header/footer bars paint inline, not here).
     * @returns {Float32Array}
     */
    changeBars() {
        const ret = wasm.scriptordoc_changeBars(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Clear every click-to-expand paragraph (on a display-mode change or a new document).
     */
    clearExpandedParagraphs() {
        wasm.scriptordoc_clearExpandedParagraphs(this.__wbg_ptr);
    }
    /**
     * Clear all inline run formatting over `[start, end)` (the Clear Formatting eraser). Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     */
    clearFormatting(para, start, end) {
        const ret = wasm.scriptordoc_clearFormatting(this.__wbg_ptr, para, start, end);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Highlight rectangles (device px, flattened `[x,y,w,h,...]`) behind every text range anchored by
     * at least one *unresolved* comment, across body + header/footer. Editor chrome (not exported) -
     * the view paints these on its overlay like the selection.
     * @returns {Float32Array}
     */
    commentRects() {
        const ret = wasm.scriptordoc_commentRects(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The comment ids anchored at `(para, off)` (the run under the caret, or one ending at it).
     * Region-routed; empty when the caret isn't inside a comment anchor.
     * @param {number} para
     * @param {number} off
     * @returns {Uint32Array}
     */
    commentsAt(para, off) {
        const ret = wasm.scriptordoc_commentsAt(this.__wbg_ptr, para, off);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Delete comment `id` and its replies (clearing the anchor in whichever story holds it).
     * @param {number} id
     * @returns {boolean}
     */
    deleteComment(id) {
        const ret = wasm.scriptordoc_deleteComment(this.__wbg_ptr, id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Delete codepoint range `[start, end)` in paragraph `para` as a direct human edit. No-op when
     * the range is empty. Call [`paint`] after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     */
    deleteRange(para, start, end) {
        const ret = wasm.scriptordoc_deleteRange(this.__wbg_ptr, para, start, end);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Delete the caret's table column. With Track-Changes on it's *marked* (`w:tcPr/w:cellDel` on the
     * column's cells, retained until accepted); otherwise removed (and the table if it empties).
     * Returns the caret paragraph after the edit, or `-1` if not in a table. Re-layout + re-paint.
     * @param {number} para
     * @returns {number}
     */
    deleteTableColumn(para) {
        const ret = wasm.scriptordoc_deleteTableColumn(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Delete the caret's table row. With Track-Changes on it's *marked* (`w:trPr/w:del`, the row
     * survives until accepted); otherwise it's removed (and the table if it was the last row). Returns
     * the caret paragraph after the edit, or `-1` if `para` isn't in a table. Re-layout + re-paint.
     * @param {number} para
     * @returns {number}
     */
    deleteTableRow(para) {
        const ret = wasm.scriptordoc_deleteTableRow(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Ensure a default footer story exists (see [`ensure_header`](Self::ensure_header)) and return
     * the namespaced index of its first paragraph.
     * @returns {number}
     */
    ensureFooter() {
        const ret = wasm.scriptordoc_ensureFooter(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Ensure a default header story exists - creating an empty one (a single blank paragraph) if the
     * document has none - and return the namespaced paragraph index of its first paragraph, so the
     * shell can drop the caret into the header to edit it (Word's Insert > Header). Idempotent: an
     * existing header is left untouched (its content preserved). Re-layout + paint after.
     * @returns {number}
     */
    ensureHeader() {
        const ret = wasm.scriptordoc_ensureHeader(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Export the ops committed since `version` (from `oplogVersion`) as a loro
     * update delta to send to peers.
     * @param {Uint8Array} version
     * @returns {Uint8Array}
     */
    exportUpdatesSince(version) {
        const ptr0 = passArray8ToWasm0(version, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_exportUpdatesSince(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v2;
    }
    /**
     * The default footer as plain text.
     * @returns {string}
     */
    footerText() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_footerText(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Toggle/set bold over codepoint `[start, end)` in paragraph `para`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {boolean} on
     */
    formatBold(para, start, end, on) {
        const ret = wasm.scriptordoc_formatBold(this.__wbg_ptr, para, start, end, on);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set text color (RRGGBB hex) over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} hex
     */
    formatColor(para, start, end, hex) {
        const ptr0 = passStringToWasm0(hex, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_formatColor(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set font family over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} family
     */
    formatFont(para, start, end, family) {
        const ptr0 = passStringToWasm0(family, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_formatFont(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set / clear the highlight color over `[start, end)` (`""` clears it). Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} color
     */
    formatHighlight(para, start, end, color) {
        const ptr0 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_formatHighlight(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Toggle/set italic over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {boolean} on
     */
    formatItalic(para, start, end, on) {
        const ret = wasm.scriptordoc_formatItalic(this.__wbg_ptr, para, start, end, on);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set font size (half-points, OOXML `w:sz`) over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {number} half_points
     */
    formatSize(para, start, end, half_points) {
        const ret = wasm.scriptordoc_formatSize(this.__wbg_ptr, para, start, end, half_points);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Toggle/set strikethrough over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {boolean} on
     */
    formatStrike(para, start, end, on) {
        const ret = wasm.scriptordoc_formatStrike(this.__wbg_ptr, para, start, end, on);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Toggle/set underline over `[start, end)`. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {boolean} on
     */
    formatUnderline(para, start, end, on) {
        const ret = wasm.scriptordoc_formatUnderline(this.__wbg_ptr, para, start, end, on);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set / clear vertical alignment over `[start, end)` ("superscript" / "subscript", or `""` to
     * clear back to baseline). Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @param {string} value
     */
    formatVertAlign(para, start, end, value) {
        const ptr0 = passStringToWasm0(value, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_formatVertAlign(this.__wbg_ptr, para, start, end, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Build a document from a loro snapshot - the collaboration join message
     * from the server. Unlike the `constructor`, it seeds NO empty paragraph:
     * the snapshot is the authoritative content, and a seed would union a stray
     * blank paragraph into the merged document.
     * @param {Uint8Array} snapshot
     * @returns {ScriptorDoc}
     */
    static fromSnapshot(snapshot) {
        const ptr0 = passArray8ToWasm0(snapshot, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_fromSnapshot(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ScriptorDoc.__wrap(ret[0]);
    }
    /**
     * The default header as plain text (one line per paragraph).
     * @returns {string}
     */
    headerText() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_headerText(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Map a canvas point (device px) to a caret position, returned as `[paragraph, codepoint]`.
     * Uses the geometry from the most recent [`paint`].
     * @param {number} x
     * @param {number} y
     * @returns {Uint32Array}
     */
    hitTest(x, y) {
        const ret = wasm.scriptordoc_hitTest(this.__wbg_ptr, x, y);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The editable picture id whose hit-rect contains the canvas point `(x, y)` (absolute px), or
     * `None`. The topmost match wins (floats over inline) - the view uses this for click-to-select.
     * @param {number} x
     * @param {number} y
     * @returns {bigint | undefined}
     */
    imageAtPoint(x, y) {
        const ret = wasm.scriptordoc_imageAtPoint(this.__wbg_ptr, x, y);
        return ret[0] === 0 ? undefined : BigInt.asUintN(64, ret[1]);
    }
    /**
     * Picture `id`'s crop as `[l, t, r, b]` (`<a:srcRect>`, thousandths of a percent - the share of
     * each edge cut), or `None`. The view seeds the crop window from this.
     * @param {bigint} id
     * @returns {Int32Array | undefined}
     */
    imageCrop(id) {
        const ret = wasm.scriptordoc_imageCrop(this.__wbg_ptr, id);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * The raw (encoded, uncropped) media bytes of picture `id`, or `None`. The view decodes these to
     * show the full image behind the crop window in crop mode (the page canvas only has the cropped
     * bitmap).
     * @param {bigint} id
     * @returns {Uint8Array | undefined}
     */
    imageMedia(id) {
        const ret = wasm.scriptordoc_imageMedia(this.__wbg_ptr, id);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Picture `id`'s rect on the canvas as `[x, y, w, h]` (absolute px), or `None` if it isn't placed
     * (e.g. off-page). The view draws the selection box + resize handles from this.
     * @param {bigint} id
     * @returns {Float32Array | undefined}
     */
    imageRect(id) {
        const ret = wasm.scriptordoc_imageRect(this.__wbg_ptr, id);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * Picture `id`'s wrap state as a single token for the Wrap Text menu + drag logic: `inline`
     * (in the flow), `square` / `tight` / `through` / `topAndBottom` (floating, text wraps), `behind`
     * (floating, behind the text), or `front` (floating, in front of the text). `None` if `id` is
     * unknown.
     * @param {bigint} id
     * @returns {string | undefined}
     */
    imageWrapState(id) {
        const ret = wasm.scriptordoc_imageWrapState(this.__wbg_ptr, id);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Insert a picture at codepoint `off` in **body** paragraph `para`: `bytes` (MIME `mime`, e.g.
     * `image/png`) ship as a fresh `word/media` part on save, shown at `w_emu` x `h_emu` (EMU; the
     * caller derives these from the decoded natural size via [`px_to_emu`]). Returns the new picture
     * id. Re-layout + repaint after. Images live on the body story only (header/footer not supported).
     * @param {number} para
     * @param {number} off
     * @param {Uint8Array} bytes
     * @param {string} mime
     * @param {number} w_emu
     * @param {number} h_emu
     * @returns {bigint}
     */
    insertImage(para, off, bytes, mime, w_emu, h_emu) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(mime, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_insertImage(this.__wbg_ptr, para, off, ptr0, len0, ptr1, len1, w_emu, h_emu);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * Insert a table column left (`right=false`) / right (`right=true`) of the caret's cell. With
     * Track-Changes on it's a tracked insertion (`w:tcPr/w:cellIns` on each new cell); otherwise
     * direct. Returns the new caret paragraph, or `-1` if not in a table. Re-layout + re-paint after.
     * @param {number} para
     * @param {boolean} right
     * @returns {number}
     */
    insertTableColumn(para, right) {
        const ret = wasm.scriptordoc_insertTableColumn(this.__wbg_ptr, para, right);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Insert a table row above (`below=false`) / below (`below=true`) the caret's row. With
     * Track-Changes on it's recorded as a tracked insertion (`w:trPr/w:ins`); otherwise direct.
     * Returns the new caret paragraph, or `-1` if `para` isn't in a table. Re-layout + re-paint after.
     * @param {number} para
     * @param {boolean} below
     * @returns {number}
     */
    insertTableRow(para, below) {
        const ret = wasm.scriptordoc_insertTableRow(this.__wbg_ptr, para, below);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Insert `text` at codepoint `off` in paragraph `para` as a direct human edit, routed through
     * the shared `scriptor_edit::apply` path (the same one the agent uses). Call [`paint`] after.
     * @param {number} para
     * @param {number} off
     * @param {string} text
     */
    insertText(para, off, text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_insertText(this.__wbg_ptr, para, off, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Insert a table of contents at body paragraph `at` (the caret), built from the document's current
     * headings (`Heading1`..`Heading9`): one `TOC{level}` line per heading - "{heading}\t{page}" -
     * wrapped as a real `TOC` field so Word can update it (F9). Page numbers come from a fresh layout
     * that already includes the inserted lines. Returns whether a TOC was inserted (`false` when there
     * are no headings, or `at` isn't a body paragraph). Re-layout + re-paint after.
     * @param {number} at
     * @returns {boolean}
     */
    insertToc(at) {
        const ret = wasm.scriptordoc_insertToc(this.__wbg_ptr, at);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Whether paragraph `para` is currently expanded (click-to-expand).
     * @param {number} para
     * @returns {boolean}
     */
    isParagraphExpanded(para) {
        const ret = wasm.scriptordoc_isParagraphExpanded(this.__wbg_ptr, para);
        return ret !== 0;
    }
    /**
     * Join paragraph `para` into the previous one (Backspace at paragraph start / Delete at end).
     * Returns the codepoint offset in the previous paragraph where the two met (the merged caret),
     * or `-1` when the join is refused because it would cross a table-cell boundary (the caller
     * should leave the caret where it is). Routed through the shared `scriptor_edit::apply` path.
     * @param {number} para
     * @returns {number}
     */
    joinParagraph(para) {
        const ret = wasm.scriptordoc_joinParagraph(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * The hyperlink target at codepoint `off` in paragraph `para` (external URL or `#bookmarkName`),
     * or `""` when the caret isn't on a link - lets the toolbar reflect / the caret follow it.
     * @param {number} para
     * @param {number} off
     * @returns {string}
     */
    linkAt(para, off) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_linkAt(this.__wbg_ptr, para, off);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Every tracked change across stories as a JSON array (for the reviewing pane): each object has
     * `id`, `kind` (`"ins"` / `"del"` / `"fmt"` / `"movefrom"` / `"moveto"` for run changes;
     * `"rowins"` / `"rowdel"` / `"colins"` / `"coldel"` for table-structure changes), `author`,
     * `date`, `text`, and the caret `para` (namespaced) / `off`. Run changes come from each story's
     * `change_carets` + `track_at`; table changes from `table_changes()`. Sorted in document order.
     * (Comments come from [`list_comments`](Self::list_comments) - the pane merges the two.)
     * @returns {string}
     */
    listChanges() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_listChanges(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Every comment as a JSON array string (for the popover / reviewing list): each object has
     * `id`, `author`, `initials`, `date`, `text`, `parent` (id or null), `resolved`, and the anchor
     * caret `para` / `off` (namespaced; `-1` when un-anchored). Replies inherit the parent's anchor.
     * @returns {string}
     */
    listComments() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_listComments(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Mark codepoint range `[start, end)` in paragraph `para` as the **source** of a move
     * (`w:moveFrom`, text retained), returning the move's revision id (or `-1` for an empty range /
     * missing story). The matching destination is added with [`add_move_dest`](Self::add_move_dest)
     * using this id - the two-step path the editor's cut-then-paste move uses. Re-paint after.
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @returns {number}
     */
    markMoveSource(para, start, end) {
        const ret = wasm.scriptordoc_markMoveSource(this.__wbg_ptr, para, start, end);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Merge a remote loro blob (a snapshot or an update delta) into this replica.
     * Loro merges are commutative + idempotent, so order does not matter and a
     * re-merge is a no-op. The caller re-renders afterward (`relayout` re-reads
     * the model); pair with `caretAnchor` + `resolveAnchor` to keep the local
     * caret on the same character when a concurrent edit shifts offsets.
     * @param {Uint8Array} bytes
     */
    merge(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_merge(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Merge the caret's cell with the `count - 1` cells below it (vertical `w:vMerge` merge). Returns
     * the caret after the merge, or `-1` if not in a table / not enough rows. Re-layout after.
     * @param {number} para
     * @param {number} count
     * @returns {number}
     */
    mergeCellsDown(para, count) {
        const ret = wasm.scriptordoc_mergeCellsDown(this.__wbg_ptr, para, count);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Merge the caret's cell with the `count - 1` cells to its right (horizontal `w:gridSpan` merge).
     * Returns the caret after the merge, or `-1` if not in a table / not enough cells. Re-layout after.
     * @param {number} para
     * @param {number} count
     * @returns {number}
     */
    mergeCellsRight(para, count) {
        const ret = wasm.scriptordoc_mergeCellsRight(this.__wbg_ptr, para, count);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Move codepoint range `[from_start, from_end)` in paragraph `from_para` to codepoint `to_off` in
     * paragraph `to_para` as a tracked move (`w:moveFrom` source + `w:moveTo` destination, one shared
     * revision id). Both endpoints must be in the same story (body / header / footer); the destination
     * must lie outside the source range. Returns the move's revision id, or `-1` when the endpoints
     * span stories / the range is empty / the move is into itself. Re-paint after.
     * @param {number} from_para
     * @param {number} from_start
     * @param {number} from_end
     * @param {number} to_para
     * @param {number} to_off
     * @returns {number}
     */
    moveRange(from_para, from_start, from_end, to_para, to_off) {
        const ret = wasm.scriptordoc_moveRange(this.__wbg_ptr, from_para, from_start, from_end, to_para, to_off);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Move the caret's table column left (`left=true`) / right one position. Returns the caret
     * paragraph after the move, or `-1` if not in a table / the move runs off the edge.
     * @param {number} para
     * @param {boolean} left
     * @returns {number}
     */
    moveTableColumn(para, left) {
        const ret = wasm.scriptordoc_moveTableColumn(this.__wbg_ptr, para, left);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Move the caret's table row up (`up=true`) / down one position (a direct structural reorder).
     * Returns the caret paragraph after the move, or `-1` if not in a table / the move runs off the
     * edge. Re-layout + re-paint after.
     * @param {number} para
     * @param {boolean} up
     * @returns {number}
     */
    moveTableRow(para, up) {
        const ret = wasm.scriptordoc_moveTableRow(this.__wbg_ptr, para, up);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Start an empty document (no file) - used to author a fresh doc in the editor. Seeds one
     * empty paragraph so the caret has a block to type into (an editor never has zero paragraphs).
     */
    constructor() {
        const ret = wasm.scriptordoc_new();
        this.__wbg_ptr = ret;
        ScriptorDocFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The caret `[para, off]` of the next tracked change after `(para, off)`, searched **across all
     * stories** (body + header + footer) and wrapping, or an empty array when the document has no
     * tracked changes. For Review > Next.
     * @param {number} para
     * @param {number} off
     * @returns {Uint32Array}
     */
    nextChange(para, off) {
        const ret = wasm.scriptordoc_nextChange(this.__wbg_ptr, para, off);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Caret `[para, off]` of the next (`forward`) / previous comment anchor across stories (wraps),
     * or an empty array when the document has no comments. For Review > Next/Previous comment.
     * @param {number} para
     * @param {number} off
     * @returns {Uint32Array}
     */
    nextComment(para, off) {
        const ret = wasm.scriptordoc_nextComment(this.__wbg_ptr, para, off);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Open a `.docx` (the raw OPC zip bytes, e.g. from a `File`) into the CRDT model.
     * @param {Uint8Array} bytes
     * @returns {ScriptorDoc}
     */
    static openDocx(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_openDocx(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ScriptorDoc.__wrap(ret[0]);
    }
    /**
     * The current oplog version, encoded. Hold it, then `exportUpdatesSince` to
     * ship only the ops committed since - the efficient wire delta.
     * @returns {Uint8Array}
     */
    oplogVersion() {
        const ret = wasm.scriptordoc_oplogVersion(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Page geometry in twips: `[width, height, marginTop, marginRight, marginBottom, marginLeft,
     * headerDist, footerDist]`. For the ruler + the Layout tab's page-size / margin controls.
     * @returns {Uint32Array}
     */
    pageGeometry() {
        const ret = wasm.scriptordoc_pageGeometry(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Rasterize a single page (0-based) of the current layout: an opaque white sheet with that
     * page's text. Returns RGBA8 (`page_width*page_height*4`); the browser blits it at
     * `y = index*(page_height+gap)`. Call after [`relayout`], only for pages whose fingerprint
     * changed. Also refreshes the page's raster cache, so a later [`Self::paint_page_band`] diffs
     * against exactly what the canvas shows.
     * @param {number} index
     * @returns {Uint8Array}
     */
    paintPage(index) {
        const ret = wasm.scriptordoc_paintPage(this.__wbg_ptr, index);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * [`Self::paint_page`], returning only the vertical band of rows that actually CHANGED since
     * the page's last raster: an 8-byte little-endian header `[y0, y1)` followed by `(y1-y0)` rows
     * of RGBA. Typing edits one paragraph, so shipping the whole ~3-4MB page across the wasm->JS
     * boundary per keystroke was mostly wasted transfer + GC; the band is pixel-diffed against the
     * cached previous raster, so it can never miss a visual change (no raster cached, or a size
     * change, degrades to the full page; nothing changed returns an empty `[0, 0)` band). Only
     * valid when the caller's canvas still shows this page's previous raster.
     * @param {number} index
     * @returns {Uint8Array}
     */
    paintPageBand(index) {
        const ret = wasm.scriptordoc_paintPageBand(this.__wbg_ptr, index);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Number of paragraphs in the **body** (back-compat for callers that don't track regions). For
     * region-aware caret bounds use [`paragraph_range`]. Served from the cached texts (O(1)) with a
     * fallback to materializing the tree before the first render.
     * @returns {number}
     */
    paragraphCount() {
        const ret = wasm.scriptordoc_paragraphCount(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * The paragraph-level formatting of paragraph `para` (for the Paragraph-group toolbar state).
     * @param {number} para
     * @returns {ParaFmt}
     */
    paragraphFormat(para) {
        const ret = wasm.scriptordoc_paragraphFormat(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ParaFmt.__wrap(ret[0]);
    }
    /**
     * The codepoint length of paragraph `para` (for caret clamping + cross-paragraph movement).
     * @param {number} para
     * @returns {number}
     */
    paragraphLength(para) {
        const ret = wasm.scriptordoc_paragraphLength(this.__wbg_ptr, para);
        return ret >>> 0;
    }
    /**
     * Paragraph `para`'s list level-0 number format (`"decimal"` / `"lowerRoman"` / `"bullet"` / ...),
     * or `""` when it isn't in a list - lets the Numbering format picker check the active format.
     * @param {number} para
     * @returns {string}
     */
    paragraphListFormat(para) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_paragraphListFormat(this.__wbg_ptr, para);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The kind of list paragraph `para` is in: `"bullet"`, `"number"`, or `""` (not a list) - lets the
     * toolbar toggle the Bullets / Numbering buttons like Word.
     * @param {number} para
     * @returns {string}
     */
    paragraphListKind(para) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_paragraphListKind(this.__wbg_ptr, para);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Paragraph `para`'s list level (`w:numPr/w:ilvl`, 0-8), or `-1` when it isn't in a list - lets
     * Tab / Shift+Tab demote / promote a list item to the next / previous level.
     * @param {number} para
     * @returns {number}
     */
    paragraphListLevel(para) {
        const ret = wasm.scriptordoc_paragraphListLevel(this.__wbg_ptr, para);
        return ret;
    }
    /**
     * Paragraph `para`'s current list id (`w:numPr/w:numId`), or `-1` when it isn't in a list - lets
     * the toolbar reflect / toggle the numbering state.
     * @param {number} para
     * @returns {number}
     */
    paragraphNumId(para) {
        const ret = wasm.scriptordoc_paragraphNumId(this.__wbg_ptr, para);
        return ret;
    }
    /**
     * The 0-based page index a paragraph sits on (from the last layout) - for "Page X of N". Body
     * paragraphs come from `placements`; table-cell paragraphs (which have no placement, only caret
     * geometry) are found by scanning the placed cells.
     * @param {number} para
     * @returns {number}
     */
    paragraphPage(para) {
        const ret = wasm.scriptordoc_paragraphPage(this.__wbg_ptr, para);
        return ret >>> 0;
    }
    /**
     * The `[firstIndex, count]` of the story (body / header / footer) that `para` belongs to - so the
     * JS shell can clamp caret movement to one story (a header caret can't arrow into the body). The
     * first index is the region's namespace base; `count` is its paragraph count.
     * @param {number} para
     * @returns {Uint32Array}
     */
    paragraphRange(para) {
        const ret = wasm.scriptordoc_paragraphRange(this.__wbg_ptr, para);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Paragraph `para`'s current named style id (`w:pStyle`), or `""` for the default (Normal) - lets
     * the Styles dropdown reflect the caret's paragraph.
     * @param {number} para
     * @returns {string}
     */
    paragraphStyle(para) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_paragraphStyle(this.__wbg_ptr, para);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The concatenated plain text of paragraph `index` (namespaced). Served from the cached texts
     * when available (avoids re-materializing the tree per call).
     * @param {number} index
     * @returns {string}
     */
    paragraphText(index) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.scriptordoc_paragraphText(this.__wbg_ptr, index);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * The caret `[para, off]` of the previous tracked change before `(para, off)`, across all
     * stories (wraps).
     * @param {number} para
     * @param {number} off
     * @returns {Uint32Array}
     */
    prevChange(para, off) {
        const ret = wasm.scriptordoc_prevChange(this.__wbg_ptr, para, off);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {number} para
     * @param {number} off
     * @returns {Uint32Array}
     */
    prevComment(para, off) {
        const ret = wasm.scriptordoc_prevComment(this.__wbg_ptr, para, off);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Redo the last undone edit (Ctrl+Y / Ctrl+Shift+Z) in the active story. Returns whether anything
     * changed.
     * @returns {boolean}
     */
    redo() {
        const ret = wasm.scriptordoc_redo(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Reject every tracked change in the document - body, header, and footer. Returns the total count.
     * @returns {number}
     */
    rejectAll() {
        const ret = wasm.scriptordoc_rejectAll(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Reject the tracked change under the caret (insertion -> remove text, deletion -> keep text).
     * @param {number} para
     * @param {number} off
     * @returns {boolean}
     */
    rejectChange(para, off) {
        const ret = wasm.scriptordoc_rejectChange(this.__wbg_ptr, para, off);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Reject a specific revision id in the region of `para`.
     * @param {number} para
     * @param {number} id
     * @returns {boolean}
     */
    rejectRevision(para, id) {
        const ret = wasm.scriptordoc_rejectRevision(this.__wbg_ptr, para, id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Render the whole document to a [`PaintResult`] (RGBA8 + dimensions) at the document's real
     * Re-resolve + lay out the whole document at the document's real page geometry (`w:sectPr` size
     * + margins), WITHOUT rasterizing - the cheap pass run on every edit. `scale` is the device-
     * pixel ratio. Each run's size / bold / italic / color is resolved from inline run formatting
     * over the paragraph's `styles.xml` style (so headings / title get their real sizing). Returns
     * a [`LayoutInfo`] (page dimensions + per-page fingerprints); the caller diffs the fingerprints
     * and calls [`paint_page`] only for the pages that changed.
     * @param {number} scale
     * @returns {LayoutInfo}
     */
    relayout(scale) {
        const ret = wasm.scriptordoc_relayout(this.__wbg_ptr, scale);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return LayoutInfo.__wrap(ret[0]);
    }
    /**
     * Remove the hyperlink at codepoint `off` in paragraph `para`. Returns whether one was removed.
     * @param {number} para
     * @param {number} off
     * @returns {boolean}
     */
    removeHyperlink(para, off) {
        const ret = wasm.scriptordoc_removeHyperlink(this.__wbg_ptr, para, off);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Remove picture `id`. Under Track Changes this is a tracked deletion (the run is marked `w:del`,
     * retained until accepted); otherwise the run + placement are dropped outright. Returns whether it
     * existed. Re-layout after.
     * @param {bigint} id
     * @returns {boolean}
     */
    removeImage(id) {
        const ret = wasm.scriptordoc_removeImage(this.__wbg_ptr, id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Reply to comment `parent` (a threaded child sharing the parent's anchor). Returns the new id.
     * @param {number} parent
     * @param {string} text
     * @returns {number}
     */
    replyComment(parent, text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_replyComment(this.__wbg_ptr, parent, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Reset picture `id`'s crop (Word's "Reset Crop"): clear `<a:srcRect>` and restore the display
     * extent so the whole image reappears at the same scale. Returns whether it was cropped.
     * Re-layout + repaint after.
     * @param {bigint} id
     * @returns {boolean}
     */
    resetImageCrop(id) {
        const ret = wasm.scriptordoc_resetImageCrop(this.__wbg_ptr, id);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Resolve an anchor (from `caretAnchor`) to a current body `[para, off]`, or
     * `undefined` if the anchored block was deleted. Both a live and a shifted
     * anchor return a position (the caret follows the content); only a deleted
     * block returns nothing.
     * @param {Uint8Array} anchor
     * @returns {Uint32Array | undefined}
     */
    resolveAnchor(anchor) {
        const ptr0 = passArray8ToWasm0(anchor, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_resolveAnchor(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v2;
    }
    /**
     * Mark comment `id`'s thread resolved / unresolved. Returns whether it existed.
     * @param {number} id
     * @param {boolean} resolved
     * @returns {boolean}
     */
    resolveComment(id, resolved) {
        const ret = wasm.scriptordoc_resolveComment(this.__wbg_ptr, id, resolved);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Resolve an anchored range (from `anchorRange`) to current body coordinates
     * `[p1, o1, p2, o2]`, or `undefined` if it no longer resolves.
     * @param {Uint8Array} range
     * @returns {Uint32Array | undefined}
     */
    resolveRange(range) {
        const ptr0 = passArray8ToWasm0(range, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_resolveRange(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v2;
    }
    /**
     * The *resolved* definition of paragraph style `id` as JSON, for prefilling the Modify-Style
     * dialog: `{"size"(half-pts,0=inherit),"bold","italic","color"(hex,""=inherit),"font"(""=inherit),
     * "lineSpacing"(240ths,0=inherit),"spaceBefore"(twips,-1=inherit),"spaceAfter"(twips,-1=inherit)}`.
     * Resolved through the style's `basedOn` chain over docDefaults, with any runtime edit folded in -
     * so the dialog opens showing what the style currently renders at.
     * @param {string} id
     * @returns {string}
     */
    resolveStyleProps(id) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.scriptordoc_resolveStyleProps(this.__wbg_ptr, ptr0, len0);
            deferred2_0 = ret[0];
            deferred2_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Every reviewer who authored a tracked change or comment, as a JSON array (the "Show Markup"
     * legend): each object has `name`, `color` (the author's hue as `#rrggbb`), and `hidden`. Sorted
     * by name.
     * @returns {string}
     */
    reviewers() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_reviewers(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The resolved formatting of codepoint `[start, end)` in paragraph `para`, for driving toolbar
     * state. Boolean getters are tri-state via `*IsMixed` (true = the selection spans both).
     * @param {number} para
     * @param {number} start
     * @param {number} end
     * @returns {SelFormat}
     */
    selectionFormat(para, start, end) {
        const ret = wasm.scriptordoc_selectionFormat(this.__wbg_ptr, para, start, end);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return SelFormat.__wrap(ret[0]);
    }
    /**
     * Selection highlight rectangles between two caret positions (codepoint offsets), flattened
     * `[x, y, w, h, ...]` (device px). Empty when the selection is collapsed.
     * @param {number} p1
     * @param {number} o1
     * @param {number} p2
     * @param {number} o2
     * @returns {Float32Array}
     */
    selectionRects(p1, o1, p2, o2) {
        const ret = wasm.scriptordoc_selectionRects(this.__wbg_ptr, p1, o1, p2, o2);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Set which story the caret is in (from a namespaced paragraph index), so undo/redo route to the
     * right child document. The JS shell calls this on every selection change.
     * @param {number} para
     */
    setActiveStory(para) {
        wasm.scriptordoc_setActiveStory(this.__wbg_ptr, para);
    }
    /**
     * Set paragraph alignment ("left" | "center" | "right" | "justify"). Re-paint after.
     * @param {number} para
     * @param {string} align
     */
    setAlignment(para, align) {
        const ptr0 = passStringToWasm0(align, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setAlignment(this.__wbg_ptr, para, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set the current author: a stable `id` (audit trail) + a display `name` (stamped as `w:author`
     * on tracked changes, and shown in the hover tooltip).
     * @param {string} id
     * @param {string} name
     */
    setAuthor(id, name) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setAuthor(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Turn revision balloons on/off (Word's "Show Revisions in Balloons"). When on, tracked deletions
     * move from the line into right-margin bubbles; it only takes visible effect in the markup display
     * modes (All / Simple). Re-layout + paint after.
     * @param {boolean} on
     */
    setBalloons(on) {
        wasm.scriptordoc_setBalloons(this.__wbg_ptr, on);
    }
    /**
     * Set the caret cell's shading fill (`fill` = RGB hex without `#`; `""` clears it). With
     * Track-Changes on it's a tracked cell-property change (`w:tcPrChange`); otherwise direct. Returns
     * whether the caret was in a table cell (so the caller re-layouts). Re-layout + re-paint after.
     * @param {number} para
     * @param {string} fill
     * @returns {boolean}
     */
    setCellShading(para, fill) {
        const ptr0 = passStringToWasm0(fill, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setCellShading(this.__wbg_ptr, para, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Replace the default footer with plain `text`. Re-paint after.
     * @param {string} text
     */
    setFooterText(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setFooterText(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set the page whose header/footer instance the caret is on (the JS shell computes it from the
     * click on a multi-page document). Lets the caret resolve to that instance, not always page 1.
     * @param {number} page
     */
    setHeaderFooterPage(page) {
        wasm.scriptordoc_setHeaderFooterPage(this.__wbg_ptr, page);
    }
    /**
     * Replace the default header with plain `text`. Re-paint after.
     * @param {string} text
     */
    setHeaderText(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setHeaderText(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set picture `id`'s crop (`<a:srcRect>` l/t/r/b, thousandths of a percent, 0..100000 - the share
     * of each edge to cut). Returns whether it existed. Re-layout + repaint after.
     * @param {bigint} id
     * @param {number} l
     * @param {number} t
     * @param {number} r
     * @param {number} b
     * @returns {boolean}
     */
    setImageCrop(id, l, t, r, b) {
        const ret = wasm.scriptordoc_setImageCrop(this.__wbg_ptr, id, l, t, r, b);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Make picture `id` floating (positioned + text-wrapped) or inline (in the flow). `wrap` is the
     * wrap type (`square` / `tight` / `topAndBottom` / `through` / `none`); `behind` paints it under
     * the text. Returns whether it existed. Re-layout after.
     * @param {bigint} id
     * @param {boolean} floating
     * @param {string} wrap
     * @param {boolean} behind
     * @returns {boolean}
     */
    setImageFloating(id, floating, wrap, behind) {
        const ptr0 = passStringToWasm0(wrap, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setImageFloating(this.__wbg_ptr, id, floating, ptr0, len0, behind);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Position floating picture `id`: `h_from`/`v_from` are the `relativeFrom` origins
     * (`column`/`page`/`margin`/...) and `x_emu`/`y_emu` the offset from it (EMU). Clears any
     * alignment (an explicit offset wins, as in Word's drag-to-move). No-op on an inline picture.
     * Returns whether it existed and was floating. Re-layout after.
     * @param {bigint} id
     * @param {string} h_from
     * @param {number} x_emu
     * @param {string} v_from
     * @param {number} y_emu
     * @returns {boolean}
     */
    setImagePosition(id, h_from, x_emu, v_from, y_emu) {
        const ptr0 = passStringToWasm0(h_from, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(v_from, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setImagePosition(this.__wbg_ptr, id, ptr0, len0, x_emu, ptr1, len1, y_emu);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Resize picture `id` to `w_emu` x `h_emu` (EMU). Returns whether it existed. Re-layout after.
     * @param {bigint} id
     * @param {number} w_emu
     * @param {number} h_emu
     * @returns {boolean}
     */
    setImageSize(id, w_emu, h_emu) {
        const ret = wasm.scriptordoc_setImageSize(this.__wbg_ptr, id, w_emu, h_emu);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Set the first-line indent (twips; negative = hanging). Re-paint after.
     * @param {number} para
     * @param {number} twips
     */
    setIndentFirst(para, twips) {
        const ret = wasm.scriptordoc_setIndentFirst(this.__wbg_ptr, para, twips);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set the left indent (twips). Re-paint after.
     * @param {number} para
     * @param {number} twips
     */
    setIndentLeft(para, twips) {
        const ret = wasm.scriptordoc_setIndentLeft(this.__wbg_ptr, para, twips);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set the right indent (twips). Re-paint after.
     * @param {number} para
     * @param {number} twips
     */
    setIndentRight(para, twips) {
        const ret = wasm.scriptordoc_setIndentRight(this.__wbg_ptr, para, twips);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set page orientation (true = landscape); swaps the page dimensions if needed.
     * @param {boolean} landscape
     */
    setLandscape(landscape) {
        wasm.scriptordoc_setLandscape(this.__wbg_ptr, landscape);
    }
    /**
     * Set line spacing in 240ths (240 = single, 360 = 1.5, 480 = double). Re-paint after.
     * @param {number} para
     * @param {number} x240
     */
    setLineSpacing(para, x240) {
        const ret = wasm.scriptordoc_setLineSpacing(this.__wbg_ptr, para, x240);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set the page margins in twips (1 inch = 1440).
     * @param {number} top
     * @param {number} right
     * @param {number} bottom
     * @param {number} left
     */
    setMargins(top, right, bottom, left) {
        wasm.scriptordoc_setMargins(this.__wbg_ptr, top, right, bottom, left);
    }
    /**
     * Hand the engine the current wall-clock time (ISO-8601) to stamp on the next tracked change.
     * The engine never invents time; the JS shell calls this with `new Date().toISOString()` before
     * a tracked edit.
     * @param {string} iso
     */
    setNow(iso) {
        const ptr0 = passStringToWasm0(iso, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setNow(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set (or clear) paragraph `para`'s list numbering (`w:numPr`): `num_id < 0` removes it from any
     * list; otherwise it joins list `num_id` at level `ilvl` (a negative `ilvl` defaults to 0). With
     * Track-Changes on this records a `w:pPrChange` (a numbering change is a paragraph-property
     * change); otherwise it applies directly. Routed through the shared edit path. Re-layout + re-paint.
     * @param {number} para
     * @param {number} num_id
     * @param {number} ilvl
     */
    setNumbering(para, num_id, ilvl) {
        const ret = wasm.scriptordoc_setNumbering(this.__wbg_ptr, para, num_id, ilvl);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set the page size in twips (Letter = 12240x15840, A4 = 11906x16838).
     * @param {number} width
     * @param {number} height
     */
    setPageSize(width, height) {
        wasm.scriptordoc_setPageSize(this.__wbg_ptr, width, height);
    }
    /**
     * Set (or clear, when `style` is empty -> Normal) paragraph `para`'s named style (`w:pStyle`).
     * With Track-Changes on this records a `w:pPrChange` (a style change is a paragraph-property
     * change); otherwise it applies directly. Routed through the shared edit path. Re-layout + re-paint.
     * @param {number} para
     * @param {string} style
     */
    setParagraphStyle(para, style) {
        const ptr0 = passStringToWasm0(style, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setParagraphStyle(this.__wbg_ptr, para, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Filter a reviewer's markup in / out of the display by `w:author` name (display-only; the model
     * is untouched). Hidden reviewers' tracked changes + comments are suppressed on the next
     * [`relayout`]. Re-layout + re-paint after.
     * @param {string} author
     * @param {boolean} hidden
     */
    setReviewerHidden(author, hidden) {
        const ptr0 = passStringToWasm0(author, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setReviewerHidden(this.__wbg_ptr, ptr0, len0, hidden);
    }
    /**
     * Set the caret row's height in twips (`twips = 0` clears it; `exact` = exact rule, else at-least).
     * Tracked as `w:trPrChange` when Track-Changes is on. Returns whether the caret was in a table row.
     * @param {number} para
     * @param {number} twips
     * @param {boolean} exact
     * @returns {boolean}
     */
    setRowHeight(para, twips, exact) {
        const ret = wasm.scriptordoc_setRowHeight(this.__wbg_ptr, para, twips, exact);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Edit style `id`'s *definition* (Word's Modify-Style): every paragraph resolving through `id`
     * re-renders with the new properties. Per-field merge - each argument is a sentinel meaning
     * "leave this field unchanged" so the dialog can write only what the user touched:
     * `size`/`line_spacing`/`space_before`/`space_after` < 0 = unchanged (else the value);
     * `bold`/`italic` < 0 = unchanged, 0 = off, 1 = on; `color`/`font` empty = unchanged. Direct, not
     * a tracked revision (Word doesn't redline a style-definition change). Body story only. Re-layout.
     * @param {string} id
     * @param {number} size
     * @param {number} bold
     * @param {number} italic
     * @param {string} color
     * @param {string} font
     * @param {number} line_spacing
     * @param {number} space_before
     * @param {number} space_after
     * @param {string} align
     * @param {string} line_rule
     */
    setStyleProps(id, size, bold, italic, color, font, line_spacing, space_before, space_after, align, line_rule) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(font, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(align, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passStringToWasm0(line_rule, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len4 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setStyleProps(this.__wbg_ptr, ptr0, len0, size, bold, italic, ptr1, len1, ptr2, len2, line_spacing, space_before, space_after, ptr3, len3, ptr4, len4);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set a uniform single-line border on every edge of the caret's table (`size_eighths` = line
     * weight in eighths of a point, `0` removes all borders; `color` = RGB hex without `#`). Tracked as
     * `w:tblPrChange` when Track-Changes is on. Returns whether the caret was in a table.
     * @param {number} para
     * @param {number} size_eighths
     * @param {string} color
     * @returns {boolean}
     */
    setTableBorders(para, size_eighths, color) {
        const ptr0 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.scriptordoc_setTableBorders(this.__wbg_ptr, para, size_eighths, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Turn Track-Changes (suggesting) mode on/off. While on, typing / deleting author tracked
     * changes attributed to the current author instead of editing the document directly. Ignored when
     * tracking is **locked** (see [`set_track_locked`](Self::set_track_locked)) - it stays on.
     * @param {boolean} on
     */
    setTrackChanges(on) {
        wasm.scriptordoc_setTrackChanges(this.__wbg_ptr, on);
    }
    /**
     * Set how tracked changes are displayed: `all` (insertions underlined + deletions struck, in
     * author colours), `simple`/`none` (deletions hidden - the Final view), or `original`
     * (insertions hidden). Unknown values are ignored. Call [`relayout`] + re-paint after. The
     * non-`all` modes are render/preview only: the caret geometry still indexes the full
     * (All-Markup) text, so edit in `all`.
     * @param {string} mode
     */
    setTrackDisplay(mode) {
        const ptr0 = passStringToWasm0(mode, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.scriptordoc_setTrackDisplay(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Lock / unlock Track-Changes (Review > Lock Tracking): while locked, tracking can't be turned
     * off (and is forced on). v1 is session state, not yet persisted to `settings.xml`.
     * @param {boolean} locked
     */
    setTrackLocked(locked) {
        wasm.scriptordoc_setTrackLocked(this.__wbg_ptr, locked);
    }
    /**
     * A full, self-contained snapshot of the document (history + state) - what a
     * joining client ships, and the merge unit the server sends on join.
     * @returns {Uint8Array}
     */
    snapshot() {
        const ret = wasm.scriptordoc_snapshot(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Split (unmerge) the caret's horizontally-merged cell back into single columns. Returns the caret,
     * or `-1` if not in a table / the cell isn't merged.
     * @param {number} para
     * @returns {number}
     */
    splitCellHorizontal(para) {
        const ret = wasm.scriptordoc_splitCellHorizontal(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Split (unmerge) the caret's vertically-merged cell. Returns the caret, or `-1` if not in a table /
     * the cell isn't a vertical-merge anchor.
     * @param {number} para
     * @returns {number}
     */
    splitCellVertical(para) {
        const ret = wasm.scriptordoc_splitCellVertical(this.__wbg_ptr, para);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Split paragraph `para` at codepoint `off` (the Enter key) - text from `off` onward moves to a
     * new paragraph after it. Routed through the shared `scriptor_edit::apply` path. Re-paint after.
     * @param {number} para
     * @param {number} off
     */
    splitParagraph(para, off) {
        const ret = wasm.scriptordoc_splitParagraph(this.__wbg_ptr, para, off);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The Styles gallery as a JSON array (Title / Subtitle / Heading N / Normal / ... - the document's
     * quick styles), for the Home tab's Styles gallery. Each entry carries the style's resolved preview
     * formatting so the gallery can render each name in its own look:
     * `{"id","name","size"(half-points,0=inherit),"bold","italic","color"(hex,""=inherit),"font"}`.
     * @returns {string}
     */
    styleGallery() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.scriptordoc_styleGallery(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Table context for paragraph `para`: `[row, col, rowCount, colCount]` (cell indices), or an
     * empty array when the paragraph isn't inside a table. Drives the table context menu.
     * @param {number} para
     * @returns {Uint32Array}
     */
    tableContext(para) {
        const ret = wasm.scriptordoc_tableContext(this.__wbg_ptr, para);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The current document serialized to OOXML `word/document.xml` (the edited body). Hook for
     * "save"; full `.docx` re-packaging (re-zip with the source's other parts) is a follow-up.
     * @returns {string}
     */
    toDocumentXml() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.scriptordoc_toDocumentXml(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Save the whole document to `.docx` bytes - the original package re-zipped with the edited
     * body + header/footer parts (or a minimal package for a from-scratch document).
     * @returns {Uint8Array}
     */
    toDocx() {
        const ret = wasm.scriptordoc_toDocx(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Toggle whether paragraph `para` is expanded to inline All-Markup while the document is in
     * Simple Markup (click-to-expand). Returns the new state. Re-layout + paint after. The override is
     * only consulted in Simple Markup, but the toggle is always recorded.
     * @param {number} para
     * @returns {boolean}
     */
    toggleParagraphExpanded(para) {
        const ret = wasm.scriptordoc_toggleParagraphExpanded(this.__wbg_ptr, para);
        return ret !== 0;
    }
    /**
     * The tracked change under `(para, off)` for the hover tooltip + click popup, or `undefined`
     * when the point isn't over a change.
     * @param {number} para
     * @param {number} off
     * @returns {TrackHit | undefined}
     */
    trackAt(para, off) {
        const ret = wasm.scriptordoc_trackAt(this.__wbg_ptr, para, off);
        return ret === 0 ? undefined : TrackHit.__wrap(ret);
    }
    /**
     * Whether Track-Changes mode is on.
     * @returns {boolean}
     */
    trackChangesOn() {
        const ret = wasm.scriptordoc_trackChangesOn(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether Track-Changes is locked on.
     * @returns {boolean}
     */
    trackLocked() {
        const ret = wasm.scriptordoc_trackLocked(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Undo the last local edit (Ctrl+Z) in the **active story** (body / header / footer - each child
     * owns its own undo history). Returns whether anything changed. Re-paint after. The active story
     * is set from the caret via [`set_active_story`](Self::set_active_story).
     * @returns {boolean}
     */
    undo() {
        const ret = wasm.scriptordoc_undo(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Update (regenerate) the document's TOC in place: delete the old field block, then rebuild it from
     * the current headings + page numbers (Word's F9). When there's no existing TOC, insert one at the
     * caret `at` instead. Returns whether a TOC was written. Re-layout + re-paint after.
     * @param {number} at
     * @returns {boolean}
     */
    updateToc(at) {
        const ret = wasm.scriptordoc_updateToc(this.__wbg_ptr, at);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Total word count across the document, computed in a single pass. (The TS shell previously
     * looped `paragraphText` per paragraph, which re-materialized the whole tree each time - O(n^2),
     * and a UI freeze once tables put 100+ paragraphs in the flow.)
     * @returns {number}
     */
    wordCount() {
        const ret = wasm.scriptordoc_wordCount(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) ScriptorDoc.prototype[Symbol.dispose] = ScriptorDoc.prototype.free;

/**
 * The selection's resolved formatting, for the toolbar. Each boolean has a companion `*Mixed`
 * getter (true when the selection spans both states); `size` is 0 when mixed/unset; `color` /
 * `font` are empty strings when mixed/unset.
 */
export class SelFormat {
    static __wrap(ptr) {
        const obj = Object.create(SelFormat.prototype);
        obj.__wbg_ptr = ptr;
        SelFormatFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SelFormatFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_selformat_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get bold() {
        const ret = wasm.selformat_bold(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get boldMixed() {
        const ret = wasm.selformat_boldMixed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Text color `RRGGBB`; empty when mixed or unset.
     * @returns {string}
     */
    get color() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.selformat_color(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Font family; empty when mixed or unset.
     * @returns {string}
     */
    get font() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.selformat_font(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Highlight color name; empty when none or mixed.
     * @returns {string}
     */
    get highlight() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.selformat_highlight(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {boolean}
     */
    get italic() {
        const ret = wasm.selformat_italic(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get italicMixed() {
        const ret = wasm.selformat_italicMixed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Font size in half-points (OOXML `w:sz`); 0 when mixed or unset.
     * @returns {number}
     */
    get size() {
        const ret = wasm.selformat_size(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get strike() {
        const ret = wasm.selformat_strike(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get strikeMixed() {
        const ret = wasm.selformat_strikeMixed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get underline() {
        const ret = wasm.selformat_underline(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get underlineMixed() {
        const ret = wasm.selformat_underlineMixed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Vertical alignment ("superscript" / "subscript"); empty when baseline or mixed.
     * @returns {string}
     */
    get vertAlign() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.selformat_vertAlign(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) SelFormat.prototype[Symbol.dispose] = SelFormat.prototype.free;

/**
 * A tracked change under a point (hover tooltip + click popup): the revision id, its kind (`"ins"`
 * or `"del"`), the author, the ISO date, and the change's text.
 */
export class TrackHit {
    static __wrap(ptr) {
        const obj = Object.create(TrackHit.prototype);
        obj.__wbg_ptr = ptr;
        TrackHitFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TrackHitFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_trackhit_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get author() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trackhit_author(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get date() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trackhit_date(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get id() {
        const ret = wasm.trackhit_id(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * `"ins"` (insertion) or `"del"` (deletion).
     * @returns {string}
     */
    get kind() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trackhit_kind(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get text() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.trackhit_text(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) TrackHit.prototype[Symbol.dispose] = TrackHit.prototype.free;

/**
 * Compare two `.docx` documents (blacklining): produce a **redline** - `original` with every
 * difference as an author-attributed tracked change - plus the change manifest. Returns a
 * `{ redline: Uint8Array, manifest: string }` object: `redline` is a Word-openable tracked-changes
 * `.docx` the view can open like any document (its changes then appear in the reviewing pane);
 * `manifest` is the deterministic change set as JSON (`{"changes":[…]}`) the UI parses for a
 * summary / change-list. The redline is attributed to `author` and dated `date` (a parameter, so the
 * result is deterministic).
 * @param {Uint8Array} original
 * @param {Uint8Array} revised
 * @param {string} author
 * @param {string} date
 * @param {boolean} detect_formatting
 * @param {boolean} detect_moves
 * @param {boolean} ignore_whitespace
 * @param {boolean} ignore_case
 * @returns {any}
 */
export function compareDocx(original, revised, author, date, detect_formatting, detect_moves, ignore_whitespace, ignore_case) {
    const ptr0 = passArray8ToWasm0(original, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(revised, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(author, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(date, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.compareDocx(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, detect_formatting, detect_moves, ignore_whitespace, ignore_case);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * EMU (English Metric Units, 914400/inch - the unit the image model + `.docx` speak) -> canvas px at
 * zoom `scale` (1.0 = 96 px/in). The view sizes selection handles + draws crop overlays in px, so it
 * converts at this boundary rather than hard-coding the magic numbers. Natural-image-size conversion
 * (DPI-independent, 96 px/in) passes `scale = 1.0`; on-screen handle math passes the current zoom.
 * @param {number} emu
 * @param {number} scale
 * @returns {number}
 */
export function emuToPx(emu, scale) {
    const ret = wasm.emuToPx(emu, scale);
    return ret;
}

/**
 * Every bundled substitute face, so the DOM chrome can register `@font-face` rules and preview a
 * font / style menu in the SAME clone the canvas renders (true WYSIWYG - the OS has none of these MS
 * fonts installed). One entry per face: `family` is the MS name it substitutes for (so a CSS
 * `font-family:'Cambria'` label draws in Caladea, matching what the shaper paints), `bold`/`italic`
 * are the style flags, and `bytes` is the raw font data (the exact bytes embedded in this module -
 * no second copy shipped as a web asset). The DejaVu broad-Unicode fallback is skipped: it stands in
 * for no MS family (`substitute_family` never returns it), so it is never a menu entry.
 * @returns {Array<any>}
 */
export function fontFaces() {
    const ret = wasm.fontFaces();
    return ret;
}

/**
 * Canvas px at zoom `scale` -> EMU (the inverse of [`emu_to_px`]). The view turns a resize-handle
 * drag (px at the current zoom) or a decoded natural size (px at `scale = 1.0`) into the EMU the
 * edit ops want. Returns 0 for a non-positive scale (no sensible inverse).
 * @param {number} px
 * @param {number} scale
 * @returns {number}
 */
export function pxToEmu(px, scale) {
    const ret = wasm.pxToEmu(px, scale);
    return ret;
}

/**
 * Route Rust panics to the browser console (dev ergonomics). Runs on module init.
 */
export function start() {
    wasm.start();
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_fdd633d4bb5dd76a: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_is_function_acc5528be2b923f2: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_0beba4a1980d3eea: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_1fca8072260dd261: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_721f8decd50c87a3: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_ea4887a5f8f9a9db: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_5575218572ead796: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_crypto_38df2bab126b63dc: function(arg0) {
            const ret = arg0.crypto;
            return ret;
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_c44a50d8cfdaebeb: function() { return handleError(function (arg0, arg1) {
            arg0.getRandomValues(arg1);
        }, arguments); },
        __wbg_length_589238bdcf171f0e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_msCrypto_bd5a034af96bcba6: function(arg0) {
            const ret = arg0.msCrypto;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_2e117a478906f062: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_36e147a8ced3c6e0: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_543b875b27789a8f: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_9b650f44b5c44a4e: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_node_84ea875411254db1: function(arg0) {
            const ret = arg0.node;
            return ret;
        },
        __wbg_now_6cc3090463b5237b: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_process_44c7a14e11e9f69e: function(arg0) {
            const ret = arg0.process;
            return ret;
        },
        __wbg_prototypesetcall_d721637c7ca66eb8: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_f724b5db8acf89d2: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_randomFillSync_6c25eac9869eb53c: function() { return handleError(function (arg0, arg1) {
            arg0.randomFillSync(arg1);
        }, arguments); },
        __wbg_require_b4edbdcf3e2a1ef0: function() { return handleError(function () {
            const ret = module.require;
            return ret;
        }, arguments); },
        __wbg_set_4564f7dc44fcb0c9: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_THIS_2fee5048bcca5938: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_ce44e66a4935da8c: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_44f6e0cb5e67cdad: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_168f178805d978fe: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_b0e8ac4ed313fea8: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_versions_276b2795b1c6a219: function(arg0) {
            const ret = arg0.versions;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./scriptor_wasm_bg.js": import0,
    };
}

const LayoutInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_layoutinfo_free(ptr, 1));
const ParaFmtFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_parafmt_free(ptr, 1));
const ScriptorDocFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_scriptordoc_free(ptr, 1));
const SelFormatFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_selformat_free(ptr, 1));
const TrackHitFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_trackhit_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getBigUint64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
    if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.byteLength === 0) {
        cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
    }
    return cachedBigUint64ArrayMemory0;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedBigUint64ArrayMemory0 = null;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('scriptor_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
