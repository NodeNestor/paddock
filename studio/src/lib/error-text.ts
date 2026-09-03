/** Turning a manager error into something a person reads in passing.
 *
 *  The engine writes its refusals as one line: the answer and the fix as
 *  complete sentences, then the arithmetic that produced them. It cannot use a
 *  blank line - the runner emits it as a single log record and the manager
 *  recovers the reason by taking the last non-empty LINE of the tail - so
 *  every surface that shows an error has to do the cutting itself.
 *
 *  Kept here rather than in whichever component noticed first, because there
 *  are four of them (the toast, the fleet row's start error, the same row's
 *  download error, the endpoint page) and they were each slicing at the first
 *  newline. That worked while errors were short and produced a 430-character
 *  run-on the moment one carried its reasoning with it.
 */

/** Characters a TOAST can show before it stops being read. Two clamped lines is
 *  ~160; 220 leaves room for the fix to complete. */
const REASON_BUDGET = 220

/** Characters a table ROW can show. Far less than a toast: the fleet table is
 *  `table-layout: fixed` and the failure cell is ~396px of one line, so the
 *  budget that suits a seven-line toast overflows the column. 110 lands on the
 *  first sentence - the ANSWER (what it needs, what fits) - and leaves the fix
 *  and the ledger to the tooltip. */
export const ROW_BUDGET = 110

/** The part worth showing: the leading sentences, cut on a full stop. */
export function reasonOf(msg: string | null | undefined, budget = REASON_BUDGET): string {
  const text = (msg ?? '').trim()
  if (!text) return ''
  const para = text.split(/\n\s*\n/, 1)[0]?.trim() || text
  if (para.length <= budget) return para
  // Cut at a SENTENCE, not a character count. The last full stop inside the
  // budget is exactly the seam between what a person needs and what a
  // diagnosis needs; a mid-word ellipsis would truncate the fix, which is the
  // half worth reading.
  const head = para.slice(0, REASON_BUDGET)
  const cut = head.lastIndexOf('. ')
  return cut > 40 ? head.slice(0, cut + 1) : `${head.trimEnd()}...`
}
