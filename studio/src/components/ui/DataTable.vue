<script setup lang="ts">
// The Studio's data table: TanStack Table v9 (headless, the Reka philosophy)
// driving house-styled markup. Born for the graph query results table (the
// hand-rolled one was not working great) and shaped as a
// ui/* wrapper so the next table joins instead of hand-rolling again.
//
// Positional rows deliberately: query results arrive as columns + row arrays,
// not keyed objects, and forcing callers to zip records would be ceremony.
// `format` turns a cell into its display string - sorting compares that same
// string (numeric-aware), so what you see ordered is what ordered it.
import { computed } from 'vue'
import {
  FlexRender,
  createSortedRowModel,
  rowSortingFeature,
  tableFeatures,
  useTable,
} from '@tanstack/vue-table'

const props = withDefaults(
  defineProps<{
    columns: { label: string; numeric?: boolean }[]
    rows: unknown[][]
    /** cell -> display string; also the sort key. Default: String(). */
    format?: (cell: unknown, colIndex: number) => string
  }>(),
  { format: undefined },
)

const fmt = (cell: unknown, i: number): string =>
  props.format ? props.format(cell, i) : String(cell ?? '')

const _features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
})

const columnDefs = computed(() =>
  props.columns.map((c, i) => ({
    id: String(i),
    header: c.label,
    accessorFn: (row: unknown[]) => row[i],
    cell: (info: { getValue: () => unknown }) => fmt(info.getValue(), i),
    // Passed per column, so it needs no sortFns registration (v9 doc).
    sortFn: (a: { original: unknown[] }, b: { original: unknown[] }) => {
      const x = fmt(a.original[i], i)
      const y = fmt(b.original[i], i)
      const nx = Number(x)
      const ny = Number(y)
      if (x !== '' && y !== '' && Number.isFinite(nx) && Number.isFinite(ny)) return nx - ny
      // empties last, so a sparse column doesn't lead with blanks
      if (x === '' || y === '') return x === '' ? 1 : -1
      return x.localeCompare(y)
    },
  })),
)

// Positional dynamic columns sit outside v9's typed-accessor sweet spot; the
// defs above are shape-correct, asserted once at this boundary.
const table = useTable({
  features: _features,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  columns: columnDefs as any,
  data: computed(() => props.rows),
})
</script>

<template>
  <div class="dt">
    <table>
      <thead>
        <tr v-for="hg in table.getHeaderGroups()" :key="hg.id">
          <th
            v-for="header in hg.headers"
            :key="header.id"
            :class="{ 'dt__th--num': props.columns[Number(header.column.id)]?.numeric }"
            :aria-sort="
              header.column.getIsSorted() === 'asc'
                ? 'ascending'
                : header.column.getIsSorted() === 'desc'
                  ? 'descending'
                  : undefined
            "
            @click="header.column.toggleSorting()"
          >
            <FlexRender :header="header" />
            <span v-if="header.column.getIsSorted()" class="dt__sort">
              {{ header.column.getIsSorted() === 'asc' ? '▲' : '▼' }}
            </span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in table.getRowModel().rows" :key="row.id">
          <td
            v-for="cell in row.getAllCells()"
            :key="cell.id"
            :class="{ 'dt__td--num': props.columns[Number(cell.column.id)]?.numeric }"
          >
            <FlexRender :cell="cell" />
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.dt {
  overflow: auto;
  min-height: 0;
}
.dt table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-xs);
}
.dt th,
.dt td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--pk-border-subtle);
  text-align: left;
  white-space: nowrap;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.dt th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--pk-bg-surface);
  color: var(--pk-text-muted);
  font-weight: 500;
  cursor: pointer;
  user-select: none;
}
.dt th:hover {
  color: var(--pk-text-primary);
}
.dt tbody tr:hover td {
  background: var(--pk-bg-base);
}
.dt td {
  font-variant-numeric: tabular-nums;
}
.dt__th--num,
.dt__td--num {
  text-align: right;
}
.dt__sort {
  font-size: 9px;
  margin-left: 3px;
}
</style>
