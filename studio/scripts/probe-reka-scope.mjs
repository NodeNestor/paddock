// Which Reka primitives drop the CALLER's scoped-CSS attribute?
//
// Vue puts the parent's `data-v-xxx` on a child component's root element, which
// is what lets a caller style a wrapper with its own scoped rule. Reka breaks
// that for any primitive rendered through an asChild/roving-focus clone: the
// class lands on the element, the scope attribute does not, and the rule
// silently matches nothing. That once shipped the Manager and Studio
// with unstyled native buttons.
//
// Run this after any reka-ui upgrade - the answer is a property of THEIR
// internals, so it can change under us:  node scripts/probe-reka-scope.mjs
import { h, defineComponent, createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import * as R from 'reka-ui'

// a style-less passthrough, exactly like our ui/ItemWrappers
const pass = (C) =>
  defineComponent({
    setup: (_, { attrs, slots }) => () => h(C, attrs, { default: () => slots.default?.() }),
  })

// [label, [ancestors, outermost first], leaf, leafProps] - some primitives
// only mount inside their provider (TabsTrigger needs TabsList's roving focus)
const CASES = [
  ['ToggleGroupItem', [[R.ToggleGroupRoot, { type: 'single', modelValue: 'a' }]], R.ToggleGroupItem, { value: 'a' }],
  ['RadioGroupItem', [[R.RadioGroupRoot, { modelValue: 'a' }]], R.RadioGroupItem, { value: 'a' }],
  ['CheckboxRoot', [], R.CheckboxRoot, { modelValue: true }],
  ['SwitchRoot', [], R.SwitchRoot, { modelValue: true }],
  ['ProgressRoot', [], R.ProgressRoot, { modelValue: 5, max: 10 }],
  ['CollapsibleRoot', [], R.CollapsibleRoot, {}],
  ['SelectTrigger', [[R.SelectRoot, {}]], R.SelectTrigger, {}],
  ['TabsTrigger', [[R.TabsRoot, { modelValue: 'a' }], [R.TabsList, {}]], R.TabsTrigger, { value: 'a' }],
]

const Parent = defineComponent({
  __scopeId: 'data-v-parent',
  render: () =>
    h(
      'div',
      null,
      CASES.map(([name, ancestors, Leaf, leafProps]) =>
        ancestors.reduceRight(
          (child, [C, props]) => h(C, props, { default: () => child }),
          h(pass(Leaf), { ...leafProps, class: `PROBE-${name}` }, { default: () => 'x' }),
        ),
      ),
    ),
})

const html = await renderToString(createSSRApp(Parent))
const drops = []
for (const [name] of CASES) {
  const tag = html.match(new RegExp(`<\\w+([^>]*class="PROBE-${name}"[^>]*)>`))
  if (!tag) {
    console.log(`${name.padEnd(18)} not rendered (skipped)`)
    continue
  }
  const ok = tag[1].includes('data-v-parent')
  if (!ok) drops.push(name)
  console.log(`${name.padEnd(18)} ${ok ? 'keeps the caller scope id' : 'DROPS IT -> callers must use :deep()'}`)
}
console.log(
  drops.length
    ? `\nStyle these from a caller only via :deep(): ${drops.join(', ')}`
    : '\nNo primitive drops the caller scope id.',
)
