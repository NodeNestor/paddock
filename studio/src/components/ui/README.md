# `ui/` - the Reka wrappers

Every interactive widget in the Studio comes from here, and every one of these
is a thin skin over a [Reka UI](https://reka-ui.com) primitive. Reka is pinned
to the current release (2.10.3).

| wrapper | primitive | notes |
|---|---|---|
| `Checkbox` | `CheckboxRoot/Indicator` | `v-model` takes `'indeterminate'`. `glyph="check"` is the menu idiom (a bare tick that holds its column), `square` the standalone box. |
| `Collapsible` | `CollapsibleRoot/Trigger/Content` | the disclosure; replaced every `<details>`. |
| `Dialog` | `DialogRoot/Portal/Overlay/Content/Title/Close` | `role="alertdialog"` also blocks backdrop dismissal - see below. |
| `Menu` + `MenuTrigger` / `MenuContent` / `MenuItem` / `MenuSeparator` / `MenuLabel` | `DropdownMenu*` | actions. Pick `Select` when the thing picked is a value. `MenuLabel` heads a group; keyboard nav steps over it. |
| `NumberField` | `NumberFieldRoot/Input/Increment/Decrement` | |
| `Popover` | `PopoverRoot/Trigger/Portal/Content/Arrow` | |
| `Progress` | `ProgressRoot/Indicator` | pass `label` - it becomes `aria-valuetext`. |
| `RadioGroup` + `RadioItem` | `RadioGroupRoot/Item` | container + item, like Menu. Items are unstyled: the caller's class lands on the item's root and its scoped CSS reaches it. |
| `ResizeHandle` | *(none - see below)* | |
| `Select` | `Select*` | the only dropdown-with-a-value. |
| `Slider` | `SliderRoot/Track/Range/Thumb` | |
| `Switch` | `SwitchRoot/Thumb` | **always pass `label`** - see below. |
| `Tabs` | `TabsRoot/List/Trigger` | value-only; the caller switches its own content. |
| `ToggleGroup` + `ToggleGroupItem` | `ToggleGroupRoot/Item` (`type="single"`) | a segmented control - a *mode*. `RadioGroup` when it's a form value. |
| `Toaster` | `Toast*` | |
| `Tooltip` | `TooltipRoot/Trigger/Portal/Content/Arrow` | the only tooltip; native `title=` is build-gated. |

`CodeEditor` (monaco) and `TextInput` are the two files here that wrap
something other than Reka.

## The rule

**Never hand-write an ARIA widget role.** `role="radio"` is a contract: one tab
stop for the group, arrow keys to move the choice, `aria-checked` maintained as
it moves. Writing the role claims the contract; Reka is what keeps it. Every
place this went wrong had the name and none of the behaviour -
quality cards with `role="radio"` and no arrow keys, an area switcher with
`role="tablist"` and no tabpanel, bulk-select checkboxes that were `<span>`s
with no role at all and no way to reach them from a keyboard, and four progress
bars that were bare `width: %` divs announcing nothing.

`scripts/check-shipped-ui.mjs` fails the build on a widget role written outside
this directory. Roles that describe *structure* rather than a widget (`list`,
`status`, `img`, `separator`, `alert`) are not covered and stay fine.

If a widget is missing, add the wrapper here rather than inlining the primitive
at a call site - that is what keeps one styling and one behaviour per widget.

## Styling a wrapper from its caller: some need `:deep()`

Vue normally puts the caller's `data-v-xxx` on a child component's root
element, which is what lets `.sf__qcard { }` in ServerForm reach a `RadioItem`.
**Reka breaks that for any primitive rendered through an asChild /
roving-focus clone**: the class lands on the element, the scope attribute does
not, and the rule silently matches nothing - the element falls back to native
browser chrome.

Measured by `scripts/probe-reka-scope.mjs`:

| primitive | caller's scope id |
|---|---|
| `ToggleGroupItem`, `RadioGroupItem`, `CheckboxRoot`, `SwitchRoot` | **dropped - caller must use `:deep()`** |
| `ProgressRoot`, `CollapsibleRoot`, `SelectTrigger`, `TabsTrigger` | kept |
| every group Root, and all slot content | kept |

So a caller writes `.sf__qcards :deep(.sf__qcard) { }`, not `.sf__qcard { }` -
scoping through the container keeps the containment the plain rule had.

This shipped once: the header's area switcher, ServerForm's
quality cards and workload pills, and the composer's tool picker all went
native at the same time, which read as "the buttons lost their style". A
`SwitchRoot` nudge in ConnectorsPanel turned out to have been dead the same way
for much longer. `check-shipped-ui.mjs` now fails the build on a bare rule for
a class placed on one of these, and the probe is committed so the table can be
re-measured after a reka-ui upgrade rather than trusted.

## Two things Reka does not solve, and what we do instead

**`Switch` has no accessible name of its own.** `SwitchRoot` renders a
`<button role="switch">` with nothing inside it, and a `<button>` is not a
*labelable* element - so the `<label class="...">` that call sites wrap it in
contributes nothing, and Reka's own `Label` primitive can't help either (it
renders `<label for>`, which needs that same native association). Pass `label`;
it becomes `aria-label`. Without it every toggle in the app is an anonymous
"switch, off". Clicking the surrounding text still doesn't toggle - that one is
open.

**`AlertDialog` is not worth a second template.** Reka's `AlertDialogContent`
is literally `DialogContent` with `role="alertdialog"` and
`pointerDownOutside`/`interactOutside` prevented. `Dialog` reproduces that one
behavioural difference when `role="alertdialog"` is set, instead of duplicating
235 lines that would then have to be changed twice forever.

**`ResizeHandle` is not `Splitter`.** Reka's `Splitter` sizes panels in
*percent* and owns their width. Ours is px, its panels are conditionally
unmounted (`v-if`), and they use their own width internally - `ArtifactPanel`
reads it as a CSS var. Converting means migrating persisted px state to
percentages and restructuring four layouts including the chat screen. What it
was actually missing was the keyboard half of the window-splitter pattern
(arrows, Home/End, `aria-valuenow`); that is now implemented in place.
