// The web-search providers an endpoint can be given, and everything the
// manager needs to present one. The `id` strings are the wire values: they
// match paddock_websearch::Provider on the Rust side and are what lands in
// servers/<port>.toml as `web_search_provider`, so they are never localized,
// never prettified, and never reordered for display.
//
// `blurb` exists because five search engines is past the point where a name
// alone is a choice. They differ in kind - one is semantic, one scrapes the
// page, one has its own index - and a user picking between them needs that
// fact, not marketing. Only the SELECTED provider's line is ever shown, so
// this is one line of orientation, not a wall of five.

export interface SearchProvider {
  id: string
  label: string
  /** where the user creates a key for this provider */
  keyUrl: string
  /** what this engine is actually for, in one line */
  blurb: string
}

export const SEARCH_PROVIDERS: SearchProvider[] = [
  {
    id: 'exa',
    label: 'Exa',
    keyUrl: 'https://dashboard.exa.ai',
    blurb: 'Semantic search - matches on meaning rather than keywords.',
  },
  {
    id: 'tavily',
    label: 'Tavily',
    keyUrl: 'https://app.tavily.com',
    blurb: 'Built for models: returns the relevant chunks of each page.',
  },
  {
    id: 'firecrawl',
    label: 'Firecrawl',
    keyUrl: 'https://firecrawl.dev/app/api-keys',
    blurb: 'Fetches every result and returns the full page as markdown.',
  },
  {
    id: 'brave',
    label: 'Brave',
    keyUrl: 'https://api-dashboard.search.brave.com',
    blurb: 'An independent index - a genuinely different set of results.',
  },
  {
    id: 'perplexity',
    label: 'Perplexity',
    keyUrl: 'https://www.perplexity.ai/account/api/keys',
    blurb: 'The index behind the answer engine, ranked for questions.',
  },
]

export function searchProvider(id: string | null | undefined): SearchProvider | undefined {
  return id ? SEARCH_PROVIDERS.find((p) => p.id === id) : undefined
}

/** Brand spelling for a stored id. An id we don't know is shown as STORED
 *  rather than hidden or guessed at - a config file naming a provider this
 *  build doesn't have is exactly the thing a user needs to see. */
export function searchLabel(id: string | null | undefined): string {
  return searchProvider(id)?.label ?? (id ?? '')
}
