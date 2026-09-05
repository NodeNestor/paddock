// Gate: MCP OAuth 2.1 authorization-code + PKCE browser-consent flow, end-to-end
// against a mock provider that implements discovery, DCR, PKCE, token exchange,
// and a Bearer-gated MCP endpoint. Proves: authorize -> consent -> code exchange ->
// token persist -> redaction -> the stored token rides as Bearer on connect.
//
//   node mcp-oauth-flow.mjs
//
// Requires paddock running on PADDOCK (default :11540).

import http from 'node:http'
import crypto from 'node:crypto'

const PADDOCK = process.env.PADDOCK || 'http://127.0.0.1:11540'
const MOCK_PORT = Number(process.env.MOCK_PORT || 3778)
const MOCK = `http://127.0.0.1:${MOCK_PORT}`
const ACCESS_TOKEN = 'mcp-access-' + crypto.randomBytes(6).toString('hex')

let PASS = true
const check = (name, ok, extra = '') => {
  if (!ok) PASS = false
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ' - ' + extra : ''}`)
}

// ── mock OAuth provider + MCP resource ───────────────────────────────────────
const record = {
  registered: null, // DCR client_id issued
  tokenReqs: [], // { client_id, client_secret, pkceOk }
  mcpBearer: null, // Authorization seen on the MCP endpoint
  pkceChallenges: new Map(), // code -> code_challenge
}

const b64urlSha256 = (v) => crypto.createHash('sha256').update(v).digest('base64url')
const readBody = (req) =>
  new Promise((res) => {
    let b = ''
    req.on('data', (c) => (b += c))
    req.on('end', () => res(b))
  })
const json = (res, code, obj, headers = {}) => {
  res.writeHead(code, { 'Content-Type': 'application/json', ...headers })
  res.end(JSON.stringify(obj))
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, MOCK)
  const path = u.pathname

  // RFC 8414 discovery (any well-known-suffixed path rmcp probes).
  if (req.method === 'GET' && path.includes('oauth-authorization-server')) {
    return json(res, 200, {
      issuer: MOCK,
      authorization_endpoint: `${MOCK}/authorize`,
      token_endpoint: `${MOCK}/token`,
      registration_endpoint: `${MOCK}/register`,
      response_types_supported: ['code'],
      grant_types_supported: ['authorization_code', 'refresh_token'],
      code_challenge_methods_supported: ['S256'],
      token_endpoint_auth_methods_supported: ['client_secret_post', 'none'],
      scopes_supported: ['read', 'write'],
    })
  }
  // No RFC 9728 protected-resource metadata -> force auth-server fallback.
  if (path.includes('oauth-protected-resource')) {
    res.writeHead(404).end()
    return
  }

  // Dynamic Client Registration (public client, PKCE).
  if (req.method === 'POST' && path === '/register') {
    const body = JSON.parse((await readBody(req)) || '{}')
    const client_id = 'dcr-' + crypto.randomBytes(5).toString('hex')
    record.registered = client_id
    return json(res, 201, {
      client_id,
      redirect_uris: body.redirect_uris || [],
      grant_types: ['authorization_code', 'refresh_token'],
      token_endpoint_auth_method: 'none',
      response_types: ['code'],
    })
  }

  // Authorization endpoint - auto-consent, echo state, mint a code bound to PKCE.
  if (req.method === 'GET' && path === '/authorize') {
    const q = u.searchParams
    const redirect = q.get('redirect_uri')
    const state = q.get('state')
    const challenge = q.get('code_challenge')
    const method = q.get('code_challenge_method')
    if (!redirect || !state || !challenge || method !== 'S256' || q.get('response_type') !== 'code') {
      res.writeHead(400).end('bad authorize request')
      return
    }
    const code = 'code-' + crypto.randomBytes(8).toString('hex')
    record.pkceChallenges.set(code, challenge)
    const loc = new URL(redirect)
    loc.searchParams.set('code', code)
    loc.searchParams.set('state', state)
    res.writeHead(302, { Location: loc.toString() }).end()
    return
  }

  // Token endpoint - verify PKCE, return tokens.
  if (req.method === 'POST' && path === '/token') {
    const form = new URLSearchParams(await readBody(req))
    const code = form.get('code')
    const verifier = form.get('code_verifier')
    const challenge = record.pkceChallenges.get(code)
    const pkceOk = !!verifier && !!challenge && b64urlSha256(verifier) === challenge
    record.tokenReqs.push({
      client_id: form.get('client_id'),
      client_secret: form.get('client_secret'),
      pkceOk,
    })
    if (form.get('grant_type') !== 'authorization_code' || !pkceOk) {
      return json(res, 400, { error: 'invalid_grant' })
    }
    return json(res, 200, {
      access_token: ACCESS_TOKEN,
      token_type: 'Bearer',
      expires_in: 3600,
      refresh_token: 'mcp-refresh-' + crypto.randomBytes(6).toString('hex'),
      scope: 'read',
    })
  }

  // MCP resource endpoint - requires the Bearer; speaks minimal Streamable HTTP.
  if (path === '/mcp') {
    const auth = req.headers['authorization'] || null
    if (auth) record.mcpBearer = auth
    if (auth !== `Bearer ${ACCESS_TOKEN}`) {
      res.writeHead(401, { 'WWW-Authenticate': 'Bearer' }).end()
      return
    }
    if (req.method === 'GET') {
      res.writeHead(405).end() // no server-initiated SSE stream
      return
    }
    const msg = JSON.parse((await readBody(req)) || '{}')
    if (msg.method === 'initialize') {
      return json(
        res,
        200,
        {
          jsonrpc: '2.0',
          id: msg.id,
          result: {
            protocolVersion: msg.params?.protocolVersion || '2025-06-18',
            capabilities: { tools: {} },
            serverInfo: { name: 'mock-mcp', version: '0.0.1' },
          },
        },
        { 'Mcp-Session-Id': 'sess-1' },
      )
    }
    if (msg.method === 'notifications/initialized' || msg.id === undefined) {
      res.writeHead(202).end()
      return
    }
    if (msg.method === 'tools/list') {
      return json(res, 200, {
        jsonrpc: '2.0',
        id: msg.id,
        result: { tools: [{ name: 'ping', description: 'ping', inputSchema: { type: 'object' } }] },
      })
    }
    return json(res, 200, { jsonrpc: '2.0', id: msg.id, result: {} })
  }

  res.writeHead(404).end()
})

// ── paddock API helpers ──────────────────────────────────────────────────────
async function api(method, path, body) {
  const r = await fetch(PADDOCK + path, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    redirect: 'manual',
  })
  const text = await r.text()
  let data
  try {
    data = text ? JSON.parse(text) : undefined
  } catch {
    data = text
  }
  return { status: r.status, data, headers: r.headers, text }
}

async function cleanup(label) {
  for (const s of (await api('GET', '/api/mcp')).data || []) {
    if (s.label === label) await api('DELETE', `/api/mcp/${s.id}`)
  }
}

// Drive one full authorize -> consent -> callback dance for a server id.
async function runFlow(id) {
  const auth = await api('POST', `/api/mcp/${id}/authorize`)
  if (auth.status !== 200) return { ok: false, err: `authorize ${auth.status}: ${JSON.stringify(auth.data)}` }
  const authUrl = auth.data.authorization_url
  const parsed = new URL(authUrl)
  const pkce = {
    hasChallenge: !!parsed.searchParams.get('code_challenge'),
    method: parsed.searchParams.get('code_challenge_method'),
    state: parsed.searchParams.get('state'),
    responseType: parsed.searchParams.get('response_type'),
    clientId: parsed.searchParams.get('client_id'),
    startsRight: authUrl.startsWith(`${MOCK}/authorize`),
  }
  // "Browser" hits the consent URL; provider 302s back to our callback.
  const consent = await fetch(authUrl, { redirect: 'manual' })
  const location = consent.headers.get('location')
  if (consent.status !== 302 || !location) return { ok: false, err: `consent ${consent.status}`, pkce }
  // Land on paddock's callback (as the browser would).
  const cb = await fetch(location, { redirect: 'manual' })
  const cbText = await cb.text()
  return { ok: true, pkce, location, callbackStatus: cb.status, callbackText: cbText }
}

async function main() {
  await new Promise((r) => server.listen(MOCK_PORT, '127.0.0.1', r))
  console.log(`mock OAuth+MCP provider on ${MOCK}\n`)

  try {
    // ── 1) PRE-REGISTERED (confidential) client ──────────────────────────────
    await cleanup('oauth-flow-reg')
    const reg = await api('POST', '/api/mcp', {
      label: 'oauth-flow-reg',
      enabled: true,
      transport: {
        type: 'http',
        url: `${MOCK}/mcp`,
        headers: {},
        oauth: { client_id: 'prereg-client', client_secret: 'prereg-secret' },
      },
    })
    const regId = reg.data.id
    const f1 = await runFlow(regId)
    check('prereg: authorize returns PKCE consent URL', f1.ok && f1.pkce.startsRight)
    check('prereg: PKCE S256 challenge present', !!f1.pkce?.hasChallenge && f1.pkce?.method === 'S256')
    check('prereg: response_type=code, client_id forwarded', f1.pkce?.responseType === 'code' && f1.pkce?.clientId === 'prereg-client')
    check('prereg: callback page = success', f1.callbackStatus === 200 && /Authorization complete/i.test(f1.callbackText || ''))
    const t1 = record.tokenReqs.at(-1)
    check('prereg: token exchange used PKCE verifier', !!t1?.pkceOk)
    check('prereg: confidential client sent client_secret', t1?.client_id === 'prereg-client' && t1?.client_secret === 'prereg-secret')

    const got1 = await api('GET', `/api/mcp/${regId}`)
    const o1 = got1.data?.transport?.oauth || {}
    check('prereg: server shows authorized=true', o1.authorized === true)
    check('prereg: token_expires_at exposed', typeof o1.token_expires_at === 'number')
    check('prereg: client_secret masked, id kept', o1.client_secret === '******' && o1.client_id === 'prereg-client')
    check('prereg: NO token material leaked', !JSON.stringify(got1.data).includes(ACCESS_TOKEN) && !JSON.stringify(got1.data).includes('mcp-refresh'))

    // token rides as Bearer on connect (test -> list_tools over the wire)
    const test1 = await api('POST', `/api/mcp/${regId}/test`)
    check('prereg: MCP endpoint received the exact Bearer', record.mcpBearer === `Bearer ${ACCESS_TOKEN}`, record.mcpBearer || 'none')
    check('prereg: /test connected with token (tools listed)', test1.data?.ok === true, JSON.stringify(test1.data))

    // ── 2) DYNAMIC CLIENT REGISTRATION (public) client ───────────────────────
    await cleanup('oauth-flow-dcr')
    record.mcpBearer = null
    const dcr = await api('POST', '/api/mcp', {
      label: 'oauth-flow-dcr',
      enabled: true,
      transport: { type: 'http', url: `${MOCK}/mcp`, headers: {} },
    })
    const dcrId = dcr.data.id
    const f2 = await runFlow(dcrId)
    check('dcr: dynamic registration happened', !!record.registered)
    check('dcr: authorize used the registered client_id', f2.pkce?.clientId === record.registered)
    check('dcr: callback page = success', f2.callbackStatus === 200 && /Authorization complete/i.test(f2.callbackText || ''))
    const t2 = record.tokenReqs.at(-1)
    check('dcr: public client sent NO secret', !t2?.client_secret)
    const got2 = await api('GET', `/api/mcp/${dcrId}`)
    check('dcr: server shows authorized=true', got2.data?.transport?.oauth?.authorized === true)

    // survives an edit that sends no oauth block (Studio DCR buildDoc)
    const edited = structuredClone(got2.data)
    edited.transport.headers = { 'X-Env': 'prod' }
    delete edited.transport.oauth
    await api('PUT', `/api/mcp/${dcrId}`, edited)
    const test2 = await api('POST', `/api/mcp/${dcrId}/test`)
    check('dcr: token survived an edit (still connects)', record.mcpBearer === `Bearer ${ACCESS_TOKEN}` && test2.data?.ok === true, JSON.stringify(test2.data))

    // ── 3) NEGATIVE: unknown/expired state ───────────────────────────────────
    const bad = await fetch(`${PADDOCK}/api/mcp-oauth/callback?code=x&state=bogus-state`, { redirect: 'manual' })
    const badText = await bad.text()
    check('callback with unknown state -> error page', bad.status === 200 && /expired or is unknown/i.test(badText))

    // cleanup
    await api('DELETE', `/api/mcp/${regId}`)
    await api('DELETE', `/api/mcp/${dcrId}`)
  } catch (e) {
    console.error('gate error:', e)
    PASS = false
  } finally {
    server.close()
  }

  console.log('\n' + (PASS ? 'MCP OAUTH FLOW GATE: PASS ✅' : 'MCP OAUTH FLOW GATE: FAIL ❌'))
  process.exit(PASS ? 0 : 1)
}
main()
