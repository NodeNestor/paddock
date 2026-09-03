/**
 * Vendor-neutral interface for plugging external signing services into
 * Lector's PDF signing pipeline.
 *
 * **Status:** Type-only contract. There is no registry, no consumer, and
 * no implementation in core yet — see
 * `docs/architecture/eidas-bankid-signing.md` for the design and the
 * future work that will activate it.
 *
 * **Design intent.** Lector core stays neutral with respect to any specific
 * signing technology. The existing local-PFX path will eventually be
 * exposed as one concrete `PdfSigningProvider`. Remote providers (BankID
 * via TIC's Ormeo.Identity, Cloud Signature Consortium QTSPs, customer
 * webhook backends, smart card / PKCS#11 in WASM, etc.) live in *separate*
 * packages so that any Lector consumer — including TIC competitors — can
 * pick the providers they need without inheriting vendor-specific code.
 *
 * The lifecycle of a remote-signing operation is:
 *
 *  1. Lector worker calls `prepareSignaturePlaceholder` (already shipped
 *     in `worker/signing-ops.ts`) which places the widget, patches
 *     `/ByteRange`, and computes the byte-range hash.
 *  2. Main thread calls `provider.signHash(input)` and waits for the CMS
 *     SignedData blob. The provider owns *all* user interaction during
 *     this step (modals, redirects, polling, etc.) and is responsible
 *     for honouring the supplied `AbortSignal`.
 *  3. Lector worker calls `embedSignatureCms` to inject the returned CMS
 *     bytes into the placeholder and produce the final signed PDF.
 *
 * The provider is a black box from Lector's perspective: hash in, CMS out,
 * cancellable. Whatever happens in between (BankID dance, HSM call,
 * QTSP authentication, smart card prompt, …) is the provider's concern.
 */
export interface PdfSigningProvider {
    /**
     * Display name shown in the UI when the user picks a signer
     * (e.g. `"Local certificate"`, `"Mobilt BankID"`, `"GlobalSign DSS"`).
     */
    readonly displayName: string;
    /**
     * Hash algorithm the provider expects for the byte-range hash.
     *
     * Lector currently always computes SHA-256 in `prepareSignaturePlaceholder`.
     * Providers that need a different algorithm will require a future
     * extension to the prepare phase. Stated explicitly here so that the
     * mismatch becomes a type error rather than a runtime surprise.
     */
    readonly hashAlgorithm: 'SHA-256' | 'SHA-384' | 'SHA-512';
    /**
     * Sign a PDF byte-range hash and return CMS SignedData ready to embed
     * verbatim into the PDF `/Contents` placeholder.
     *
     * The provider is responsible for:
     *  - any user interaction (its own modals, redirects, polling, ...)
     *  - network communication with the signing service
     *  - honouring `input.signal` for cancellation (close the modal, abort
     *    in-flight requests, release server-side sessions, ...)
     *
     * The returned CMS must be encoded as DER bytes — not base64, not hex.
     */
    signHash(input: SignHashInput): Promise<SignHashOutput>;
}
/**
 * Input passed to `PdfSigningProvider.signHash`. The hash is the only
 * required field; everything else is metadata that providers may forward
 * to their signing service for audit, display, or signed-attribute purposes.
 */
export interface SignHashInput {
    /** The byte-range digest that needs to be signed. */
    readonly hash: Uint8Array;
    /** Algorithm used to produce `hash`. Must match `provider.hashAlgorithm`. */
    readonly hashAlgorithm: 'SHA-256' | 'SHA-384' | 'SHA-512';
    /**
     * Signing time captured at prepare-time. Providers that produce CMS
     * locally must use this exact value as the `signing-time` signed
     * attribute, otherwise the byte-range hash will not validate.
     */
    readonly signingTime: Date;
    /** Original document name — typically the file name. Used for display. */
    readonly documentName?: string;
    /** Reason recorded in the signature (e.g. `"Approval"`, `"Witness"`). */
    readonly reason?: string;
    /** Geographic location of the signer (free text). */
    readonly location?: string;
    /** Contact information for the signer (free text). */
    readonly contactInfo?: string;
    /** Name of the signature field being signed, for audit logging. */
    readonly signatureFieldName?: string;
    /**
     * Cancellation signal. If aborted, the provider must cancel any
     * in-flight requests, close any UI it opened, release any server-side
     * sessions it created, and reject the returned promise with an error
     * whose name is `'AbortError'`.
     */
    readonly signal?: AbortSignal;
}
/**
 * Output returned by `PdfSigningProvider.signHash`. Only `cms` is required
 * — the rest enriches the visual signature appearance and (in the case of
 * `ocspResponses` / `crls`) enables a future PAdES B-LT upgrade.
 */
export interface SignHashOutput {
    /**
     * CMS / PKCS#7 SignedData DER bytes. Embedded verbatim into the PDF
     * `/Contents` placeholder by `embedSignatureCms`.
     */
    readonly cms: Uint8Array;
    /**
     * Subject identifier of the signing certificate as a display string
     * (e.g. an X.500 CN, a personnummer, an organisational identifier).
     * Shown next to the signer name in the visual signature.
     */
    readonly signerSubject?: string;
    /** Human-readable display name of the signer. */
    readonly signerName?: string;
    /**
     * The signing certificate as DER bytes, for embedding in the visual
     * signature appearance or for offline display in the validation panel.
     */
    readonly signerCertificate?: Uint8Array;
    /**
     * Time the provider recorded for the signature. May differ from
     * `SignHashInput.signingTime` if the provider's signing service applies
     * its own server-side timestamp.
     */
    readonly signingTime?: Date;
    /**
     * Optional revocation data. If present, Lector can write a DSS
     * dictionary into the signed PDF to upgrade it to PAdES B-LT.
     */
    readonly ocspResponses?: readonly Uint8Array[];
    /** Optional CRLs accompanying `ocspResponses`. See PAdES B-LT. */
    readonly crls?: readonly Uint8Array[];
}
//# sourceMappingURL=pdf-signing-provider.d.ts.map