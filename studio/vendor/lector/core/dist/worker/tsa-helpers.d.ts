/**
 * RFC 3161 Time-Stamp Authority (TSA) helpers for PAdES B-T signatures.
 *
 * This file does NOT perform any cryptographic operations. It only:
 *   - Builds/parses minimal ASN.1/DER structures (TimeStampReq, TimeStampResp)
 *   - Walks an existing CMS BER structure to find the signature value
 *
 * All actual crypto (hashing, signing) happens in the WASM/Botan layer.
 */
/**
 * Build an RFC 3161 TimeStampReq DER for a given message imprint.
 *
 *   TimeStampReq ::= SEQUENCE {
 *     version INTEGER (1),
 *     messageImprint MessageImprint,
 *     reqPolicy OBJECT IDENTIFIER OPTIONAL,
 *     nonce INTEGER OPTIONAL,
 *     certReq BOOLEAN DEFAULT FALSE,
 *     extensions [0] IMPLICIT Extensions OPTIONAL
 *   }
 */
export declare function buildTimeStampReq(sigHash: Uint8Array): Uint8Array;
/**
 * Extract the TimeStampToken from a TSA response.
 *
 *   TimeStampResp ::= SEQUENCE {
 *     status PKIStatusInfo,
 *     timeStampToken TimeStampToken OPTIONAL
 *   }
 *
 *   PKIStatusInfo ::= SEQUENCE {
 *     status PKIStatus, -- INTEGER, 0 = granted, 1 = grantedWithMods
 *     ...
 *   }
 *
 * Returns the TimeStampToken bytes (a ContentInfo SignedData) or throws.
 */
export declare function extractTimeStampToken(response: Uint8Array): Uint8Array;
/**
 * Walk a CMS SignedData BER structure to find the signature value
 * (the OCTET STRING in the first SignerInfo).
 *
 * CMS structure:
 *   ContentInfo SEQUENCE {
 *     contentType OID (signedData)
 *     [0] EXPLICIT {
 *       SignedData SEQUENCE {
 *         version, digestAlgorithms, encapContentInfo,
 *         [0] certificates IMPLICIT (optional),
 *         [1] crls IMPLICIT (optional),
 *         signerInfos SET {
 *           SignerInfo SEQUENCE {
 *             version, sid, digestAlgorithm,
 *             [0] signedAttrs (optional),
 *             signatureAlgorithm,
 *             signature OCTET STRING  ← we want this
 *             [1] unsignedAttrs (optional)
 *           }
 *         }
 *       }
 *     }
 *   }
 */
export declare function findSignatureValueInCms(cms: Uint8Array): Uint8Array;
//# sourceMappingURL=tsa-helpers.d.ts.map