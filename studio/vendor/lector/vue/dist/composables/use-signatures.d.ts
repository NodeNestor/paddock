/** Composable for reading digital signature metadata. */
export declare function useSignatures(): {
    getCount: (docId: import("@truespar/lector-core").DocumentId) => Promise<number>;
    getInfo: (docId: import("@truespar/lector-core").DocumentId, sigIndex: number) => Promise<import("@truespar/lector-core").SignatureInfo>;
    getAllInfo: (docId: import("@truespar/lector-core").DocumentId) => Promise<import("@truespar/lector-core").SignatureInfo[]>;
};
//# sourceMappingURL=use-signatures.d.ts.map