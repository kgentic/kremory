//! Namespace administration (register / upgrade policy) and embedding-backfill
//! maintenance operations.

use napi_derive::napi;
use super::JsMemory;
use crate::convert;

#[napi]
impl JsMemory {
    /// Register a namespace and its policy explicitly, ahead of any writes.
    ///
    /// Wraps `Memory::register_namespace`. Idempotent for same policy; returns
    /// error on downgrade attempt. Calling at startup before `remember` is the
    /// recommended pattern.
    #[napi]
    pub async fn register_namespace(&self, namespace: String) -> napi::Result<()> {
        self.inner
            .register_namespace(kremory::Namespace::new(namespace))
            .await
            .map_err(|e| napi::Error::from_reason(format!("kremory registerNamespace failed: {e}")))
    }

    /// Monotonically upgrade a namespace's immutability from `Mutable` to
    /// `AppendOnly`.
    ///
    /// Wraps `Memory::upgrade_namespace_policy`. One-way ratchet: downgrade
    /// attempts return an error. Idempotent if already `AppendOnly`.
    #[napi]
    pub async fn upgrade_namespace_policy(&self, namespace: String) -> napi::Result<()> {
        self.inner
            .upgrade_namespace_policy(kremory::Namespace::new(namespace))
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory upgradeNamespacePolicy failed: {e}"))
            })
    }

    /// Fill the entity embedding gap — every entity whose `embedding` is
    /// still NULL.
    /// Wraps `Memory::backfill_entity_embeddings`. Unlike a full re-embed,
    /// this NEVER touches a row that already carries an embedding — safe to
    /// run against a live corpus, and idempotent (a second call finds
    /// nothing left to do once the gap is closed).
    ///
    /// Feature-gated behind `content-search` — the substrate method is
    /// `#[cfg(feature = "content-search")]` (the `embedding` column only
    /// exists there), so this binding must be too or a
    /// `--no-default-features` build fails to compile.
    #[cfg(feature = "content-search")]
    #[napi]
    pub async fn backfill_entity_embeddings(
        &self,
        batch_size: u32,
    ) -> napi::Result<convert::JsEmbeddingBackfill> {
        let stats = self
            .inner
            .backfill_entity_embeddings(batch_size as usize)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory backfillEntityEmbeddings failed: {e}"))
            })?;
        Ok(convert::embedding_backfill_to_js(stats))
    }

    /// Fill the fact embedding gap — every fact whose `embedding` is
    /// still NULL. Wraps `Memory::backfill_fact_embeddings`. Same NULL-only,
    /// idempotent contract as `backfillEntityEmbeddings`.
    ///
    /// Feature-gated behind `content-search` — same reason as
    /// `backfillEntityEmbeddings` above.
    #[cfg(feature = "content-search")]
    #[napi]
    pub async fn backfill_fact_embeddings(
        &self,
        batch_size: u32,
    ) -> napi::Result<convert::JsEmbeddingBackfill> {
        let stats = self
            .inner
            .backfill_fact_embeddings(batch_size as usize)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("kremory backfillFactEmbeddings failed: {e}"))
            })?;
        Ok(convert::embedding_backfill_to_js(stats))
    }

}
