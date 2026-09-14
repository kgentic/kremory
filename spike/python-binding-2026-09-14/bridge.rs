//! SPIKE A — Python-object embedder bridge -> kremory::DynEmbeddingProvider.
//!
//! NO `unsafe` is written in this file. Verified by the `forbid(unsafe_code)` below.
#![forbid(unsafe_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::anyhow;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use kremory::core::error::Result as CoreResult;
use kremory::core::intelligence::{EntityExtractor, ExtractionContext, ExtractionResult};
use kremory::core::provider::DynEmbeddingProvider;
use kremory::{CoreError, Memory, Namespace, StructuredFact};

// ---------------------------------------------------------------------------
// The bridge
// ---------------------------------------------------------------------------

/// Holds a Python callable (`async def embed(text) -> list[float]`) plus the
/// asyncio TaskLocals captured at construction time, so the coroutine can be
/// scheduled on the caller's loop even from a tokio worker thread.
struct PyEmbedderBridge {
    cb: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
    expected_dim: usize,
    calls: Arc<AtomicUsize>,
}

impl DynEmbeddingProvider for PyEmbedderBridge {
    fn embed_dyn<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = CoreResult<Vec<f32>>> + Send + 'a>> {
        let text_owned = text.to_owned();
        let expected = self.expected_dim;
        let locals = self.locals.clone();
        let calls = self.calls.clone();
        // `scope` demands a 'static future, so own everything the body needs
        // instead of borrowing `self`. `Py::clone_ref` is a refcount bump under
        // an attached GIL — not `unsafe`, and pre-registered as a PASS shape.
        let cb: Py<PyAny> = Python::attach(|py| self.cb.clone_ref(py));

        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);

            // scope() installs the asyncio TaskLocals into this tokio task so
            // `into_future` finds the running loop even if kremory awaited us
            // from a task that never had them.
            pyo3_async_runtimes::tokio::scope(locals, async move {
                // Call the Python callable -> coroutine object, convert to a
                // Send Rust future. GIL is attached only for this step; it is
                // NOT held across the await below.
                // Call the callable. If it returned an awaitable, convert to a
                // Send Rust future; if it returned a plain list, take it directly.
                // Either way the GIL is NOT held across the await below.
                enum Called {
                    Awaitable(std::pin::Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>),
                    Ready(Py<PyAny>),
                }
                let called = Python::attach(|py| -> PyResult<Called> {
                    let got = cb.call1(py, (text_owned,))?;
                    let b = got.into_bound(py);
                    if b.hasattr("__await__")? {
                        Ok(Called::Awaitable(Box::pin(
                            pyo3_async_runtimes::tokio::into_future(b)?,
                        )))
                    } else {
                        Ok(Called::Ready(b.unbind()))
                    }
                })
                .map_err(|e| CoreError::Other(anyhow!("embedder callback error: {e}")))?;

                let obj = match called {
                    Called::Awaitable(fut) => fut
                        .await
                        .map_err(|e| CoreError::Other(anyhow!("embedder callback error: {e}")))?,
                    Called::Ready(o) => o,
                };

                let f64_vec: Vec<f64> = Python::attach(|py| obj.extract::<Vec<f64>>(py))
                    .map_err(|e| CoreError::Other(anyhow!("embedder callback error: {e}")))?;

                let f32_vec: Vec<f32> = f64_vec.iter().map(|&v| v as f32).collect();

                if f32_vec.len() != expected {
                    return Err(CoreError::Other(anyhow!(
                        "embedder returned {} dims, expected {}",
                        f32_vec.len(),
                        expected
                    )));
                }
                Ok(f32_vec)
            })
            .await
        })
    }

    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

// ---------------------------------------------------------------------------
// Do-nothing extractor so no LLM is needed.
// ---------------------------------------------------------------------------

struct NoExtraction;

impl EntityExtractor for NoExtraction {
    fn name(&self) -> &'static str {
        "no-extraction"
    }
    async fn extract(
        &self,
        _text: &str,
        _ctx: &ExtractionContext<'_>,
    ) -> CoreResult<ExtractionResult> {
        Ok(ExtractionResult {
            entities: Vec::new(),
            facts: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// #[pyfunction] — full remember -> recall round trip driven by the Python cb.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn to_py(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Tag an error with the facade stage it came from, so the Python test can
/// prove WHICH call returned Err (N-A4 hinges on `remember` returning Ok).
fn stage<E: std::fmt::Display>(tag: &'static str) -> impl Fn(E) -> PyErr {
    move |e| pyo3::exceptions::PyRuntimeError::new_err(format!("[{tag}] {e}"))
}

/// Pure-Rust control embedder that always errors. Contains NO Python at all,
/// so any partial-write it produces is kremory semantics, not bridge fallout.
struct AlwaysErrEmbedder;

impl DynEmbeddingProvider for AlwaysErrEmbedder {
    fn embed_dyn<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = CoreResult<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move { Err(CoreError::Other(anyhow!("control: rust embedder always fails"))) })
    }
    fn last_usage_tokens_dyn(&self) -> Option<u64> {
        None
    }
}

/// Control: same journey, NO Python callback. Blocking (no asyncio needed).
#[pyfunction]
fn rust_control_always_err(py: Python<'_>, dim: usize, db_path: String) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let ns = Namespace::new("spike");
        let mem = Memory::open(db_path)
            .embedding_dim(dim)
            .default_namespace(ns.clone())
            .with_embedder(Arc::new(AlwaysErrEmbedder))
            .with_extractor(Arc::new(NoExtraction))
            .await
            .map_err(stage("open"))?;
        let r = mem
            .remember("Notes about Jim.")
            .in_namespace(ns.clone())
            .with_facts(vec![StructuredFact {
                subject: "jim".into(),
                predicate: "writes".into(),
                object: "Rust".into(),
                valid_from: None,
                valid_to: None,
                memory_type: None,
            }])
            .skip_extraction()
            .await;
        let remember_msg = match r {
            Ok(_) => "OK".to_string(),
            Err(e) => format!("ERR: {e}"),
        };
        mem.close().await.map_err(stage("close"))?;
        Ok(remember_msg)
    })
}

/// Returns an awaitable resolving to a dict:
///   {"rust_calls": int, "facts": [ "s p o", ... ], "rendered_len": int}
#[pyfunction]
fn round_trip(
    py: Python<'_>,
    cb: Py<PyAny>,
    dim: usize,
    db_path: String,
) -> PyResult<Bound<'_, PyAny>> {
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_out = calls.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let ns = Namespace::new("spike");
        let bridge = PyEmbedderBridge {
            cb,
            locals,
            expected_dim: dim,
            calls,
        };

        let mem = Memory::open(db_path)
            .embedding_dim(dim)
            .default_namespace(ns.clone())
            .with_embedder(Arc::new(bridge))
            .with_extractor(Arc::new(NoExtraction))
            .await
            .map_err(stage("open"))?;

        mem.remember("Notes about Jim.")
            .in_namespace(ns.clone())
            .with_facts(vec![
                StructuredFact {
                    subject: "jim".into(),
                    predicate: "writes".into(),
                    object: "Rust".into(),
                    valid_from: None,
                    valid_to: None,
                    memory_type: None,
                },
                StructuredFact {
                    subject: "jim".into(),
                    predicate: "prefers".into(),
                    object: "concise replies".into(),
                    valid_from: None,
                    valid_to: None,
                    memory_type: None,
                },
            ])
            .skip_extraction()
            .await
            .map_err(stage("remember"))?;

        let facts: Vec<String> = mem
            .recall("what do we know about jim")
            .in_namespace(ns.clone())
            .raw()
            .await
            .map_err(stage("recall_raw"))?
            .into_iter()
            .flat_map(|c| c.facts)
            .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
            .collect();

        let rendered = mem
            .recall("what do we know about jim")
            .in_namespace(ns)
            .await
            .map_err(stage("recall_rendered"))?;

        mem.close().await.map_err(stage("close"))?;

        let n = calls_out.load(Ordering::SeqCst);
        let rendered_len = rendered.len();
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let d = PyDict::new(py);
            d.set_item("rust_calls", n)?;
            d.set_item("facts", facts)?;
            d.set_item("rendered_len", rendered_len)?;
            Ok(d.into_any().unbind())
        })
    })
}

/// Minimal compile+construct proof that the bridge really is usable as
/// `Arc<dyn DynEmbeddingProvider>` (the type the facade demands).
#[pyfunction]
fn bridge_is_dyn_object(py: Python<'_>, cb: Py<PyAny>, dim: usize) -> PyResult<bool> {
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    let arc: Arc<dyn DynEmbeddingProvider> = Arc::new(PyEmbedderBridge {
        cb,
        locals,
        expected_dim: dim,
        calls: Arc::new(AtomicUsize::new(0)),
    });
    // Prove Send + Sync are satisfied without us writing `unsafe impl`.
    fn assert_send_sync<T: Send + Sync + ?Sized>(_: &T) {}
    assert_send_sync(&*arc);
    Ok(arc.last_usage_tokens_dyn().is_none())
}

/// DELIBERATELY WRONG SHAPE — blocks the calling thread on the tokio runtime
/// instead of returning an awaitable. If the caller's asyncio loop lives on this
/// thread, the Python embedder coroutine can never be scheduled. This exists to
/// MEASURE the failure mode, not to ship it.
#[pyfunction]
fn round_trip_blocking(py: Python<'_>, cb: Py<PyAny>, dim: usize, db_path: String) -> PyResult<usize> {
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_out = calls.clone();
    let rt = pyo3_async_runtimes::tokio::get_runtime();

    // Release the GIL for the duration of the blocking wait. Without this it is
    // an instant hard deadlock; with it, the question is whether the loop thread
    // is still free to run the coroutine (it is not — it is parked right here).
    py.detach(|| {
        rt.block_on(async move {
            let ns = Namespace::new("spike");
            let bridge = PyEmbedderBridge { cb, locals, expected_dim: dim, calls };
            let mem = Memory::open(db_path)
                .embedding_dim(dim)
                .default_namespace(ns.clone())
                .with_embedder(Arc::new(bridge))
                .with_extractor(Arc::new(NoExtraction))
                .await
                .map_err(stage("open"))?;
            mem.remember("Notes about Jim.")
                .in_namespace(ns.clone())
                .with_facts(vec![StructuredFact {
                    subject: "jim".into(), predicate: "writes".into(), object: "Rust".into(),
                    valid_from: None, valid_to: None, memory_type: None,
                }])
                .skip_extraction()
                .await
                .map_err(stage("remember"))?;
            let n = mem.recall("jim").in_namespace(ns).raw().await.map_err(stage("recall_raw"))?.len();
            mem.close().await.map_err(stage("close"))?;
            Ok::<usize, PyErr>(n)
        })
    })?;
    Ok(calls_out.load(Ordering::SeqCst))
}

#[pymodule]
fn kremory_py_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(bridge_is_dyn_object, m)?)?;
    m.add_function(wrap_pyfunction!(rust_control_always_err, m)?)?;
    m.add_function(wrap_pyfunction!(round_trip_blocking, m)?)?;
    Ok(())
}
