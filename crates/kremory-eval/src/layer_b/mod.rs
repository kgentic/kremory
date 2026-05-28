//! Layer B — diagnostic metrics (Day 3 + Day 4).
//!
//! | Module              | Metric family                                       | Judge needed |
//! |---------------------|-----------------------------------------------------|--------------|
//! | `entity_extraction` | Entity precision / recall / F1 (deterministic)      | no           |
//! | `ragas`             | RAGAS 6 metrics (Faithfulness, Relevancy, etc.)     | yes          |
//! | `graph_integrity`   | Programmatic DB invariants                          | no           |
//! | `contradiction`     | G-Eval contradiction detection                      | yes          |
//! | `temporal`          | G-Eval temporal correctness                         | yes          |

pub mod contradiction;
pub mod entity_extraction;
pub mod graph_integrity;
pub mod ragas;
pub mod temporal;
