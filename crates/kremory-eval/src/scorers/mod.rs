//! Built-in scorers for the kremory eval harness.
//!
//! | Scorer               | Type            | Judge needed |
//! |----------------------|-----------------|--------------|
//! | `exact_match`        | deterministic   | no           |
//! | `f1`                 | deterministic   | no           |
//! | `model_graded_qa`    | model-graded    | yes          |
//! | `model_graded_fact`  | model-graded    | yes          |

pub mod exact_match;
pub mod f1;
pub mod model_graded_fact;
pub mod model_graded_qa;
