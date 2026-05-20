use std::sync::Arc;

use metrics::{counter, histogram};
use std::time::Instant;

use crate::error::Result;
use crate::extraction::{PromptVersion, SingleCallExtractor};
use crate::grounding::{GroundingChecker, TokenOverlapGroundingChecker};
use crate::intelligence::{EntityExtractor, ExtractedEntity, ExtractionContext, ExtractionResult};
use crate::provider::{chat_msg_system, chat_msg_user, ChatProvider};
use crate::resolver::normalize_name;
use crate::text_utils::OovAuditor;

#[path = "hybrid_extractor_helpers.rs"]
mod helpers;
#[path = "hybrid_extractor_parsing.rs"]
mod parsing;
#[path = "hybrid_extractor_prompts.rs"]
mod prompts;

use helpers::{filter_new_entities, merge_entities_with_grounding};
use parsing::{parse_entity_list_response, parse_typed_orphans_response};
use prompts::{build_gleaning_prompt, build_typing_prompt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionConfig {
    pub gleaning_rounds: u32,
    pub single_call_max_tokens: usize,
    pub typing_max_tokens: usize,
    pub max_oov_orphans: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            gleaning_rounds: 0,
            single_call_max_tokens: 512,
            typing_max_tokens: 256,
            max_oov_orphans: 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionMetadata {
    pub extractor: &'static str,
    pub prompt_version: PromptVersion,
    pub llm_calls: u32,
    pub gleaning_rounds: u32,
    pub used_oov_audit: bool,
    pub used_typing_call: bool,
}

pub struct HybridExtractor<L: ChatProvider> {
    llm: Arc<L>,
    config: ExtractionConfig,
    prompt_version: PromptVersion,
    auditor: Option<Arc<OovAuditor>>,
    grounding_checker: Arc<dyn GroundingChecker>,
}

impl<L: ChatProvider> HybridExtractor<L> {
    pub fn new(llm: Arc<L>) -> Self {
        Self {
            llm,
            config: ExtractionConfig::default(),
            prompt_version: PromptVersion::default(),
            auditor: None,
            grounding_checker: Arc::new(TokenOverlapGroundingChecker::default()),
        }
    }

    pub fn with_config(mut self, config: ExtractionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_prompt_version(mut self, version: PromptVersion) -> Self {
        self.prompt_version = version;
        self
    }

    pub fn with_auditor(mut self, auditor: Arc<OovAuditor>) -> Self {
        self.auditor = Some(auditor);
        self
    }

    pub fn with_grounding_checker(mut self, grounding_checker: Arc<dyn GroundingChecker>) -> Self {
        self.grounding_checker = grounding_checker;
        self
    }

    pub fn config(&self) -> &ExtractionConfig {
        &self.config
    }

    pub fn metadata(&self) -> ExtractionMetadata {
        ExtractionMetadata {
            extractor: "hybrid_extractor",
            prompt_version: self.prompt_version,
            llm_calls: 1 + self.config.gleaning_rounds,
            gleaning_rounds: self.config.gleaning_rounds,
            used_oov_audit: self.auditor.is_some(),
            used_typing_call: self.auditor.is_some(),
        }
    }

    pub async fn stage1_singlecall<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        SingleCallExtractor::new(Arc::clone(&self.llm))
            .with_prompt_version(self.prompt_version)
            .extract(text, ctx)
            .await
    }

    async fn stage2_gleaning<'a>(
        &'a self,
        text: &'a str,
        base_entities: &[ExtractedEntity],
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<Vec<ExtractedEntity>> {
        if self.config.gleaning_rounds == 0 {
            return Ok(Vec::new());
        }

        let mut discovered = base_entities.to_vec();
        let mut additive = Vec::new();

        for _ in 0..self.config.gleaning_rounds {
            let prompt = build_gleaning_prompt(text, &discovered, ctx);
            let start = Instant::now();
            let gleaning_msgs = vec![
                chat_msg_system("You are a knowledge graph extraction system. Output valid JSON only."),
                chat_msg_user(prompt),
            ];
            let response = self
                .llm
                .chat_with_tools(&gleaning_msgs, None, None)
                .await
                .map_err(|e| crate::error::RqlError::Llm(e.to_string()))?;
            histogram!("rql.extraction.stage_ms", "stage" => "hybrid_gleaning")
                .record(start.elapsed().as_secs_f64() * 1000.0);
            let response_text = response.text().unwrap_or_default();

            let gleaned = parse_entity_list_response(&response_text)?;
            let new_entities = filter_new_entities(gleaned, &discovered);
            if new_entities.is_empty() {
                break;
            }

            counter!("rql.extraction.gleaning_entity_adds").increment(new_entities.len() as u64);
            discovered.extend(new_entities.clone());
            additive.extend(new_entities);
        }

        Ok(additive)
    }

    fn stage3_oov_audit(&self, text: &str, entities: &[ExtractedEntity]) -> Vec<String> {
        let Some(auditor) = &self.auditor else {
            return Vec::new();
        };

        let audit_adds = auditor.audit(text, entities);
        let mut seen = std::collections::HashSet::new();
        let mut orphans = Vec::new();

        for entity in audit_adds {
            let normalized = normalize_name(&entity.name);
            if seen.insert(normalized) {
                orphans.push(entity.name);
            }
            if orphans.len() >= self.config.max_oov_orphans {
                break;
            }
        }

        counter!("rql.extraction.oov_orphan_count").increment(orphans.len() as u64);
        orphans
    }

    async fn stage4_typing<'a>(
        &'a self,
        text: &'a str,
        orphans: &[String],
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<Vec<ExtractedEntity>> {
        if orphans.is_empty() {
            return Ok(Vec::new());
        }

        let prompt = build_typing_prompt(text, orphans, ctx);
        let start = Instant::now();
        let typing_msgs = vec![
            chat_msg_system("You are a knowledge graph extraction system. Output valid JSON only."),
            chat_msg_user(prompt),
        ];
        let response = self
            .llm
            .chat_with_tools(&typing_msgs, None, None)
            .await
            .map_err(|e| crate::error::RqlError::Llm(e.to_string()))?;
        histogram!("rql.extraction.stage_ms", "stage" => "hybrid_typing")
            .record(start.elapsed().as_secs_f64() * 1000.0);
        let response_text = response.text().unwrap_or_default();

        let typed = parse_typed_orphans_response(&response_text, orphans)?;
        counter!("rql.extraction.oov_typed_adds").increment(typed.len() as u64);
        Ok(typed)
    }

    pub fn stage5_merge(
        &self,
        base: Vec<ExtractedEntity>,
        additive: Vec<ExtractedEntity>,
        source_text: &str,
    ) -> Vec<ExtractedEntity> {
        merge_entities_with_grounding(base, additive, source_text, self.grounding_checker.as_ref())
    }
}

impl<L: ChatProvider> EntityExtractor for HybridExtractor<L> {
    async fn extract<'a>(
        &'a self,
        text: &'a str,
        ctx: &'a ExtractionContext<'a>,
    ) -> Result<ExtractionResult> {
        let stage1 = self.stage1_singlecall(text, ctx).await?;
        let gleaned = self.stage2_gleaning(text, &stage1.entities, ctx).await?;

        let mut base_entities = stage1.entities.clone();
        base_entities.extend(gleaned);
        base_entities.sort_by_key(|e| normalize_name(&e.name));
        base_entities.dedup_by(|a, b| normalize_name(&a.name) == normalize_name(&b.name));

        let orphans = self.stage3_oov_audit(text, &base_entities);
        let typed = self.stage4_typing(text, &orphans, ctx).await?;
        let merged_entities = self.stage5_merge(base_entities, typed, text);

        Ok(ExtractionResult {
            entities: merged_entities,
            facts: stage1.facts,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::Value;

    use super::{ExtractionConfig, HybridExtractor};
    use crate::config::ContentType;
    use crate::extraction::PromptVersion;
    use crate::grounding::TokenOverlapGroundingChecker;
    use crate::intelligence::{EntityExtractor, ExtractedEntity, ExtractionContext};
    use crate::provider::MockChatProvider;
    use crate::text_utils::OovAuditor;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn extraction_config_defaults_match_plan() {
        let config = ExtractionConfig::default();
        assert_eq!(config.gleaning_rounds, 0);
        assert_eq!(config.single_call_max_tokens, 512);
        assert_eq!(config.typing_max_tokens, 256);
        assert_eq!(config.max_oov_orphans, 20);
    }

    #[test]
    fn stage5_merge_dedups_by_normalized_name() {
        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(HashMap::new())));
        let merged = extractor.stage5_merge(
            vec![ExtractedEntity {
                name: "Acme Corp".to_string(),
                label: "Organisation".to_string(),
                properties: Value::Null,
            }],
            vec![ExtractedEntity {
                name: "acme corp".to_string(),
                label: "Organisation".to_string(),
                properties: Value::Null,
            }],
            "Acme Corp shipped a product.",
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "Acme Corp");
    }

    #[test]
    fn stage5_merge_sets_grounded_property() {
        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(HashMap::new())))
            .with_grounding_checker(Arc::new(TokenOverlapGroundingChecker::default()));

        let merged = extractor.stage5_merge(
            vec![ExtractedEntity {
                name: "Alice".to_string(),
                label: "Person".to_string(),
                properties: Value::Null,
            }],
            vec![],
            "Alice spoke first.",
        );

        assert_eq!(merged[0].properties["grounded"], Value::Bool(true));
    }

    #[test]
    fn metadata_tracks_stage1_only_shape() {
        let metadata = HybridExtractor::new(Arc::new(MockChatProvider::new(HashMap::new())))
            .with_config(ExtractionConfig {
                gleaning_rounds: 1,
                ..ExtractionConfig::default()
            })
            .with_prompt_version(PromptVersion::V1Rules)
            .metadata();

        assert_eq!(metadata.extractor, "hybrid_extractor");
        assert_eq!(metadata.prompt_version, PromptVersion::V1Rules);
        assert_eq!(metadata.llm_calls, 2);
        assert_eq!(metadata.gleaning_rounds, 1);
        assert!(!metadata.used_oov_audit);
    }

    #[test]
    fn hybrid_stage1_matches_single_call_behavior() {
        let mut responses = HashMap::new();
        responses.insert(
            "Extract all entities and relationships from the text.".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"Acme Corp","label":"Organisation"}],"relationships":[{"subject":"Alice","predicate":"works_at","object":"Acme Corp"}]}"#.to_string(),
        );
        let llm = Arc::new(MockChatProvider::new(responses));
        let extractor = HybridExtractor::new(llm).with_prompt_version(PromptVersion::V2SchemaLight);

        let result =
            block_on(extractor.extract("Alice works at Acme Corp.", &ExtractionContext::default()))
                .unwrap();

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.facts.len(), 1);
        let names: Vec<&str> = result
            .entities
            .iter()
            .map(|entity| entity.name.as_str())
            .collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Acme Corp"));
        assert_eq!(result.facts[0].predicate, "works_at");
    }

    #[test]
    fn stage2_gleaning_returns_only_new_entities() {
        let mut responses = HashMap::new();
        responses.insert(
            "Review the text and return only entities that were missed".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"},{"name":"Mercury Bank","label":"Organisation"}]}"#.to_string(),
        );
        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(responses))).with_config(
            ExtractionConfig {
                gleaning_rounds: 1,
                ..ExtractionConfig::default()
            },
        );
        let base_entities = vec![ExtractedEntity {
            name: "Alice".to_string(),
            label: "Person".to_string(),
            properties: Value::Null,
        }];
        let ctx = ExtractionContext {
            content_type: ContentType::Text,
            ..ExtractionContext::default()
        };

        let gleaned = block_on(extractor.stage2_gleaning(
            "Alice mentioned Mercury Bank.",
            &base_entities,
            &ctx,
        ))
        .unwrap();

        assert_eq!(gleaned.len(), 1);
        assert_eq!(gleaned[0].name, "Mercury Bank");
    }

    #[test]
    fn stage3_oov_audit_returns_orphan_names() {
        let auditor = make_test_auditor();
        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(HashMap::new())))
            .with_auditor(Arc::new(auditor));

        let orphans = extractor.stage3_oov_audit(
            "Alice discussed Zyntriq with the team.",
            &[ExtractedEntity {
                name: "Alice".to_string(),
                label: "Person".to_string(),
                properties: Value::Null,
            }],
        );

        assert!(orphans.iter().any(|name| name == "Zyntriq"));
    }

    #[test]
    fn stage4_typing_falls_back_to_entity_for_missing_candidate() {
        let mut responses = HashMap::new();
        responses.insert(
            "Type every candidate entity using the source text for context.".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"}]}"#.to_string(),
        );
        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(responses)));
        let ctx = ExtractionContext::default();

        let typed = block_on(extractor.stage4_typing(
            "Alice mentioned Zyntriq.",
            &["Alice".to_string(), "Zyntriq".to_string()],
            &ctx,
        ))
        .unwrap();

        assert_eq!(typed.len(), 2);
        assert_eq!(typed[0].label, "Person");
        assert_eq!(typed[1].name, "Zyntriq");
        assert_eq!(typed[1].label, "Entity");
    }

    #[test]
    fn full_pipeline_merges_stage_outputs_and_preserves_stage1_facts() {
        let mut responses = HashMap::new();
        responses.insert(
            "Extract all entities and relationships from the text.".to_string(),
            r#"{"entities":[{"name":"Alice","label":"Person"}],"relationships":[{"subject":"Alice","predicate":"works_at","object":"Acme Corp"}]}"#.to_string(),
        );
        responses.insert(
            "Review the text and return only entities that were missed".to_string(),
            r#"{"entities":[{"name":"Mercury Bank","label":"Organisation"}]}"#.to_string(),
        );
        responses.insert(
            "Type every candidate entity using the source text for context.".to_string(),
            r#"{"entities":[{"name":"Zyntriq","label":"Product"}]}"#.to_string(),
        );

        let extractor = HybridExtractor::new(Arc::new(MockChatProvider::new(responses)))
            .with_config(ExtractionConfig {
                gleaning_rounds: 1,
                ..ExtractionConfig::default()
            })
            .with_auditor(Arc::new(make_test_auditor()));

        let result = block_on(extractor.extract(
            "Alice works at Acme Corp and mentioned Mercury Bank and Zyntriq.",
            &ExtractionContext {
                content_type: ContentType::Text,
                ..ExtractionContext::default()
            },
        ))
        .unwrap();

        let names: Vec<String> = result
            .entities
            .iter()
            .map(|entity| entity.name.clone())
            .collect();
        assert!(names.iter().any(|name| name == "Alice"));
        assert!(names.iter().any(|name| name == "Mercury Bank"));
        assert!(names.iter().any(|name| name == "Zyntriq"));
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].predicate, "works_at");
    }

    fn make_test_auditor() -> OovAuditor {
        let aff = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.aff"
        ))
        .expect("en_US.aff");
        let dic = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/dictionaries/en_US.dic"
        ))
        .expect("en_US.dic");
        let dict = zspell::builder()
            .config_str(&aff)
            .dict_str(&dic)
            .build()
            .expect("build dictionary");
        let stop_words = stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(|word| word.to_string())
            .collect();
        OovAuditor::new(dict, stop_words)
    }
}
