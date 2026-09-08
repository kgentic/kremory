import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

/**
 * Homepage narrative sections, following the source mock's alternating
 * full-width layout (.context/stitch/kremory-redesign/).
 *
 * Content rule for this file, and it is the reason several panels look
 * sparser than the mock: EVERY claim here is checked against this repo's own
 * docs, and anything that could not be checked was cut rather than softened.
 * The mock filled these same panels with invented p99 latencies, a memory
 * footprint, a cold-start figure and a competitor benchmark table; none of
 * those numbers were ever measured, so none of them appear.
 *
 * Consequence worth preserving: the comparison in `ArchitectureSection`
 * contrasts kremory with the SHAPE of a memory service (a separate process,
 * reached over a network) rather than with a named competitor carrying
 * numbers we have not verified.
 *
 * Code samples are copied from the pages that own them, so the homepage can
 * never drift into being a second, contradictory tutorial — the failure the
 * docs audit's A1/A2 findings had just fixed for `docs/intro.md`:
 *   - quickstart      docs/getting-started.md §2 "The smallest working example"
 *   - undo            docs/api.md §6a "UNDO — the unified dispatcher"
 *   - as_of           docs/api.md §"`as_of` (bi-temporal filtering)"
 *   - provider tiers  docs/api.md §2
 */

const UNDO = `// Every mutation the consolidation phase makes is logged.
let history = mem.list_mutations()
    .in_namespace(Namespace::new("agent"))
    .await?;

// And every logged mutation can be reversed by id.
if let Some(rec) = history.first() {
    let outcome = mem.undo(rec.mutation_id).execute().await?;
    println!("reversed: {outcome:?}");
}`;

const AS_OF = `use chrono::{Utc, Duration};

// Valid time: what was TRUE in the world a week ago,
// regardless of when kremory learned it.
let ctx = mem.recall("what was the policy last week?")
    .as_of(Utc::now() - Duration::days(7))
    .await?;`;

const PROVIDERS = `// Tier 1 — shortcuts for a provider you already run.
let mem = Memory::with_ollama("./agent.db").await?;
let mem = Memory::with_openai("./agent.db").await?;

// Tier 2 — anything else. Implement the traits, pass them in.
let mem = Memory::open("./agent.db")
    .with_llm(Arc::new(my_llm))
    .with_embedder(Arc::new(my_embedder))
    .await?;`;

type SectionProps = {
  readonly id: string;
  readonly label: string;
  readonly title: string;
  readonly children: ReactNode;
  readonly aside: ReactNode;
  /** Puts the aside on the left, so consecutive sections alternate. */
  readonly reversed?: boolean;
};

function Section({id, label, title, children, aside, reversed}: SectionProps) {
  return (
    <section className={styles.section} id={id}>
      <div className="container">
        <div className={styles.sectionLabel}>{label}</div>
        <div className={clsx('row', styles.sectionRow, reversed && styles.reversed)}>
          <div className={clsx('col col--5', styles.prose)}>
            <Heading as="h2" className={styles.sectionTitle}>
              {title}
            </Heading>
            {children}
          </div>
          <div className={clsx('col col--7', styles.asideCol)}>{aside}</div>
        </div>
      </div>
    </section>
  );
}

/** A bounded technical frame with a mono utility bar, per the design system. */
function Panel({title, children}: {title: string; children: ReactNode}) {
  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>{title}</div>
      <div className={styles.panelBody}>{children}</div>
    </div>
  );
}

/**
 * Side-by-side shape comparison. Deliberately non-numeric: it contrasts an
 * in-process library with a networked service, which is a structural fact,
 * not a benchmark.
 */
function ArchitectureAside() {
  const rows: ReadonlyArray<{axis: string; service: string; kremory: string}> = [
    {axis: 'Reached by', service: 'A network call', kremory: 'A function call'},
    {axis: 'To run it', service: 'A process you deploy', kremory: 'A crate you link'},
    {axis: 'Lives in', service: "Someone's cluster", kremory: 'One file on disk'},
    {axis: 'Your data', service: 'Leaves the box', kremory: 'Stays put'},
  ];
  return (
    <div className={styles.matrix}>
      <div className={clsx(styles.matrixRow, styles.matrixHead)}>
        <span className={styles.matrixAxis} />
        <span className={styles.matrixCol}>A memory service</span>
        <span className={clsx(styles.matrixCol, styles.matrixColOwn)}>kremory</span>
      </div>
      {rows.map(({axis, service, kremory}) => (
        <div className={styles.matrixRow} key={axis}>
          <span className={styles.matrixAxis}>{axis}</span>
          <span className={styles.matrixCol}>{service}</span>
          <span className={clsx(styles.matrixCol, styles.matrixColOwn)}>{kremory}</span>
        </div>
      ))}
    </div>
  );
}

export default function HomepageSections(): ReactNode {
  return (
    <>
      <Section
        id="architecture"
        label="01 — Architecture"
        title="It runs in your process, not in someone else's."
        aside={<ArchitectureAside />}>
        <p>
          kremory ships as a library, not a service: one crate, one embedded
          libSQL file, no server process to run and no API key required to ship.
          Your agent&apos;s memory is a function call away, on the same machine
          as the agent.
        </p>
        <p>
          Point the same API at a remote Turso URL later if you need to —
          nothing changes but the connection string.
        </p>
      </Section>

      <Section
        id="undo"
        label="02 — Reversibility"
        title="Memory you can undo. Deterministically."
        reversed
        aside={
          <Panel title="Reverse a consolidation">
            <CodeBlock language="rust">{UNDO}</CodeBlock>
          </Panel>
        }>
        <p>
          Agent memory that rewrites itself in the background is memory you
          cannot audit. Every merge, edit and delete kremory&apos;s
          consolidation (&quot;dream&quot;) phase makes is written to a mutation
          log first.
        </p>
        <p>
          <code>mem.undo(mutation_id)</code> reads the mutation&apos;s kind,
          dispatches to the right reversal and undoes it — no model involved, so
          the same input reverses the same way every time. Nothing kremory
          writes is a silent overwrite.
        </p>
        <p>
          <Link to="/docs/api">
            Reversibility in the API reference →
          </Link>
        </p>
      </Section>

      <Section
        id="bi-temporal"
        label="03 — Bi-temporal"
        title="Ask what was true then, not just what is true now."
        aside={
          <Panel title="Valid-time query">
            <CodeBlock language="rust">{AS_OF}</CodeBlock>
          </Panel>
        }>
        <p>
          Every fact carries two clocks: when it was true in the world, and when
          kremory recorded it. <code>.as_of(t)</code> queries the first.
        </p>
        <p>
          That distinction is what makes the history auditable. A fact learned
          today about last March shows up in a query for last March — and the
          record of when you learned it survives alongside it, rather than being
          overwritten by it.
        </p>
      </Section>

      <Section
        id="models"
        label="04 — Models"
        title="Bring your own model. We don't ship weights."
        reversed
        aside={
          <Panel title="Two tiers, one API">
            <CodeBlock language="rust">{PROVIDERS}</CodeBlock>
          </Panel>
        }>
        <p>
          kremory bundles no LLM and no embedding weights. Wire any provider —
          OpenAI, Anthropic, a local Ollama server, or your own implementation
          of the chat and embedder traits.
        </p>
        <p>
          Your API keys, your inference costs, your data. The shortcuts exist
          for providers you already run; everything else goes through the same
          builder.
        </p>
        <p>
          <Link to="/docs/api">Provider setup in the API reference →</Link>
        </p>
      </Section>

      <section className={clsx(styles.section, styles.cta)}>
        <div className="container">
          <div className={styles.ctaInner}>
            <div>
              <Heading as="h2" className={styles.sectionTitle}>
                Build your agent&apos;s memory on primitives you can inspect.
              </Heading>
              <p className={styles.ctaText}>
                One crate. One file. Every write reversible, every fact
                timestamped twice.
              </p>
              <div className={styles.ctaButtons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/getting-started">
                  Get Started
                </Link>
                <Link
                  className="button button--secondary button--lg"
                  to="/docs/comparison">
                  Compare Alternatives
                </Link>
              </div>
            </div>
            <div className={styles.ctaPanel}>
              <CodeBlock language="bash">{'cargo add kremory'}</CodeBlock>
            </div>
          </div>
        </div>
      </section>

    </>
  );
}
