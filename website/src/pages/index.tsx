import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import HomepageSections from '@site/src/components/HomepageSections';
import Heading from '@theme/Heading';

import styles from './index.module.css';

/**
 * The canonical first program, kept byte-identical to
 * `docs/getting-started.md` §2 apart from dropping that page's inline
 * commentary. The docs audit's A1/A2 findings were exactly this failure in
 * the other direction — a landing page carrying a DIFFERENT, more elaborate
 * "smallest example" than the tutorial — so there is one example on this
 * site and this is it. Change it here only by changing it there too.
 */
const QUICKSTART = `use kremory::{Memory, Namespace};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mem = Memory::with_ollama("./agent.db").await?;
    let ns = Namespace::new("agent");

    mem.remember("User prefers concise replies")
        .in_namespace(ns.clone())
        .await?;

    let context: String = mem
        .recall("what does the user prefer?")
        .in_namespace(ns)
        .await?;

    println!("{context}");
    Ok(())
}`;

/**
 * Structural facts only — nothing that drifts with a release, and nothing
 * measured. No version badge, no star count, no latency figure: the source
 * mock had all three and every one was invented.
 */
const SPEC_ROWS: ReadonlyArray<{key: string; value: string}> = [
  {key: 'Runtime', value: 'Rust, embedded in your process'},
  {key: 'Storage', value: 'One file, libSQL / SQLite'},
  {key: 'Models', value: 'Bring your own — no bundled weights'},
  {key: 'Licence', value: 'Apache-2.0'},
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className="row">
          <div className="col col--5">
            <span className={styles.eyebrow}>
              <span className={styles.eyebrowNode} aria-hidden="true" />
              Agent memory, local-first
            </span>
            <Heading as="h1" className={styles.heroTitle}>
              {siteConfig.title}
            </Heading>
            <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
            <p className={styles.heroBody}>
              Embeddable, bi-temporal knowledge-graph memory for AI agents — a
              single Rust crate you link in, not a service you call out to.
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/getting-started">
                Get Started
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/api">
                API Reference
              </Link>
            </div>
          </div>
          <div className="col col--7">
            <div className={styles.heroPanel}>
              <div className={styles.heroPanelHeader}>
                <span>main.rs</span>
                <span className={styles.heroPanelNote}>
                  the whole program
                </span>
              </div>
              <CodeBlock language="rust">{QUICKSTART}</CodeBlock>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function SpecStrip() {
  return (
    <section className={styles.specStrip}>
      <div className="container">
        <div className={styles.specRow}>
          {SPEC_ROWS.map(({key, value}) => (
            <div className={styles.specCell} key={key}>
              <span className={styles.specKey}>{key}</span>
              <span className={styles.specValue}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <SpecStrip />
        <HomepageSections />
      </main>
    </Layout>
  );
}
