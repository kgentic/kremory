import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

/**
 * The specification plate deliberately carries only structural facts that
 * cannot drift: no version number, no benchmark figure, no star count. The
 * design mock this page is styled from filled the equivalent panel with
 * invented metrics (p99 latencies, footprint, competitor timings); none of
 * them were measured, so none of them are reproduced here.
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
          <div className="col col--7">
            <span className={styles.eyebrow}>
              <span className={styles.eyebrowNode} aria-hidden="true" />
              Agent memory, local-first
            </span>
            <Heading as="h1" className={styles.heroTitle}>
              {siteConfig.title}
            </Heading>
            <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
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
          <div className="col col--5">
            <div className={styles.specPlate}>
              <div className={styles.specPlateHeader}>
                <span>Specification</span>
              </div>
              {SPEC_ROWS.map(({key, value}) => (
                <div className={styles.specRow} key={key}>
                  <span className={styles.specKey}>{key}</span>
                  <span className={styles.specValue}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
