import type {ReactNode} from 'react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

/**
 * The stock template shipped Docusaurus's own undraw mascot illustrations
 * here. The design system explicitly rejects that register ("abandons playful
 * tech tropes … in favour of sober structural clarity") and names what
 * replaces it: functional architectural state graphs built from 1px tracks
 * and 6px circular junction nodes, with 45-degree connectors echoing the
 * diagonal geometry of the logo.
 *
 * These marks are that, drawn to the logo's own vocabulary — hairline tracks
 * in `currentColor` so they follow the colour mode, and the junction node in
 * brand gold, which is the one element the logo renders in gold too.
 */
type MarkProps = {readonly variant: 'anchor' | 'reverse' | 'pluggable'};

function NodeMark({variant}: MarkProps): ReactNode {
  const track = {
    stroke: 'currentColor',
    strokeWidth: 1,
    vectorEffect: 'non-scaling-stroke' as const,
  };
  return (
    <svg
      className={styles.mark}
      viewBox="0 0 96 96"
      fill="none"
      role="img"
      aria-hidden="true">
      {variant === 'anchor' && (
        <>
          {/* A single node held by four orthogonal tracks — one file, one place. */}
          <path d="M8 48h26M62 48h26M48 8v26M48 62v26" {...track} />
          <rect x="20" y="20" width="56" height="56" rx="3" {...track} />
        </>
      )}
      {variant === 'reverse' && (
        <>
          {/* A branch that returns: 45-degree connectors out and back. */}
          <path d="M8 72h20l20-20 20-20h20" {...track} />
          <path d="M68 32l-14 14M68 32l14 14" {...track} />
          <path d="M8 72v-14" {...track} />
        </>
      )}
      {variant === 'pluggable' && (
        <>
          {/* Three interchangeable inbound tracks converging on one node. */}
          <path d="M8 24h24l16 24M8 48h40M8 72h24l16-24" {...track} />
          <path d="M48 48h40" {...track} />
        </>
      )}
      <circle cx="48" cy="48" r="6" fill="var(--krem-gold)" />
      <circle cx="48" cy="48" r="10" {...track} />
    </svg>
  );
}

type FeatureItem = {
  index: string;
  title: string;
  variant: MarkProps['variant'];
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    index: '01',
    title: 'Local-first',
    variant: 'anchor',
    description: (
      <>
        Runs entirely on your machine — no server, no API key, no data leaving
        the box. Point the same API at a remote Turso URL later if you need
        to; nothing changes but the connection string.
      </>
    ),
  },
  {
    index: '02',
    title: 'Memory you can undo',
    variant: 'reverse',
    description: (
      <>
        Every merge, edit, and delete kremory&apos;s background consolidation
        (&quot;dream&quot;) phase makes is logged and reversible —{' '}
        <code>mem.undo(mutation_id)</code> reverses it, deterministically, no
        LLM involved.
      </>
    ),
  },
  {
    index: '03',
    title: 'Bring your own model',
    variant: 'pluggable',
    description: (
      <>
        kremory bundles no LLM or embedding weights. Wire any provider —
        OpenAI, Ollama, local GGUF, sentence-transformers via HTTP — your API
        keys, your inference costs, your data.
      </>
    ),
  },
];

function Feature({index, title, variant, description}: FeatureItem) {
  return (
    <div className="col col--4">
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardIndex}>{index}</span>
          <NodeMark variant={variant} />
        </div>
        <div className={styles.cardBody}>
          <Heading as="h3" className={styles.cardTitle}>
            {title}
          </Heading>
          <p className={styles.cardText}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionLabel}>Why kremory</div>
        <div className="row">
          {FeatureList.map((props) => (
            <Feature key={props.index} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
