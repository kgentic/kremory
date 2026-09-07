import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Local-first',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Runs entirely on your machine — no server, no API key, no data leaving
        the box. Point the same API at a remote Turso URL later if you need
        to; nothing changes but the connection string.
      </>
    ),
  },
  {
    title: 'Memory you can undo',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
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
    title: 'Bring your own model',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        kremory bundles no LLM or embedding weights. Wire any provider —
        OpenAI, Ollama, local GGUF, sentence-transformers via HTTP — your API
        keys, your inference costs, your data.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
