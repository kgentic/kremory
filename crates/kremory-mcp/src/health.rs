//! Boot-time reachability check for the Ollama endpoint the kremory-mcp
//! binaries (`main.rs` for the MCP stdio server, `bin/kremory-http.rs` for
//! the REST bin) wire `kremory::Memory` against.
//!
//! `kremory::facade::providers::with_ollama_at` / `with_ollama_at_model` do
//! NOT probe reachability at construction time — the doc comment on
//! `with_ollama` is explicit: "If no Ollama is available at runtime, the
//! first `.remember()` / `.recall()` call returns a network error." That is
//! too late for a server binary — per CLAUDE.md's "fail fast and loud" rule,
//! every binary must know at boot, not on the first tool call. This module is
//! a connectivity-only check (TCP connect to the parsed host:port) — it does
//! NOT verify Ollama is actually serving the HTTP API or that the configured
//! model is pulled; it only rules out "nothing is listening there".
//!
//! `pub` (not `pub(crate)`) because both binaries live in separate crate
//! targets (`[[bin]]` targets are distinct crates from the `[lib]` target
//! even within the same package) and reach this module via
//! `kremory_mcp::health::...`.

use std::net::ToSocketAddrs as _;
use std::time::Duration;

/// Parse `host:port` out of an `http(s)://host:port[/path]` URL without
/// pulling in a full URL-parsing crate (kremory-mcp has no other need for
/// one). Returns `None` for shapes this can't confidently parse — callers
/// should treat that as a configuration error, not silently skip the check.
pub fn parse_host_port(url: &str) -> Option<(String, u16)> {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let host_port = without_scheme
        .split(['/', '?'])
        .next()
        .unwrap_or(without_scheme);
    if host_port.is_empty() {
        return None;
    }
    let (host, port_str) = host_port.rsplit_once(':')?;
    if host.is_empty() {
        return None;
    }
    let port: u16 = port_str.parse().ok()?;
    Some((host.to_string(), port))
}

/// TCP-connect to `url`'s host:port with a bounded timeout. `Ok(())` means
/// something is listening; `Err` carries a human-readable reason suitable
/// for a fail-loud boot error.
pub async fn check_reachable(url: &str, timeout: Duration) -> Result<(), String> {
    let (host, port) = parse_host_port(url)
        .ok_or_else(|| format!("could not parse host:port from url {url:?}"))?;

    // `(host, port)` implements `ToSocketAddrs` via std, which `tokio::net`
    // reuses — resolve first so a DNS failure gets its own clear message
    // rather than being folded into the connect-timeout branch.
    let addrs = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| format!("could not resolve {host}:{port}: {e}"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        return Err(format!("{host}:{port} resolved to no addresses"));
    }

    match tokio::time::timeout(
        timeout,
        tokio::net::TcpStream::connect((host.as_str(), port)),
    )
    .await
    {
        Ok(Ok(_stream)) => Ok(()),
        Ok(Err(e)) => Err(format!("TCP connect to {host}:{port} failed: {e}")),
        Err(_) => Err(format!(
            "TCP connect to {host}:{port} timed out after {timeout:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_url_with_port() {
        assert_eq!(
            parse_host_port("http://localhost:11434"),
            Some(("localhost".to_string(), 11434))
        );
    }

    #[test]
    fn parses_https_url_with_path() {
        assert_eq!(
            parse_host_port("https://ollama.internal:443/api/tags"),
            Some(("ollama.internal".to_string(), 443))
        );
    }

    #[test]
    fn rejects_url_without_port() {
        assert_eq!(parse_host_port("http://localhost"), None);
    }

    #[test]
    fn rejects_empty_string() {
        assert_eq!(parse_host_port(""), None);
    }

    #[tokio::test]
    async fn check_reachable_fails_fast_on_closed_port() {
        // Port 1 is a reserved/unlikely-to-be-listening port on loopback —
        // connect should be refused near-instantly, well inside the timeout.
        let result = check_reachable("http://127.0.0.1:1", Duration::from_secs(2)).await;
        assert!(result.is_err(), "expected connect to a closed port to fail");
    }

    #[tokio::test]
    async fn check_reachable_succeeds_against_a_real_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("listener has a local addr");
        // Accept in the background so the connecting side completes its
        // handshake instead of blocking on a backlog with nothing accepting.
        tokio::spawn(async move {
            let _ = listener.accept().await;
        });
        let url = format!("http://{}:{}", addr.ip(), addr.port());
        let result = check_reachable(&url, Duration::from_secs(2)).await;
        assert!(
            result.is_ok(),
            "expected connect to a real listener to succeed: {result:?}"
        );
    }
}
