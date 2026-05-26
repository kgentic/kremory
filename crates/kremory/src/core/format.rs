//! Binary format constants and validation for kremory snapshot files.
//!
//! All kremory-produced binary blobs (export snapshots, embedded vector stores)
//! begin with a fixed 4-byte magic number followed by a 1-byte format version
//! byte. A `REBUILD_HINT` flag byte at position 5 signals whether the consumer
//! should discard any cached derived data (community clusters, speculative cache)
//! and rebuild from the source graph.
//!
//! Stories #151, #152, #164.

/// 4-byte magic prefix for kremory binary snapshot files. Story #151.
///
/// ASCII encoding of `"KMRY"`. Every kremory-produced binary blob begins with
/// these 4 bytes; any blob that does NOT start with them is rejected before
/// further parsing.
pub const KREMORY_MAGIC: [u8; 4] = *b"KMRY";

/// Current binary format version byte. Story #152.
///
/// Increment when the binary layout changes in an incompatible way.
/// Version 1 = initial kremory v0.1.0 layout.
pub const FORMAT_VERSION: u8 = 1;

/// Rebuild-hint flag byte. Story #152.
///
/// When present at byte position 5 of a kremory snapshot (value `0x01`),
/// instructs the consumer to invalidate and rebuild all derived data
/// (speculative cache, community clusters) before using the snapshot.
/// Value `0x00` = no rebuild required (incremental update).
pub const REBUILD_HINT: u8 = 0x01;

/// Minimum valid snapshot header length: magic (4) + version (1) + hint (1).
const MIN_HEADER_LEN: usize = 6;

/// Reason a binary blob was rejected by `validate_snapshot_header`. Story #164.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptReason {
    /// Blob is shorter than the minimum header size.
    TooShort { actual: usize, minimum: usize },
    /// Magic bytes do not match `KREMORY_MAGIC`.
    BadMagic { actual: [u8; 4] },
    /// Format version byte is not recognised by this build.
    UnknownVersion { version: u8 },
}

impl std::fmt::Display for CorruptReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { actual, minimum } => {
                write!(f, "blob too short: {actual} < {minimum} bytes")
            }
            Self::BadMagic { actual } => {
                write!(
                    f,
                    "bad magic: expected {:?}, got {:?}",
                    KREMORY_MAGIC, actual
                )
            }
            Self::UnknownVersion { version } => {
                write!(
                    f,
                    "unknown format version 0x{version:02x}; max supported = 0x{FORMAT_VERSION:02x}"
                )
            }
        }
    }
}

/// Validate the 6-byte header of a kremory snapshot blob. Story #164.
///
/// Returns `Ok(rebuild_hint)` where `rebuild_hint` is `true` when byte 5
/// equals `REBUILD_HINT`. Returns `Err(CorruptReason)` on any structural
/// violation.
///
/// This function is deliberately allocation-free (no `String`, no `Vec`)
/// so it can run in hot paths (e.g. before seeking into a large file).
pub fn validate_snapshot_header(data: &[u8]) -> Result<bool, CorruptReason> {
    if data.len() < MIN_HEADER_LEN {
        return Err(CorruptReason::TooShort {
            actual: data.len(),
            minimum: MIN_HEADER_LEN,
        });
    }
    // SAFETY: we checked `data.len() >= MIN_HEADER_LEN (6)` above, so
    // `data[0..4]` is always exactly 4 bytes. The array copy never panics.
    let magic: [u8; 4] = [data[0], data[1], data[2], data[3]];
    if magic != KREMORY_MAGIC {
        return Err(CorruptReason::BadMagic { actual: magic });
    }
    let version = data[4];
    if version > FORMAT_VERSION {
        return Err(CorruptReason::UnknownVersion { version });
    }
    let rebuild = data[5] == REBUILD_HINT;
    Ok(rebuild)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(magic: &[u8; 4], version: u8, hint: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(6);
        v.extend_from_slice(magic);
        v.push(version);
        v.push(hint);
        v
    }

    #[test]
    fn valid_header_no_rebuild() {
        let hdr = make_header(&KREMORY_MAGIC, FORMAT_VERSION, 0x00);
        let result = validate_snapshot_header(&hdr);
        assert_eq!(result, Ok(false));
    }

    #[test]
    fn valid_header_with_rebuild() {
        let hdr = make_header(&KREMORY_MAGIC, FORMAT_VERSION, REBUILD_HINT);
        let result = validate_snapshot_header(&hdr);
        assert_eq!(result, Ok(true));
    }

    #[test]
    fn too_short_rejected() {
        let short = [b'K', b'M', b'R'];
        assert_eq!(
            validate_snapshot_header(&short),
            Err(CorruptReason::TooShort {
                actual: 3,
                minimum: 6
            })
        );
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(
            validate_snapshot_header(&[]),
            Err(CorruptReason::TooShort {
                actual: 0,
                minimum: 6
            })
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let hdr = make_header(b"NOPE", FORMAT_VERSION, 0x00);
        assert_eq!(
            validate_snapshot_header(&hdr),
            Err(CorruptReason::BadMagic { actual: *b"NOPE" })
        );
    }

    #[test]
    fn unknown_version_rejected() {
        let hdr = make_header(&KREMORY_MAGIC, FORMAT_VERSION + 1, 0x00);
        assert_eq!(
            validate_snapshot_header(&hdr),
            Err(CorruptReason::UnknownVersion {
                version: FORMAT_VERSION + 1
            })
        );
    }

    #[test]
    fn kremory_magic_constant_is_kmry() {
        assert_eq!(&KREMORY_MAGIC, b"KMRY");
    }

    #[test]
    fn version_1_is_current() {
        assert_eq!(FORMAT_VERSION, 1);
    }
}
