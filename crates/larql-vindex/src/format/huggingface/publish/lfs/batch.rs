//! LFS batch endpoint — request signed upload + verify URLs for one
//! object, parse the response into actions our caller dispatches on.

use std::collections::HashMap;

use crate::error::VindexError;

use super::super::hf_repo_url;
use super::super::protocol::{
    CONTENT_TYPE_LFS_JSON, HASH_ALGO_SHA256, LFS_MULTIPART_CHUNK_SIZE_HEADER, LFS_OP_UPLOAD,
    LFS_OP_VERIFY, LFS_TRANSFER_BASIC, LFS_TRANSFER_MULTIPART,
};
use super::{LfsAction, LfsBatchResponse, UploadAction};

/// POST to the LFS batch endpoint asking for an upload URL for one
/// object. Returns the upload + verify actions.
///
/// `actions.upload` may be absent (object already stored on the hub),
/// or it may be either a single-PUT (`UploadAction::Single`) or
/// multipart (`UploadAction::Multipart`) shape. We declare both
/// `basic` and `multipart` capabilities in the batch request so HF
/// chooses the right shape based on `size`: files ≤5 GB come back as
/// `basic` (single signed PUT), files >5 GB come back as `multipart`
/// (chunk_size + numbered part URLs). Without declaring `multipart`,
/// HF returns `400 "You need to configure your repository to enable
/// upload of files > 5GB"` for any object that exceeds the basic
/// adapter's single-PUT ceiling.
pub(super) fn lfs_batch_upload(
    repo_id: &str,
    token: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<LfsBatchResponse, VindexError> {
    let url = format!(
        "{}.git/info/lfs/objects/batch",
        hf_repo_url(repo_type, repo_id)
    );
    let body = serde_json::json!({
        "operation":  LFS_OP_UPLOAD,
        "transfers":  [LFS_TRANSFER_BASIC, LFS_TRANSFER_MULTIPART],
        "hash_algo":  HASH_ALGO_SHA256,
        "objects":    [{"oid": sha256, "size": size}],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON)
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS batch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS batch ({status}): {body}")));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("LFS batch JSON: {e}")))?;
    parse_lfs_batch_response(&json)
}

/// Parse the JSON body of an LFS batch response into the upload/verify
/// actions our caller dispatches on. Pure helper so the JSON contract
/// can be unit-tested without an HTTP server.
///
/// `actions.upload` shape detection:
/// - **Multipart** when `header.chunk_size` exists. Numbered keys
///   (`00001`, `00002`, …) hold pre-signed S3 PUT URLs for each part;
///   sorted by integer value of the key, they form the ordered parts
///   list. `upload.href` is the completion endpoint.
/// - **Single-PUT** otherwise. `upload.href` is the single PUT URL;
///   `upload.header` is the request-headers map (S3 sigv4 etc.).
///
/// Matches the Python `huggingface_hub._commit_api._upload_multi_part`
/// detection rule: `chunk_size` presence is the signal, not the
/// number of numbered keys (HF may include extra header keys we need
/// to ignore).
pub(super) fn parse_lfs_batch_response(
    json: &serde_json::Value,
) -> Result<LfsBatchResponse, VindexError> {
    let objects = json
        .get("objects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("LFS batch response missing `objects`".into()))?;
    let obj = objects
        .first()
        .ok_or_else(|| VindexError::Parse("LFS batch objects[] empty".into()))?;

    if let Some(err) = obj.get("error") {
        return Err(VindexError::Parse(format!("LFS batch object error: {err}")));
    }

    let actions = obj.get("actions");
    let parse_lfs_action = |key: &str| -> Option<LfsAction> {
        let a = actions?.get(key)?;
        let href = a.get("href").and_then(|v| v.as_str())?.to_string();
        let header: HashMap<String, String> = a
            .get("header")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        Some(LfsAction { href, header })
    };

    let upload = parse_lfs_action(LFS_OP_UPLOAD).map(promote_upload_action);
    let verify = parse_lfs_action(LFS_OP_VERIFY);
    Ok(LfsBatchResponse { upload, verify })
}

/// Promote a generic `LfsAction` into the right `UploadAction` variant
/// based on whether `header.chunk_size` is present. If the header
/// indicates multipart but the chunk_size or part URLs are malformed,
/// fall back to `Single` — the upstream caller will then attempt the
/// single-PUT path; if HF then rejects (because the object is
/// >5 GB) the user gets the same error they had before this fix and
/// can re-extract / file a bug. Better than silently corrupting an
/// upload on a malformed response.
fn promote_upload_action(action: LfsAction) -> UploadAction {
    let chunk_size = match action
        .header
        .get(LFS_MULTIPART_CHUNK_SIZE_HEADER)
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(c) if c > 0 => c,
        _ => return UploadAction::Single(action),
    };
    let parts = sorted_part_urls(&action.header);
    if parts.is_empty() {
        return UploadAction::Single(action);
    }
    UploadAction::Multipart {
        completion_href: action.href,
        chunk_size,
        parts,
    }
}

/// Collect the numbered `0000N` keys from the multipart upload header
/// and return their values (pre-signed PUT URLs) sorted by integer
/// part number ascending. Non-digit-only keys (including
/// `chunk_size`) are ignored.
///
/// Mirrors `huggingface_hub._commit_api._get_sorted_parts_urls`.
fn sorted_part_urls(header: &HashMap<String, String>) -> Vec<String> {
    let mut indexed: Vec<(u32, String)> = header
        .iter()
        .filter_map(|(k, v)| {
            if !k.chars().all(|c| c.is_ascii_digit()) || k.is_empty() {
                return None;
            }
            let n = k.parse::<u32>().ok()?;
            Some((n, v.clone()))
        })
        .collect();
    indexed.sort_by_key(|&(n, _)| n);
    indexed.into_iter().map(|(_, url)| url).collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_support::EnvBaseGuard;
    use super::*;
    use serial_test::serial;

    // ─── parse_lfs_batch_response ──────────────────────────────────

    /// Helper to assert the parsed upload action is a single-PUT and
    /// project out the inner `LfsAction`.
    fn assert_single_upload(upload: Option<UploadAction>) -> LfsAction {
        match upload {
            Some(UploadAction::Single(action)) => action,
            other => panic!("expected UploadAction::Single, got {other:?}"),
        }
    }

    #[test]
    fn parse_lfs_batch_with_upload_and_verify_actions() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/upload",
                        "header": {"Authorization": "Bearer x", "X-Amz": "sig"}
                    },
                    "verify": {
                        "href": "https://lfs.example/verify",
                        "header": {}
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        let upload = assert_single_upload(parsed.upload);
        assert_eq!(upload.href, "https://lfs.example/upload");
        assert_eq!(
            upload.header.get("Authorization").map(|s| s.as_str()),
            Some("Bearer x")
        );
        assert_eq!(upload.header.get("X-Amz").map(|s| s.as_str()), Some("sig"));
        let verify = parsed.verify.expect("verify action present");
        assert_eq!(verify.href, "https://lfs.example/verify");
        assert!(verify.header.is_empty());
    }

    /// Multipart response: `header.chunk_size` present plus numbered
    /// part URLs (`00001`, `00002`, …). Detection rule is `chunk_size`
    /// presence — matches Python's `_upload_multi_part` branch.
    #[test]
    fn parse_lfs_batch_multipart_shape_is_promoted() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/complete",
                        "header": {
                            "chunk_size": "104857600",
                            "00001": "https://lfs.example/part-1",
                            "00002": "https://lfs.example/part-2",
                            "00003": "https://lfs.example/part-3"
                        }
                    },
                    "verify": {"href": "https://lfs.example/verify", "header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        match parsed.upload {
            Some(UploadAction::Multipart {
                completion_href,
                chunk_size,
                parts,
            }) => {
                assert_eq!(completion_href, "https://lfs.example/complete");
                assert_eq!(chunk_size, 104_857_600);
                assert_eq!(parts.len(), 3);
                assert_eq!(parts[0], "https://lfs.example/part-1");
                assert_eq!(parts[1], "https://lfs.example/part-2");
                assert_eq!(parts[2], "https://lfs.example/part-3");
            }
            other => panic!("expected Multipart, got {other:?}"),
        }
        assert!(parsed.verify.is_some());
    }

    /// Part URLs returned out of order in the JSON header must come
    /// back ordered by numeric part number. HF is permitted to
    /// return them in any order; the multipart upload spec requires
    /// us to PUT in `partNumber` order and reference the resulting
    /// ETags in the same order on completion.
    #[test]
    fn parse_lfs_batch_multipart_parts_are_sorted_numerically() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/complete",
                        "header": {
                            "chunk_size": "100",
                            "00010": "https://lfs.example/part-10",
                            "00002": "https://lfs.example/part-2",
                            "00001": "https://lfs.example/part-1"
                        }
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        match parsed.upload.unwrap() {
            UploadAction::Multipart { parts, .. } => {
                assert_eq!(parts.len(), 3);
                assert_eq!(parts[0], "https://lfs.example/part-1");
                assert_eq!(parts[1], "https://lfs.example/part-2");
                // Numeric (not lexicographic) sort: 10 > 2.
                assert_eq!(parts[2], "https://lfs.example/part-10");
            }
            _ => panic!("expected Multipart"),
        }
    }

    /// Malformed multipart response: `chunk_size` present but no
    /// numbered part URLs. We fall back to `Single` rather than
    /// panicking — caller will then attempt single-PUT, which HF
    /// will reject for >5 GB files, but at least the user gets a
    /// real error message instead of a panic. `chunk_size=0` is
    /// likewise treated as malformed.
    #[test]
    fn parse_lfs_batch_multipart_without_parts_falls_back_to_single() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/upload",
                        "header": {"chunk_size": "1048576"}
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        let upload = assert_single_upload(parsed.upload);
        assert_eq!(upload.href, "https://lfs.example/upload");
    }

    /// `chunk_size: "0"` is malformed — treated like missing.
    #[test]
    fn parse_lfs_batch_multipart_with_zero_chunk_size_falls_back_to_single() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/upload",
                        "header": {"chunk_size": "0", "00001": "https://x"}
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(matches!(parsed.upload, Some(UploadAction::Single(_))));
    }

    /// Direct unit test of the helper that picks single vs multipart.
    /// Decoupled from `parse_lfs_batch_response` so a future caller
    /// (e.g. resume-from-checkpoint) that builds an `LfsAction` by
    /// hand and calls `promote_upload_action` gets the same dispatch.
    #[test]
    fn promote_upload_action_single_when_chunk_size_absent() {
        let action = LfsAction {
            href: "https://lfs.example/up".into(),
            header: {
                let mut h = HashMap::new();
                h.insert("Authorization".into(), "Bearer x".into());
                h
            },
        };
        match promote_upload_action(action) {
            UploadAction::Single(a) => assert_eq!(a.href, "https://lfs.example/up"),
            other => panic!("expected Single, got {other:?}"),
        }
    }

    /// `promote_upload_action` on a non-numeric `chunk_size` value
    /// (e.g. `"oops"`) falls back to Single — same defensive rule as
    /// the missing/zero case. Mirrors HF's Python:
    /// `int(chunk_size)` failure raises `ValueError`; we choose the
    /// gentler fallback so the user can still attempt single-PUT
    /// and get a clear error from HF if that fails.
    #[test]
    fn promote_upload_action_falls_back_when_chunk_size_unparseable() {
        let mut header = HashMap::new();
        header.insert("chunk_size".into(), "not-a-number".into());
        header.insert("00001".into(), "https://x/p1".into());
        let action = LfsAction {
            href: "https://x/up".into(),
            header,
        };
        assert!(matches!(
            promote_upload_action(action),
            UploadAction::Single(_)
        ));
    }

    /// Direct test of the part-URL sorter — invariants that the
    /// public `parse_lfs_batch_response` test relies on but doesn't
    /// pin down structurally.
    #[test]
    fn sorted_part_urls_handles_empty_key_and_non_digit_key() {
        let mut header = HashMap::new();
        header.insert("".into(), "https://x/empty-key".into());
        header.insert("00001".into(), "https://x/p1".into());
        header.insert("12a".into(), "https://x/mixed".into());
        header.insert("00002".into(), "https://x/p2".into());
        let parts = sorted_part_urls(&header);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "https://x/p1");
        assert_eq!(parts[1], "https://x/p2");
    }

    /// Non-numeric keys in the multipart header (e.g. AWS sig headers
    /// HF may include alongside the per-part URLs) must be ignored
    /// when building the parts list — only `\d+` keys count.
    #[test]
    fn parse_lfs_batch_multipart_ignores_non_numeric_keys() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/complete",
                        "header": {
                            "chunk_size": "100",
                            "00001": "https://x/p1",
                            "X-Aux-Signature": "ignore-me",
                            "00002": "https://x/p2"
                        }
                    }
                }
            }]
        });
        match parse_lfs_batch_response(&json).unwrap().upload.unwrap() {
            UploadAction::Multipart { parts, .. } => assert_eq!(parts.len(), 2),
            _ => panic!("expected Multipart"),
        }
    }

    #[test]
    fn parse_lfs_batch_with_no_upload_means_object_already_present() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "verify": {"href": "https://lfs.example/verify", "header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none(), "upload action absent");
        assert!(parsed.verify.is_some(), "verify action present");
    }

    #[test]
    fn parse_lfs_batch_with_no_actions_returns_both_none() {
        let json = serde_json::json!({"objects": [{}]});
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
        assert!(parsed.verify.is_none());
    }

    #[test]
    fn parse_lfs_batch_missing_objects_array_errors() {
        let json = serde_json::json!({});
        let err = parse_lfs_batch_response(&json).expect_err("objects[] missing must error");
        assert!(err.to_string().contains("missing `objects`"));
    }

    #[test]
    fn parse_lfs_batch_empty_objects_array_errors() {
        let json = serde_json::json!({"objects": []});
        let err = parse_lfs_batch_response(&json).expect_err("empty objects[] must error");
        assert!(err.to_string().contains("objects[] empty"));
    }

    #[test]
    fn parse_lfs_batch_per_object_error_surfaces() {
        let json = serde_json::json!({
            "objects": [{
                "error": {"code": 422, "message": "object too large"}
            }]
        });
        let err = parse_lfs_batch_response(&json).expect_err("inline object error");
        let msg = err.to_string();
        assert!(msg.contains("LFS batch object error"), "{msg}");
        assert!(msg.contains("too large"), "{msg}");
    }

    #[test]
    fn parse_lfs_batch_action_without_href_is_skipped() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {"header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
    }

    // ─── lfs_batch_upload (HTTP-mocked) ─────────────────────────────

    #[test]
    #[serial]
    fn lfs_batch_upload_returns_actions_from_server() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .match_header("authorization", "Bearer t")
            .match_header("accept", "application/vnd.git-lfs+json")
            .match_header("content-type", "application/vnd.git-lfs+json")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "operation": "upload",
                "transfers": ["basic"],
                "hash_algo": "sha256",
            })))
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": "https://lfs.example/up", "header": {"X-Sig": "abc"}},
                            "verify": {"href": "https://lfs.example/v", "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();

        let resp = lfs_batch_upload("org/repo", "t", "deadbeef", 1024, "model").unwrap();
        mock.assert();
        let upload = assert_single_upload(resp.upload);
        assert_eq!(upload.href, "https://lfs.example/up");
        assert_eq!(upload.header.get("X-Sig").map(|s| s.as_str()), Some("abc"));
        assert!(resp.verify.is_some());
    }

    /// The batch request must declare both `basic` and `multipart`
    /// transfer adapters — without `multipart` in the list HF returns
    /// `400 "You need to configure your repository to enable upload
    /// of files > 5GB"` for any object that exceeds the basic single-
    /// PUT ceiling.
    #[test]
    #[serial]
    fn lfs_batch_request_declares_multipart_transfer_capability() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "transfers": ["basic", "multipart"],
            })))
            .with_status(200)
            .with_body(r#"{"objects":[{"actions":{}}]}"#)
            .create();

        let _ = lfs_batch_upload("org/repo", "t", "x", 1, "model").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_dataset_repo_path() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/datasets/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(r#"{"objects":[{"actions":{}}]}"#)
            .create();

        let _ = lfs_batch_upload("org/repo", "t", "x", 1, "dataset").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(500)
            .with_body("boom")
            .create();

        let err = lfs_batch_upload("org/repo", "t", "x", 1, "model").expect_err("500 must error");
        mock.assert();
        assert!(err.to_string().contains("500"));
    }

    /// Malformed JSON body (200 status but body is not parseable as
    /// JSON) surfaces a `LFS batch JSON` parse error rather than
    /// panicking. Hits the `.json()` deserialization branch.
    #[test]
    #[serial]
    fn lfs_batch_upload_malformed_json_body_surfaces_parse_error() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_header("content-type", "application/vnd.git-lfs+json")
            .with_body("{not-json")
            .create();

        let err = lfs_batch_upload("org/repo", "t", "x", 1, "model")
            .expect_err("malformed JSON must error");
        mock.assert();
        assert!(err.to_string().contains("LFS batch JSON"), "{err}");
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_per_object_error_surfaces() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(r#"{"objects":[{"error":{"code":422,"message":"too big"}}]}"#)
            .create();

        let err = lfs_batch_upload("org/repo", "t", "x", 1, "model").expect_err("inline error");
        mock.assert();
        assert!(err.to_string().contains("LFS batch object error"));
    }
}
