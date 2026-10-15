# Original PGM access audit

Checked 2026-09-24. The user supplied a signed-in Google Cloud console screenshot listing the original `ravens-matrices` archives, including `extrapolation.tar.gz` (displayed as 40.4 GB) and `neutral.tar.gz` (51.1 GB). These are screenshot display sizes, not independently verified exact byte counts. The [official DeepMind repository](https://github.com/google-deepmind/abstract-reasoning-matrices) still links to [this bucket](https://console.cloud.google.com/storage/browser/ravens-matrices); its README and public issue/comment history did not identify a replacement location.

The screenshot establishes that the archives are listed for that signed-in account. Our anonymous requests were denied. It is therefore misleading to describe these results as evidence that the dataset was deleted or moved. Browser login also does not automatically authenticate a separate CLI on ICE. Listing permission alone does not establish permission to read an object's contents; the next useful check is a small authenticated request using the same Google account.

## Bounded anonymous checks

All probes used `curl -q`, no credentials or cookies, a requested byte range of `0-1023`, a 15-second timeout, and a 65,536-byte maximum response size. Total response bodies downloaded across all ten probes: **3,352 bytes**. No archive data was returned. There were no bucket mutations.

In the table, `B = ravens-matrices` and `O = extrapolation.tar.gz`.

| Endpoint tested | Result |
| --- | --- |
| `https://storage.googleapis.com/B/O` | 403; anonymous `storage.objects.get` denied |
| `https://B.storage.googleapis.com/O` | 403; anonymous `storage.objects.get` denied |
| `https://storage.googleapis.com/storage/v1/b/B/o/O?alt=media` | 401; anonymous `storage.objects.get` denied |
| `https://storage.googleapis.com/download/storage/v1/b/B/o/O?alt=media` | 401; anonymous `storage.objects.get` denied |
| `https://www.googleapis.com/storage/v1/b/B/o/O?alt=media` | 401; anonymous `storage.objects.get` denied |
| `https://content-storage.googleapis.com/download/storage/v1/b/B/o/O?alt=media` | 307 to a Google download host; following the redirect with the same bounds returned 401 |
| JSON object metadata, requesting only name/size/generation/MD5 fields | 401; anonymous `storage.objects.get` denied |
| JSON bucket list, exact object-name prefix and maximum one result | 401; anonymous `storage.objects.list` denied |
| `https://storage.cloud.google.com/B/O` | 302 to Google sign-in; sign-in redirect was not followed |

These are documented [Cloud Storage endpoint forms](https://docs.cloud.google.com/storage/docs/request-endpoints). The browser download endpoint always requires authentication, even for publicly readable objects; its sign-in redirect by itself is not evidence of restrictive object permissions. JSON object retrieval supports ranges and requires [`storage.objects.get`](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/get). Changing from XML to JSON did not make anonymous access work in this audit.

## User-controlled authentication, if needed

First check whether `gcloud` is installed on the machine that will read the archive. If the CLI already has a suitable authorized account, a new login may be unnecessary. Do not inspect or copy credential files, browser cookies, or access tokens.

For ICE without a browser, the user can run this in their own SSH terminal:

```bash
gcloud auth login --no-launch-browser
```

The user opens the displayed Google authorization URL in a trusted local browser, chooses the same account used in the screenshot, and pastes the authorization code back into their SSH terminal. The code should not be pasted into chat. This command makes the authorized account active in the current gcloud configuration. A local machine with a browser can instead use `gcloud auth login`. Google also documents `--no-browser` with a local `--remote-bootstrap` command when both machines have gcloud installed. See the [official login instructions](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login). Application Default Credentials and service-account keys are unnecessary for these CLI checks.

After the user completes authentication, perform metadata and a 1 KiB read before considering an archive download:

```bash
gcloud storage objects describe \
  gs://ravens-matrices/extrapolation.tar.gz \
  --raw --format='json(name,size,generation,md5Hash,crc32c)'

PGM_PROBE=$(mktemp "${TMPDIR:-/tmp}/pgm-extrapolation-probe.XXXXXX")
gcloud storage cat gs://ravens-matrices/extrapolation.tar.gz \
  --range=0-1023 > "$PGM_PROBE"
wc -c "$PGM_PROBE"
```

The [metadata command](https://docs.cloud.google.com/sdk/gcloud/reference/storage/objects/describe) does not download the archive. The [`cat` range](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cat) is inclusive, so successful output is 1,024 bytes. No full archive download or authenticated check was performed as part of this audit.

If the same signed-in account still receives a permission error, authentication alone did not grant object access; preserve the non-secret error and seek access from the dataset owner. If the bounded read succeeds, record exact object size, generation and checksum before arranging storage and a deliberate download on ICE. This audit does not change the current experiment scope of the accessible PGM neutral mirror and the already rendered 16 CLEVR-ART RMTS examples.
