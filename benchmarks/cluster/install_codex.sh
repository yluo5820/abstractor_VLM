#!/usr/bin/env bash
# Install a pinned official native package into one explicit user-owned prefix.
# Run inside a GPU allocation after confirming PACE user-local software policy.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  printf 'Usage: install_codex.sh /absolute/verified/scratch/tools/codex-0.156.1\n' >&2
  exit 2
fi
codex_prefix=$1
case "$codex_prefix" in
  /*) ;;
  *) printf 'Installation prefix must be an absolute path.\n' >&2; exit 2;;
esac
case "/${codex_prefix#/}/" in
  */../*|*/./*|//*|/home/|/scratch/|/tmp/)
    printf 'Choose a specific absolute user-owned installation directory.\n' >&2
    exit 2;;
esac
if [ -e "$codex_prefix" ] || [ -L "$codex_prefix" ]; then
  printf 'Refusing to overwrite existing path: %s\n' "$codex_prefix" >&2
  exit 2
fi
cluster_dir=$(cd "$(dirname "$0")" && pwd)
source "$cluster_dir/guard.sh"
require_gpu_allocation
test "$(uname -s)" = Linux || { printf 'This helper targets Linux compute nodes.\n' >&2; exit 2; }
for utility in curl sha256sum tar mktemp; do command -v "$utility" >/dev/null; done
codex_version=0.156.1
case "$(uname -m)" in
  x86_64|amd64)
    codex_target=x86_64-unknown-linux-musl
    codex_sha256=8b711520beddf385467b8da4d2c93736637c6ba1e46811cf0d8606b7c490b6f6;;
  aarch64|arm64)
    codex_target=aarch64-unknown-linux-musl
    codex_sha256=fdd47ed6aade0360796fd3f6f95a45096f327c15e19e8c7339f9dc5633041786;;
  *) printf 'Unsupported Linux architecture.\n' >&2; exit 2;;
esac
codex_parent=$(dirname "$codex_prefix")
mkdir -p "$codex_parent"
if [ ! -O "$codex_parent" ]; then
  printf 'Installation parent must be owned by the current user: %s\n' "$codex_parent" >&2
  exit 2
fi
codex_download=$(mktemp -d "$codex_parent/.codex-download.XXXXXX")
trap 'rm -rf -- "$codex_download"' EXIT
codex_url="https://releases.openai.com/codex/releases/$codex_version/codex-package-$codex_target.tar.gz"
curl --fail --location --silent --show-error --connect-timeout 10 --max-time 300 \
  "$codex_url" --output "$codex_download/codex.tar.gz"
printf '%s  %s\n' "$codex_sha256" "$codex_download/codex.tar.gz" | sha256sum -c -
# mkdir atomically refuses an existing destination, including a concurrent install.
mkdir "$codex_prefix"
tar --no-same-owner -xzf "$codex_download/codex.tar.gz" -C "$codex_prefix"
chmod 0755 "$codex_prefix/bin/codex" "$codex_prefix/bin/codex-code-mode-host" \
  "$codex_prefix/codex-path/rg" "$codex_prefix/codex-resources/bwrap"
printf 'version=%s\nurl=%s\nsha256=%s\n' "$codex_version" "$codex_url" "$codex_sha256" \
  > "$codex_prefix/installation-source.txt"
"$codex_prefix/bin/codex" --version
printf '\nInstalled without changing shell startup files. User authentication command:\n'
printf '%q login --device-auth\n' "$codex_prefix/bin/codex"
