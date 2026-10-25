#!/usr/bin/env bash
# Runs ON GitHub Actions (unrestricted network). Downloads each manifest entry,
# verifies sha256, splits into <=95MB chunks for in-repo storage, uploads the
# full file to a GitHub release (for direct download by end users).
# Never exits non-zero for a single entry: writes model-fetch/out/_failures.txt
# instead, so successful entries are still committed by the workflow.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=model-fetch/out
mkdir -p "$OUT"
exec > >(tee -a "$OUT/_log.txt") 2>&1
: > "$OUT/_failures.txt"

sha_of_parts() { cat "$OUT/$1".part-* 2>/dev/null | sha256sum | cut -d' ' -f1; }

jq -c '.entries[]' model-fetch/manifest.json | while IFS= read -r e; do
  id=$(jq -r .id <<<"$e")
  fname=$(jq -r .filename <<<"$e")
  echo "=== [$id] ==="

  # --- skip when chunks already present and hash matches ---
  if [ -f "$OUT/$id.sha256" ] && ls "$OUT/$id".part-* >/dev/null 2>&1; then
    expected=$(cut -d' ' -f1 "$OUT/$id.sha256")
    if [ "$(sha_of_parts "$id")" = "$expected" ]; then
      echo "skip $id (chunks already present); ensuring release asset exists"
      tag=$(jq -r .release_tag model-fetch/manifest.json)
      if ! gh release view "$tag" --json assets 2>/dev/null | jq -e --arg n "$fname" '.assets[]|select(.name==$n)' >/dev/null 2>&1; then
        cat "$OUT/$id".part-* > "$OUT/$fname"
        if gh release view "$tag" >/dev/null 2>&1; then
          gh release upload "$tag" "$OUT/$fname" --clobber || echo "RELEASE_UPLOAD_FAILED $id" | tee -a "$OUT/_failures.txt"
        else
          gh release create "$tag" "$OUT/$fname" --title "Ravan-AI v3 model assets" \
            --notes "Pretrained open-source assets for Ravan-AI v3." || echo "RELEASE_CREATE_FAILED $id" | tee -a "$OUT/_failures.txt"
        fi
        rm -f "$OUT/$fname"
      fi
      continue
    fi
  fi

  rm -f "$OUT/$id".part-* "$OUT/$id.full" "$OUT/$id.sha256"

  # --- resolve download URL ---
  url=""
  src=$(jq -r .source <<<"$e")
  if [ "$src" = "hf" ]; then
    pat=$(jq -r .filename_regex <<<"$e")
    while IFS= read -r repo; do
      [ -n "$repo" ] || continue
      meta=$(curl -fsSL --retry 4 --retry-delay 4 "https://huggingface.co/api/models/$repo" || true)
      [ -n "$meta" ] || continue
      f=$(jq -r '.siblings[]?.rfilename' <<<"$meta" | grep -Ei "$pat" | head -1 || true)
      if [ -n "$f" ]; then
        url="https://huggingface.co/$repo/resolve/main/$f"
        echo "resolved HF $repo/$f"
        break
      fi
    done < <(jq -r '.repo, (.fallback_repos[]?)' <<<"$e")
  elif [ "$src" = "ghrelease" ]; then
    repo=$(jq -r .repo <<<"$e")
    pat=$(jq -r .asset_regex <<<"$e")
    url=$(gh api "repos/$repo/releases?per_page=20" \
      | jq -r --arg pat "$pat" '[.[] | .assets[]? | select(.name | test($pat))][0].browser_download_url // empty')
    echo "resolved GH release url: ${url:-NONE}"
  fi

  if [ -z "$url" ]; then
    echo "RESOLVE_FAILED $id" | tee -a "$OUT/_failures.txt"
    continue
  fi

  # --- download ---
  if ! curl -fL --retry 5 --retry-delay 4 --connect-timeout 20 -o "$OUT/$id.full" "$url"; then
    echo "DOWNLOAD_FAILED $id ($url)" | tee -a "$OUT/_failures.txt"
    rm -f "$OUT/$id.full"
    continue
  fi

  sha=$(sha256sum "$OUT/$id.full" | cut -d' ' -f1)
  echo "$sha  $fname" > "$OUT/$id.sha256"
  size=$(stat -c%s "$OUT/$id.full")
  echo "downloaded $id: $size bytes sha=$sha"

  # --- upload full file to the shared release (user-facing single-file download) ---
  tag=$(jq -r .release_tag model-fetch/manifest.json)
  cp "$OUT/$id.full" "$OUT/$fname"
  if gh release view "$tag" >/dev/null 2>&1; then
    gh release upload "$tag" "$OUT/$fname" --clobber || \
      echo "RELEASE_UPLOAD_FAILED $id" | tee -a "$OUT/_failures.txt"
  else
    gh release create "$tag" "$OUT/$fname" \
      --title "Ravan-AI v3 model assets" \
      --notes "Pretrained open-source assets fetched for Ravan-AI v3 (Qwen3 GGUF quantizations, llama.cpp Windows runtime). Chunks also committed under model-fetch/out/ for in-repo retrieval." || \
      echo "RELEASE_CREATE_FAILED $id" | tee -a "$OUT/_failures.txt"
  fi
  rm -f "$OUT/$fname"

  # --- split into <=95MB chunks (GitHub hard limit is 100MB/file) ---
  split -b 95000000 -d -a 2 "$OUT/$id.full" "$OUT/$id.part-"
  rm -f "$OUT/$id.full"
  # sanity: reassemble hash must match
  if [ "$(sha_of_parts "$id")" = "$sha" ]; then
    echo "chunks ok: $(ls "$OUT/$id".part-* | wc -l) parts"
  else
    echo "CHUNK_VERIFY_FAILED $id" | tee -a "$OUT/_failures.txt"
    rm -f "$OUT/$id".part-*
  fi
done

echo "=== done; failures: ==="
cat "$OUT/_failures.txt" || true
