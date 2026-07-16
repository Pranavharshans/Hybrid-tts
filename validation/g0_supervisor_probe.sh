#!/usr/bin/env bash
set -euo pipefail

artifact_dir="${NANO_FLASH_ARTIFACT_DIR:-/workspace/nano-flash-artifacts/g0}"
mkdir -p "${artifact_dir}"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sleep 5
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

temporary="${artifact_dir}/supervisor-probe.json.tmp"
final="${artifact_dir}/supervisor-probe.json"
printf '{"status":"completed","started_at":"%s","finished_at":"%s","pid":%d}\n' \
  "${started_at}" "${finished_at}" "$$" > "${temporary}"
mv "${temporary}" "${final}"
