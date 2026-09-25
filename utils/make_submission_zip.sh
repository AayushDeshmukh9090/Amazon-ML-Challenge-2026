#!/usr/bin/env bash
# Build <team>_submission.zip in the exact structure the organisers require.
#   bash utils/make_submission_zip.sh <team_name>
set -euo pipefail
TEAM=${1:?usage: make_submission_zip.sh <team_name>}
STAGE=$(mktemp -d)
mkdir -p "$STAGE/output" "$STAGE/code"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
rsync -a --exclude '__pycache__' code/business_entity_resolution "$STAGE/code/"
cp Documentation_template.md "$STAGE/"
(cd "$STAGE" && zip -qr "$OLDPWD/${TEAM}_submission.zip" .)
rm -rf "$STAGE"
unzip -l "${TEAM}_submission.zip"
