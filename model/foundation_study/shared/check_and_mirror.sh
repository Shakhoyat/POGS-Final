#!/usr/bin/env bash
# Verify every row's run data exists on Linux (WSL), copy it to the E: drive
# share, then confirm the copy is byte-identical.
#
# WSL is the source of truth; E: is a transfer/staging area. Nothing in the
# pipeline writes to E:, so a run that "worked" leaves artefacts only under
# ~/model_study until this script mirrors them out.
#
# Usage (run inside WSL on the ML20 box):
#     bash shared/check_and_mirror.sh
#
# It checks, per row, per scene: bench/latency.json, train/training.json,
# track/tracking.json -- plus the row RESULTS.md, the sweep log and the root
# RESULTS.md. Any file missing or suspiciously small is reported as MISSING.
# It then copies artefacts + reports to E: and md5-verifies every copied file.
set -uo pipefail

STUDY="${STUDY:-$HOME/model_study}"
EDEST="${EDEST:-/mnt/e/do-not-delete -steerablity-challenge/model_study_results}"

# Rows and scenes come from the study itself, never hard-coded, so adding a
# row or scene cannot silently escape the check.
mapfile -t ROWS < <(python3 "$STUDY/shared/study_config.py" --list-rows)
mapfile -t SCENES < <(python3 "$STUDY/shared/download_weights.py" --list-scenes)

if [ "${#ROWS[@]}" -eq 0 ] || [ "${#SCENES[@]}" -eq 0 ]; then
  echo "could not read rows/scenes from study_config.py -- set STUDY or run from WSL" >&2
  exit 2
fi

# A zero-byte or near-empty json means the stage started and died, not that it
# produced a result. Threshold is bytes; 20 is enough for any real record.
MIN_BYTES="${MIN_BYTES:-20}"

MISSING=0
CHECKED=0

check_file() {
  local f="$1" label="$2"
  CHECKED=$((CHECKED + 1))
  if [ ! -f "$f" ]; then
    printf '  MISSING  %-70s (%s)\n' "${f#"$STUDY"/}" "$label"
    MISSING=$((MISSING + 1))
  elif [ "$(stat -c%s "$f")" -lt "$MIN_BYTES" ]; then
    printf '  EMPTY    %-70s (%s, %s bytes)\n' "${f#"$STUDY"/}" "$label" "$(stat -c%s "$f")"
    MISSING=$((MISSING + 1))
  fi
}

echo "=============================================================="
echo " CHECK  study root : $STUDY"
echo "        rows       : ${#ROWS[@]}   scenes: ${#SCENES[@]}"
echo "=============================================================="

for row in "${ROWS[@]}"; do
  echo
  echo "-- $row"
  for scene in "${SCENES[@]}"; do
    base="$STUDY/artefacts/$row/$scene"
    check_file "$base/bench/latency.json"     "bench/$scene"
    check_file "$base/train/training.json"    "train/$scene"
    check_file "$base/track/tracking.json"    "track/$scene"
  done
  check_file "$STUDY/$row/RESULTS.md" "row report"
  # A sweep log is named sweep.txt, or sweep_<set>.txt for non-default sets.
  if ! compgen -G "$STUDY/artefacts/$row/sweep*.txt" >/dev/null; then
    printf '  MISSING  %-70s (sweep log)\n' "artefacts/$row/sweep*.txt"
    MISSING=$((MISSING + 1))
  fi
  CHECKED=$((CHECKED + 1))
done

check_file "$STUDY/RESULTS.md" "root summary"
echo
echo "--------------------------------------------------------------"
echo " checked $CHECKED required files, $MISSING missing/empty"
echo "--------------------------------------------------------------"

if [ "$MISSING" -gt 0 ]; then
  echo "NOT COMPLETE on Linux. Re-run the missing rows/stages before mirroring."
  echo "A row that was 'tested' still needs its artefacts under ~/model_study/artefacts/<row>/<scene>/."
  exit 1
fi

echo
echo "Linux side is COMPLETE. Mirroring to E: ..."
mkdir -p "$EDEST" || { echo "cannot create $EDEST -- is the E: drive mounted?" >&2; exit 2; }

# Copy artefacts and reports. -a preserves mtimes so the mirror is checkable.
cp -a "$STUDY/artefacts" "$EDEST/" || { echo "copy of artefacts failed" >&2; exit 2; }
for row in "${ROWS[@]}"; do
  mkdir -p "$EDEST/$row"
  cp -a "$STUDY/$row/RESULTS.md" "$EDEST/$row/" 2>/dev/null
done
cp -a "$STUDY/RESULTS.md" "$EDEST/" 2>/dev/null

echo "Verifying the E: copy is byte-identical ..."
LIST_L="/tmp/_mirror_linux.txt"
LIST_E="/tmp/_mirror_estate.txt"
( cd "$STUDY"         && find artefacts RESULTS.md */RESULTS.md -type f 2>/dev/null | sort ) > "$LIST_L"
( cd "$EDEST" && find artefacts RESULTS.md */RESULTS.md -type f 2>/dev/null | sort ) > "$LIST_E"

DIFFF=$(comm -3 "$LIST_L" "$LIST_E")
if [ -n "$DIFFF" ]; then
  echo "FILE SET MISMATCH between Linux and E: (blank on one side only):"
  echo "$DIFFF"
  exit 1
fi

# md5 every file on both sides and compare the manifests.
( cd "$STUDY" && find artefacts -type f -print0 | sort -z | xargs -0 md5sum ) > /tmp/_md5_l.txt
( cd "$EDEST"  && find artefacts -type f -print0 | sort -z | xargs -0 md5sum ) > /tmp/_md5_e.txt
if ! diff -q /tmp/_md5_l.txt /tmp/_md5_e.txt >/dev/null; then
  echo "CONTENT MISMATCH -- some copied files differ:"
  diff /tmp/_md5_l.txt /tmp/_md5_e.txt | head -40
  exit 1
fi

echo
echo "=============================================================="
echo " OK  $(wc -l < /tmp/_md5_l.txt) files verified identical on Linux and E:."
echo "     Linux : $STUDY"
echo "     E:    : $EDEST"
echo "=============================================================="
