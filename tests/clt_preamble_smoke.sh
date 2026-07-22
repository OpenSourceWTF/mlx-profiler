#!/bin/bash

set -euo pipefail

SRC_DIR=$1
OUTPUT_DIR=$2
CC=$3
FAKE_BIN=${OUTPUT_DIR}/fake-bin

mkdir -p "${FAKE_BIN}"
printf '#!/bin/bash\nexit 1\n' > "${FAKE_BIN}/xcrun"
chmod +x "${FAKE_BIN}/xcrun"

PATH="${FAKE_BIN}:${PATH}" \
  bash "${SRC_DIR}/mlx/backend/metal/make_compiled_preamble.sh" \
  "${OUTPUT_DIR}" "${CC}" "${SRC_DIR}" binary_ops ""

PREAMBLE=${OUTPUT_DIR}/binary_ops.cpp
grep -q 'Contents from "mlx/backend/metal/kernels/binary_ops.h"' "${PREAMBLE}"

if grep -q 'clt_stub_includes' "${PREAMBLE}"; then
  echo "CLT-only stub headers leaked into the generated Metal preamble" >&2
  exit 1
fi
