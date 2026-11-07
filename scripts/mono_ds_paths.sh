# Sourced after env.sh: finds the extracted mono IEMOCAP and VoxCeleb1 roots under $SAC_DATA_ROOT/mono.
_mono=${SAC_DATA_ROOT}/mono
export IEMOCAP_ROOT=${IEMOCAP_ROOT:-$(find "${_mono}" -maxdepth 3 -type d -name Session1 -printf '%h\n' | head -1)}
export VOXCELEB1_ROOT=${VOXCELEB1_ROOT:-$(find "${_mono}" -maxdepth 3 -type d -path '*oxceleb*' -name wav -printf '%h\n' | head -1)}
echo "IEMOCAP_ROOT=${IEMOCAP_ROOT} VOXCELEB1_ROOT=${VOXCELEB1_ROOT}"
[ -n "${IEMOCAP_ROOT}" ] && [ -n "${VOXCELEB1_ROOT}" ]
