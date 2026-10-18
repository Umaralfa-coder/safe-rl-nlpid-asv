#!/usr/bin/env bash
# Switch the VRX sydney_regatta world between the two weather settings used in the paper.
#   normal: mean wind 6 m/s, gust gain 2, wave gain 0.8   (E1-E3)
#   strong: mean wind 12 m/s, gust gain 4, wave gain 1.6  (E2d, everything doubled)
# Usage: scripts/set_weather.sh normal|strong [path/to/sydney_regatta.sdf]
# Without a path, the world of the sourced vrx_gz install is used. Restart Gazebo afterwards.
set -euo pipefail
mode="${1:-}"
case "$mode" in
  normal) W=6.0;  G=2.0; A=0.8 ;;
  strong) W=12.0; G=4.0; A=1.6 ;;
  *) echo "usage: $0 normal|strong [world.sdf]"; exit 1 ;;
esac
if [ $# -ge 2 ]; then
  f="$2"
else
  prefix="$(ros2 pkg prefix vrx_gz)"
  f="$prefix/share/vrx_gz/worlds/sydney_regatta.sdf"
fi
[ -f "$f" ] || { echo "world file not found: $f"; exit 1; }
[ -f "$f.orig" ] || cp "$f" "$f.orig"
sed -i -E \
  -e "s|<wind_mean_velocity>[0-9.]+</wind_mean_velocity>|<wind_mean_velocity>$W</wind_mean_velocity>|" \
  -e "s|<var_wind_gain_constants>[0-9.]+</var_wind_gain_constants>|<var_wind_gain_constants>$G</var_wind_gain_constants>|" \
  "$f"
# wave gain: the double_value line right after key "gain" in the wave-field publisher
python3 - "$f" "$A" <<'PY'
import re, sys
f, a = sys.argv[1], sys.argv[2]
s = open(f).read()
s2, n = re.subn(r'(key:\s*"gain"\s*value\s*\{\s*type:\s*DOUBLE\s*double_value:\s*)[0-9.]+', r'\g<1>' + a, s)
if n != 1:
    sys.exit(f'expected one wave "gain" entry, found {n}')
open(f, 'w').write(s2)
PY
echo "$mode weather set in $f"
grep -E "wind_mean_velocity|var_wind_gain_constants" "$f"
grep -A4 'key: "gain"' "$f" | grep double_value
