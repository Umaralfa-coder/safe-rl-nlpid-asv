# Mission videos

These videos are **replays of the logged VRX runs**, drawn from the controller CSV logs
(positions, heading, cross-track error), not screen recordings of Gazebo. Runs are started at
the first waypoint they share and synchronised by distance sailed, so the boats stay side by
side even when one is slower.

| File | Runs | Content |
|---|---|---|
| `calm_handtuned_vs_learned.mp4` | A_s7, TURN_eval | calm water (wind 6 m/s), hand-tuned A vs learned turn-aware L_t, first 540 m |
| `strong_wind_frozen_vs_resumed.mp4` | S1_strong_A, S2b_strong_LT, R3_strong_eval | doubled wind and waves, A vs frozen L_t vs resumed L_t+ |
| `wq_map_raw_vs_lag_corrected.mp4` | TURN_eval, field seed 3 | GP map built along the path: true field, raw 5-s probe, lag-corrected probe |

Gaps in the cross-track curve are long transit legs: there the guidance flies straight to the
next waypoint and logs no cross-track error.

Regenerate (needs ffmpeg):

```bash
L=~/seal_ws/logs
python3 analysis/figures/make_videos.py replay --skip 250 --length 540 \
  --title "Calm water: hand-tuned vs learned NLPID gains" \
  --run "A (hand-tuned)=$L/A_s7" --run "L_t (learned, turn-aware cost)=$L/TURN_eval" \
  --out video/calm_handtuned_vs_learned.mp4
python3 analysis/figures/make_videos.py replay --skip 60 --length 540 --cte-max 6 \
  --title "Doubled wind and waves: frozen vs resumed learning" \
  --run "A (hand-tuned)=$L/S1_strong_A" --run "L_t (frozen)=$L/S2b_strong_LT" \
  --run "L_t+ (resumed learning)=$L/R3_strong_eval" --out video/strong_wind_frozen_vs_resumed.mp4
python3 analysis/figures/make_videos.py wqmap --run $L/TURN_eval --field 3 \
  --out video/wq_map_raw_vs_lag_corrected.mp4
```

(`--skip 250` for A_s7 drops the start of that log, where two controllers were running.)

To add a real Gazebo recording, record the screen while a mission runs (e.g. with OBS or
`ffmpeg -f x11grab`), compress it (`ffmpeg -i in.mp4 -vcodec libx264 -crf 28 out.mp4`) and keep
it under about 20 MB.
