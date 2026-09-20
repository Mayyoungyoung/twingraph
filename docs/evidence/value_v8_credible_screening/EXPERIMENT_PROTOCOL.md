# V8 development protocol and current limits

Task: complete clean, five-part assembly, bidirectional post-handle stroke and final release. All candidates share the 80 mm minimum stroke and the revised pin geometry. Candidate pool is scripted and fixed per seed. Paired trial fields currently applied are friction and actuator gain. Repeat zero is nominal for both twin and target namespaces; this does not establish target independence.

Command used for the 2080 server pilot: `MUJOCO_GL=egl python -m simbench.value.collect_v7 --out ~/twingraph-v8-results/pilot --seed 1200 --groups 1 --n 2 --repeats 1 --workers 1 --level L1 --timeout 360`. The source was transferred as an archive of `simbench` and `tests` from commit `38116c2`. Server environment: Python 3.12, MuJoCo, EGL. Seed 1200 is a regression/development case.

The collector currently uses a shared Session restored from a snapshot. A unit test covers two omitted fields, but full fresh-Session versus shared-Session order invariance has not been shown. Therefore these runs are development diagnostics, not final fair selector comparisons. No train/validation/test claims or twin-versus-independent-target claims are made. Incomplete groups are preserved; the existing collector rejects a partially populated directory on resume, so robust per-candidate resume remains outstanding.
