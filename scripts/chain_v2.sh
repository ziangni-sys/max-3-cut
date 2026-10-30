#!/bin/bash
cd /home/user/maxkcut-gadgets
CK="results/s2_k4_i0_s51_cap28.ckpt results/s2_k4_i1_s51_cap28.ckpt results/s2_k4_i2_s51_cap28.ckpt results/s2_k4_i3_s51_cap28.ckpt"
export OMP_NUM_THREADS=1
python3 scripts/polish_joint.py 4 results/fam_k4_v2r.json $CK --workers 4 > results/polish_v2r.log 2>&1
python3 scripts/verify_cert2.py results/fam_k4_v2r.json 4 > results/verify_v2r.log 2>&1
python3 scripts/polish_joint.py 4 results/fam_k4_v2o.json $CK --workers 4 --original > results/polish_v2o.log 2>&1
python3 scripts/verify_cert2.py results/fam_k4_v2o.json 4 > results/verify_v2o.log 2>&1
echo CHAIN_DONE >> results/verify_v2o.log
