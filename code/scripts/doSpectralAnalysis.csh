#!/bin/csh

foreach cid ( 0 1 4 6 7 15 26 34 35 37 44 49 50 52 58 66 72 )
    echo Processing cluster_id=$cid
    ipython --pdb doSpectralAnalysis.py -- --cluster_id $cid
end
