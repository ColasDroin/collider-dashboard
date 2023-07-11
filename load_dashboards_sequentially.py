import init

for x in range(26):
    path_collider = f"/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/all_optics_2024_reverted/collider_{x:02}/xtrack_0000/collider.json"
    path_config = None
    path_job = path_collider.split("/final_collider.json")[0]
    dic_without_bb, dic_with_bb = init.init_from_collider(
        path_collider, load_global_variables_from_pickle=False
    )
