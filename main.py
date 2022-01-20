import argparse
from ortho_classes import Isoacceptor2, id_dict, synth_clean
from parallel import rnafold_in_parallel
import pandas as pd
from itertools import zip_longest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from pathlib import Path
import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help='Clean tRNADB-CE file')
    parser.add_argument("synth_file", help='File containing synthetase information')
    parser.add_argument("synth_name", help='Name of synthetase matching to entry in synth_file')
    parser.add_argument("amino_acid", help='Amino Acid specified for tRNA generation')
    parser.add_argument('-o', '--output_directory', help='Directory to store output files', default='')
    parser.add_argument('-ip', '--id_part_change', help='Identity parts that should be chimerified (except ID element)',
                        nargs='+')
    parser.add_argument("-cp", "--cluster_parts", type=int,
                        help='Number of parts for each part type to cluster', default=200)
    parser.add_argument("-l", "--length_filt", action="store_true", help='Filter chimeras if longer than 78 nts')
    parser.add_argument("-cf", "--cervettini_filt", nargs='+', help='Parameters for Cervettini Filtering in order '
                                                                    'start_stringency, minimum_stringency, target '
                                                                    'number of chimeras, and step size',
                        type=float, default=[0.5, 0, 2500000, 0.05])
    parser.add_argument('-a', '--anticodons', nargs='+', help='Anticodons to iterate through')

    parser.add_argument('-s', '--select', action='store_true', help='Decides if final selection step occurs')
    args = parser.parse_args()

    if args.cervettini_filt and len(args.cervettini_filt) > 4:
        raise Exception("Too many filtering parameters!")
    else:
        args.cervettini_filt = [(i if i is not None else j)
                                for i, j in zip_longest(args.cervettini_filt, [0.5, 0, 2500000, 0.05])]
    cf_start, cf_min, cf_targ, cf_ss = args.cervettini_filt

    first_ac = args.anticodons[0]

    Path(f'{args.output_directory}/folding').mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.file)
    synth_df = synth_clean(args.synth_file)
    iso = Isoacceptor2(synth_df, id_dict, args.amino_acid, df, ac=first_ac, id_part_change=args.id_part_change)
    iso.cluster_parts(args.cluster_parts, synth_name=args.synth_name, clust_id_parts=False)
    iso.chimera(args.synth_name, length_filt=args.length_filt)

    iso.cervettini_filter(start_stringency=cf_start, min_stringency=cf_min, target=cf_targ, step_size=cf_ss)

    iso.store_trnas(f'{args.output_directory}/{args.synth_name}_initial.csv')
    # iso.designs_2_fa(args.synth_name, ac=first_ac)
    # rnafold_in_parallel(iso, args.synth_name + '_para', first_ac)
    rnafold_in_parallel(iso, f'{args.output_directory}/folding/{args.synth_name}_para', first_ac)
    # iso.fold_filter(first_ac, args.synth_name + '_para_' + first_ac + '_complete_fold.out')
    iso.fold_filter(first_ac, f'{args.output_directory}/folding/{args.synth_name}_para_{first_ac}_complete_fold.out')
    for ac in args.anticodons[1:]:
        iso.change_ac([ac], args.synth_name)
        # iso.designs_2_fa(args.synth_name, ac=ac)
        # rnafold_in_parallel(iso, args.synth_name + '_para', ac)
        rnafold_in_parallel(iso, f'{args.output_directory}/folding/{args.synth_name}_para', ac)

        # iso.fold_filter(ac, args.synth_name + '_para_' + ac + '_complete_fold.out')
        iso.fold_filter(ac, f'{args.output_directory}/folding/{args.synth_name}_para_{ac}_complete_fold.out')

    iso.final_filter()
    iso.store_trnas(f'{args.output_directory}/{args.synth_name}_finalfold.csv')
    if args.select:
        iso.select(args.synth_name)
        iso.store_trnas(f'{args.output_directory}/{args.synth_name}_selected.csv')

    cluster = len(iso.trnas) > 40
    iso.cluster_select(cluster=cluster)

    success = False
    try:
        driver = webdriver.Safari()
        driver.maximize_window()
        url = 'http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi'
        final_seqs = [trna.seq[first_ac] for trna in iso.final_trnas.values()]

        for i in range(3):
            driver.execute_script("window.open('');")

        for seq, handle in zip(final_seqs, driver.window_handles):
            driver.switch_to.window(handle)
            driver.get(url)
            text_box = driver.find_element(By.ID, 'SCREEN')
            text_box.send_keys(seq)
            submit = driver.find_element(By.CLASS_NAME, 'proceed')
            submit.click()

        handles = driver.window_handles
        handles_to_remove = []
        while True:
            success = True
            handles = [handle for handle in handles if handle not in handles_to_remove]
            if len(handles) == 0:
                break
            for handle in handles:
                time.sleep(0.25)
                try:
                    driver.switch_to.window(handle)
                    driver.find_element(By.XPATH, '//*[@id="contentmain"]/h3[1]')
                    driver.execute_script("window.scrollTo(0, 300)")
                    handles_to_remove.append(handle)
                except NoSuchElementException:
                    pass
    except:
        pass

    while True:
        accept = input('Use these tRNAs? (y)es or (n)o\n> ')
        if accept.lower() == 'y':
            iso.trnas = iso.final_trnas
            iso.store_trnas(f'{args.output_directory}/{args.synth_name}_final_four.csv')
            break
        elif accept.lower() == 'n':
            print("Well I dunno man what do you want from me I'm just a machine make your own tRNAs")
            break
        else:
            print('Innapropriate value!')
            continue

    if success:
        driver.quit()






