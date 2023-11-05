"""
This module provides the code to evaluate the code generated for DotPrompts dataset
"""

import sys
import pandas as pd
import tiktoken
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import gc
import concurrent.futures

from tqdm import tqdm
from typing import Any, Dict, List
from transformers import AutoTokenizer
from eval_utils import get_first_token, tokenizer_pl, find_method_stop, get_identifiers
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_generation(ghrepo, method_d, dot_idx, row, fileContents, tokenizer_ml_obj, testcase_level_cache) -> Dict[str, Any]:
    """
    This function evaluates the generated code for a single testcase
    """

    if row['model'] == 'text-davinci-003':
        tokenizer_ml = lambda x: tokenizer_ml_obj.encode(x)
    else:
        tokenizer_ml = lambda x: tokenizer_ml_obj(x).input_ids
    
    ld: Dict[str, Any] = {}
    ld["repo"] = ghrepo

    ld.update(method_d)
    ld['method_d'] = str(method_d)

    classFileText = fileContents[method_d["classFileName"]]

    assert classFileText[dot_idx] == '.'
    prompt = classFileText[:dot_idx+1]
    
    actual_method_text = fileContents[method_d["classFileName"]][dot_idx + 1:method_d['methodStopIdx']+1]
    actual_next_line = actual_method_text.lstrip().split('\n')[0]

    assert len(actual_method_text.strip()) != 0, (ghrepo, method_d, dot_idx, actual_method_text)

    method_cache_key = (ghrepo, method_d["classFileName"], method_d['methodStartIdx'], method_d['methodStopIdx'], dot_idx)
    if method_cache_key in testcase_level_cache:
        first_token_actual_method_text = testcase_level_cache[method_cache_key]['first_token_actual_method_text']
        actual_method_text_pl_tok = testcase_level_cache[method_cache_key]['actual_method_text_pl_tok']
        actual_nextline_pl_tok = testcase_level_cache[method_cache_key]['actual_nextline_pl_tok']
    else:
        first_token_actual_method_text = get_first_token(actual_method_text)
        assert first_token_actual_method_text is not None, (ghrepo, method_d, dot_idx, actual_method_text)
        actual_method_text_pl_tok = tokenizer_pl(actual_method_text)
        assert len(actual_method_text_pl_tok) != 0
        actual_nextline_pl_tok = tokenizer_pl(actual_next_line)
        assert len(actual_nextline_pl_tok) != 0
        testcase_level_cache[method_cache_key] = {
            'first_token_actual_method_text': first_token_actual_method_text,
            'actual_method_text_pl_tok': actual_method_text_pl_tok,
            'actual_nextline_pl_tok': actual_nextline_pl_tok
        }
    
    num_tokens_required_for_first_identifier = len(tokenizer_ml(first_token_actual_method_text))
    
    ld["num_tokens_required_for_first_identifier"] = num_tokens_required_for_first_identifier

    ld["prompt"] = prompt
    ld["dot_idx"] = dot_idx

    generated_text = row['output'] if type(row['output']) == str else ''
    try:
        generated_method_text = (fileContents[method_d["classFileName"]][:method_d['methodStartIdx']+1] + find_method_stop(fileContents[method_d["classFileName"]][method_d['methodStartIdx']+1:dot_idx+1] + generated_text))[dot_idx + 1:]
    except Exception as e:
        print("Exception", e)
        print("method_d", method_d)
        print("dot_idx", dot_idx)
        print("generated_text", generated_text)
        print(row)
        raise e
    generated_next_line = generated_method_text.lstrip().split('\n')[0]
    assert len(generated_method_text.strip()) != 0, (ghrepo, method_d, dot_idx, generated_method_text, row['output'])
    assert len(generated_next_line.strip()) != 0, (ghrepo, method_d, dot_idx, generated_next_line, row['output'])

    ld["first_identifier_match"] = (get_first_token(generated_method_text) == first_token_actual_method_text)

    for upto, actual_text, gen_text in [
        ("method_close", actual_method_text, generated_method_text),
        ("nextline", actual_next_line, generated_next_line)
    ]:
        ld[f"actual_text_upto_{upto}"] = actual_text
        ld[f"gen_text_upto_{upto}"] = gen_text
        
        ld[f"exact_match_string_upto_{upto}"] = (actual_text == gen_text)

        actual_text_pl_tokenized = tokenizer_pl(actual_text)
        gen_text_pl_tokenized = tokenizer_pl(gen_text)

        match_len = 0
        for idx in range(min(len(actual_text_pl_tokenized), len(gen_text_pl_tokenized))):
            if actual_text_pl_tokenized[idx] != gen_text_pl_tokenized[idx]:
                break
            match_len = idx + 1
        ld[f"num_pl_tokens_exact_match_upto_{upto}"] = match_len
        ld[f"perc_pl_tokens_exact_match_upto_{upto}"] = match_len / len(actual_text_pl_tokenized)
        ld[f"num_pl_tokens_actual_text_upto_{upto}"] = len(actual_text_pl_tokenized)

        ld[f"exact_match_tokenstream_upto_{upto}"] = (actual_text_pl_tokenized == gen_text_pl_tokenized)

        actual_text_sequence_of_identifiers = get_identifiers(actual_text)
        gen_text_sequence_of_identifiers = get_identifiers(gen_text)

        match_len = 0
        for idx in range(min(len(actual_text_sequence_of_identifiers), len(gen_text_sequence_of_identifiers))):
            if actual_text_sequence_of_identifiers[idx] != gen_text_sequence_of_identifiers[idx]:
                break
            match_len = idx + 1
        ld[f"num_ord_identifiers_exact_match_upto_{upto}"] = match_len
        ld[f"perc_ord_identifiers_exact_match_upto_{upto}"] = match_len / len(actual_text_sequence_of_identifiers) if len(actual_text_sequence_of_identifiers) > 0 else None
        ld[f"num_ord_identifiers_actual_text_upto_{upto}"] = len(actual_text_sequence_of_identifiers)

        ld[f"exact_match_identifier_tokenstream_upto_{upto}"] = (actual_text_sequence_of_identifiers == gen_text_sequence_of_identifiers)
        
        assert not(ld[f"exact_match_string_upto_{upto}"]) or ld[f"exact_match_tokenstream_upto_{upto}"]
        
        if len(actual_text) == 0:
            raise Exception("Empty actual text: " + actual_text, ghrepo, method_d, dot_idx, row)

        if len(actual_text_pl_tokenized) == 0:
            raise Exception("Empty actual text pl tokenized: " + actual_text, ghrepo, method_d, dot_idx, row)
    
    ld.update(row)

    return ld

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k
    
    As defined in https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k: 
        return 1.0
    return 1.0 - np.prod(1.0 - k/np.arange(n - c + 1, n + 1))

def score_at_k(n: int, k: int, vals: List[float]):
    """
    Calculates the score@k metric as defined in the paper appendix D for a single testcase

    :param n: total number of samples
    :param k: k in score@k
    :param vals: list of scores obtained for this testcase by the model (in descending order)
    """
    assert len(vals) == n
    assert n >= k, (n, k)
    assert all(vals[i-1] >= vals[i] for i in range(1, len(vals))), (n, k, vals)
    
    isum = 0
    for i in range(1, n-k+2):
        # i ranges from 1 to n-k+1
        isum += (vals[i-1]*math.comb(n-i, k-1))
    
    return isum/math.comb(n, k)

def calculate_score_at_k_for_all_testcases_for_one_configuration(x, score_name, k=None):
    """
    Calculates the score@k metric as defined in the paper appendix D 
    across all testcases for a single configuration
    """
    gb = x.groupby(['repo', 'classFileName', 'methodStartIdx', 'methodStopIdx', 'method_d']).apply(lambda y: score_at_k(len(y[score_name]), len(y[score_name]) if k is None else k, y[score_name].sort_values(ascending=False).tolist()))
    avg_f1_rate = gb.mean()*100
    return pd.Series({"score" : avg_f1_rate})

def calculate_score_at_k_for_all_testcases_and_configurations(dfx: pd.DataFrame, score_name: str, k: int):
    """
    Calculates the score@k metric as defined in the paper appendix D 
    for all testcases for all configurations, by grouping by configuration
    """
    res = dfx.groupby('configuration').apply(lambda x: calculate_score_at_k_for_all_testcases_for_one_configuration(x, score_name, k=k)['score'])
    return res

def generate_plots(scores, scores_df, colors_d, scores_to_abbrevs, results_dir):
    """
    Generates plots for score@k and relative change in score@6 for all metrics
    """
    for score_name, score_title in scores.items():
        for configuration_subset_name, configurations in [
            ('models', ['CG-350M', 'CG-350M-MGD', 'CG-2B', 'CG-2B-MGD', 'CG-6B', 'CG-6B-MGD', 'SC', 'SC-MGD', 'TD-3', 'TD-3-MGD']),
            ('prompt', ['SC', 'SC-MGD', 'SC-classExprTypes', 'SC-classExprTypes-MGD', 'SC-RLPG', 'SC-RLPG-MGD', 'TD-3']),
            ('fim', ['SC', 'SC-MGD', 'SC-FIM', 'SC-FIM-MGD', 'SC-classExprTypes', 'SC-classExprTypes-MGD', 'SC-FIM-classExprTypes', 'SC-FIM-classExprTypes-MGD', 'TD-3']),
        ]:
            xdf = scores_df[score_name].loc[configurations]
            fig, ax = plt.subplots(1,1, figsize=(10, 10))
            
            for config in configurations:
                xdf_temp = xdf.loc[[config]]
                format_d = {}
                if '-MGD' in config or '-DPI' in config:
                    format_d['linestyle'] = 'solid'
                else:
                    format_d['linestyle'] = 'dashed'
                
                if '-FIM-classExprTypes' in config:
                    format_d['marker'] = 'd'
                    format_d['fillstyle'] = 'full'
                    format_d['markersize'] = 12
                elif '-classExprTypes' in config:
                    format_d['marker'] = 'X'
                    format_d['fillstyle'] = 'full'
                    format_d['markersize'] = 12
                elif '-FIM' in config:
                    format_d['marker'] = 's'
                    format_d['markersize'] = 10
                elif '-RLPG' in config:
                    format_d['marker'] = 'o'
                    format_d['fillstyle'] = 'full'
                    format_d['markersize'] = 12
                elif '-Random' in config:
                    format_d['linestyle'] = 'dotted'
                    format_d['marker'] = 'v'
                    format_d['fillstyle'] = 'full'
                    format_d['markersize'] = 10
                else:
                    format_d['marker'] = None
                
                xdf_temp.T.plot(color=colors_d, linewidth=3, grid=True, ax=ax, **format_d)

            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
            # ax.set_title(f"{score_title} across score@k", fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            # ax.get_legend().remove()
            fig.savefig(os.path.join(results_dir, "figures", "score_at_k", f"fig_{scores_to_abbrevs[score_name]}_{configuration_subset_name}"), bbox_inches='tight')
            # fig.savefig(os.path.join(results_dir, "figures", "score_at_k", f"fig_{score_name}_{configuration_subset_name}"), bbox_inches='tight')
            del fig, ax
            gc.collect()

            fig, ax = plt.subplots(1,1, figsize=(10, 10))
            
            xdfi = pd.Series([
                (xdf["score@6"][config2] - xdf["score@6"][config1])/xdf["score@6"][config1] 
                for config1 in [c for c in configurations if '-MGD' not in c] 
                for config2 in [c for c in configurations if '-MGD' in c]
            ], index=pd.MultiIndex.from_tuples([
                (config1, config2) 
                for config1 in [c for c in configurations if '-MGD' not in c] 
                for config2 in [c for c in configurations if '-MGD' in c]
            ]))
            xdfi = xdfi.unstack(-1).T*100
            xdfi = xdfi[sorted(xdfi.columns, key=lambda x: configurations.index(x))]
            xdfi = xdfi.loc[sorted(xdfi.index, key=lambda x: configurations.index(x))]

            ax_heatmap = sns.heatmap(xdfi, annot=True, center=0, cmap=sns.color_palette("coolwarm_r", as_cmap=True), ax=ax, fmt=".2f") # , 
            ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation = 45)
            ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation = 45)
            for t in ax_heatmap.texts:
                t.set_text(t.get_text() + "%")
            
            # ax.set_title(f"Relative change in {score_title} (score@6)", fontweight='bold')

            fig.savefig(os.path.join(results_dir, "figures", "rel_changes", f"fig_{scores_to_abbrevs[score_name]}_{configuration_subset_name}_heatmap"), bbox_inches='tight')
            del fig, ax, ax_heatmap, xdfi
            gc.collect()

def generate_plots_id_complexity(fim_by_identifier_complexity_scores, colors_d, results_dir):
    """
    Generates plots for Next Identifier Match (NIM) score@6 and relative change in NIM score@6
    across identifier complexity - appendix F
    """
    for configuration_subset_name, configurations in [
        ('models', ['CG-350M', 'CG-350M-MGD', 'CG-2B', 'CG-2B-MGD', 'CG-6B', 'CG-6B-MGD', 'SC', 'SC-MGD', 'TD-3', 'TD-3-MGD']),
        ('prompt', ['SC', 'SC-MGD', 'SC-classExprTypes', 'SC-classExprTypes-MGD', 'SC-RLPG', 'SC-RLPG-MGD', 'TD-3']),
        ('fim', ['SC', 'SC-MGD', 'SC-FIM', 'SC-FIM-MGD', 'SC-classExprTypes', 'SC-classExprTypes-MGD', 'SC-FIM-classExprTypes', 'SC-FIM-classExprTypes-MGD', 'TD-3']),
    ]:
        xdf = fim_by_identifier_complexity_scores[configurations]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

        ax.grid()

        for config in configurations:
            xdf_temp = xdf[config]
            format_d = {}
            if '-MGD' in config or '-DPI' in config:
                format_d['linestyle'] = 'solid'
            else:
                format_d['linestyle'] = 'dashed'
            
            if '-FIM-classExprTypes' in config:
                format_d['marker'] = 'd'
                format_d['fillstyle'] = 'full'
                format_d['markersize'] = 12
            elif '-classExprTypes' in config:
                format_d['marker'] = 'X'
                format_d['fillstyle'] = 'full'
                format_d['markersize'] = 12
            elif '-FIM' in config:
                format_d['marker'] = 's'
                format_d['markersize'] = 10
            elif '-RLPG' in config:
                format_d['marker'] = 'o'
                format_d['fillstyle'] = 'full'
                format_d['markersize'] = 12
            elif '-Random' in config:
                format_d['linestyle'] = 'dotted'
                format_d['marker'] = 'v'
                format_d['fillstyle'] = 'full'
                format_d['markersize'] = 10
            else:
                format_d['marker'] = None
            
            xdf_temp.plot(color=colors_d, linewidth=3, grid=True, ax=ax, **format_d)

        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        # ax.set_title('Next Identifier Match across identifier complexity (score@6)', fontweight='bold')
        
        fig.savefig(os.path.join(results_dir, "figures", "score_at_k", f"fig_next_identifier_match_{configuration_subset_name}_id_complexity"), bbox_inches='tight')
        del fig, ax
        gc.collect()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        xdfi = pd.Series([
            (xdf.loc[xdf.index[-1]][config2] - xdf.loc[xdf.index[-1]][config1])/xdf.loc[xdf.index[-1]][config1] 
            for config1 in [c for c in configurations if '-MGD' not in c] 
            for config2 in [c for c in configurations if '-MGD' in c]], index=pd.MultiIndex.from_tuples([
                (config1, config2) 
                for config1 in [c for c in configurations if '-MGD' not in c] 
                for config2 in [c for c in configurations if '-MGD' in c]
            ]))

        xdfi = xdfi.unstack(-1).T*100
        xdfi = xdfi[sorted(xdfi.columns, key=lambda x: configurations.index(x))]
        xdfi = xdfi.loc[sorted(xdfi.index, key=lambda x: configurations.index(x))]

        ax_heatmap = sns.heatmap(xdfi, annot=True, center=0, cmap=sns.color_palette("coolwarm_r", as_cmap=True), square=False, ax=ax, fmt=".2f")
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation = 45)
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation = 45)
        for t in ax_heatmap.texts:
            t.set_text(t.get_text() + "%")

        # ax.set_title(f"Relative change in Next Identifier Match {xdf.index[-1]} (score@6)", fontweight='bold')
        
        fig.savefig(os.path.join(results_dir, "figures", "rel_changes", f"fig_next_identifier_match_{configuration_subset_name}_id_complexity_heatmap"), bbox_inches='tight')
        del fig, ax, ax_heatmap, xdfi
        gc.collect()

def dump_scores_and_plots_to_directory(df, results_dir):
    """
    Given the dataframe containing the evaluation results, this function plots the aggregate scores and graphs
    """
    df.loc[:, 'method_d'] = df.apply(lambda x: str({**eval(x['method_d']), 'dot_idx': x['dot_idx']}), axis=1)

    assert df['compilationSucceeded'].isna().sum() == 0
    assert df['perc_ord_identifiers_exact_match_upto_method_close'].isna().sum() == 0
    assert df['perc_ord_identifiers_exact_match_upto_nextline'].isna().sum() == 0
    assert df['perc_pl_tokens_exact_match_upto_method_close'].isna().sum() == 0
    assert (df.groupby(['method_d', 'configuration']).apply(lambda x: len(x)) == 6).all()
    assert len(df.groupby('configuration').apply(lambda x: len(x)).unique()) == 1
    assert len(df[['repo', 'classFileName', 'methodStartIdx', 'methodStopIdx', 'dot_idx']].drop_duplicates())*6*len(df['configuration'].unique()) == len(df)

    scores = {
        'compilationSucceeded' : "Compilation Rate (CR)",
        'first_identifier_match' : "Next Identifier Match (NIM)",
        'perc_ord_identifiers_exact_match_upto_method_close' : "Identifier Sequence Match (ISM)",
        'perc_pl_tokens_exact_match_upto_method_close' : "Prefix Match (PM)"
    }

    scores_to_abbrevs = {
        'compilationSucceeded' : "CR",
        'first_identifier_match' : "NIM",
        'perc_ord_identifiers_exact_match_upto_method_close' : "ISM",
        'perc_pl_tokens_exact_match_upto_method_close' : "PM"
    }

    plt.rcParams["figure.figsize"] = (10,10)

    assert not df[[k for k in scores]].isna().any().any()

    scores_xdf = {}
    with ProcessPoolExecutor(10) as executor:
        score_futures = {
            executor.submit(calculate_score_at_k_for_all_testcases_and_configurations, df[['configuration', 'repo', 'classFileName', 'methodStartIdx', 'methodStopIdx', 'method_d', score_name]], score_name, k) : (score_name, k)
            for score_name in scores for k in [1, 2, 3, 4, 5, 6] if (score_name, f"score@{k}") not in scores_xdf
        }
        for future in tqdm(as_completed(score_futures), total=len(score_futures)):
            score_name, k = score_futures[future]
            score_result = future.result()
            scores_xdf[(score_name, f"score@{k}")] = score_result

    scores_df = pd.DataFrame(scores_xdf)
    del scores_xdf
    scores_df = scores_df[sorted(scores_df.columns)]
    scores_df_to_write = scores_df.copy()
    table_order = [
        'CG-350M', 'CG-350M-MGD',
        'CG-2B', 'CG-2B-MGD', 
        'CG-6B', 'CG-6B-MGD',
        'SC', 'SC-MGD', 'SC-classExprTypes', 'SC-classExprTypes-MGD', 'SC-RLPG', 'SC-RLPG-MGD', 'SC-FIM', 'SC-FIM-MGD', 'SC-FIM-classExprTypes',  'SC-FIM-classExprTypes-MGD',
        'TD-3', 'TD-3-MGD'
    ]
    scores_df_to_write = scores_df_to_write.loc[table_order]
    scores_df_to_write.columns = pd.MultiIndex.from_tuples([(scores[x[0]], x[1]) for x in scores_df_to_write.columns])

    print("Writing evaluation result for all metrics to", os.path.join(results_dir, "all_metrics_table.md"))
    scores_df_to_write.to_markdown(os.path.join(results_dir, "all_metrics_table.md"), floatfmt=".2f")
    
    print("Writing evaluation result for all metrics to", os.path.join(results_dir, "all_metrics_table.csv"))
    scores_df_to_write.to_csv(os.path.join(results_dir, "all_metrics_table.csv"))

    colors_d = {
        'CG-350M' : '#439034',
        'CG-350M-MGD' : '#439034',
        
        'CG-2B' : '#1f628e',
        'CG-2B-MGD' : '#1f628e', 
        
        'CG-6B': '#414141',
        'CG-6B-MGD': '#414141', 
        
        'TD-3' : '#961EE1',
        'TD-3-MGD' : '#961EE1',
        
        'SC': '#DE0F3F',
        'SC-MGD': '#DE0F3F',
        'SC-classExprTypes' : '#DE0F3F', 
        'SC-classExprTypes-MGD' : '#DE0F3F', 
        'SC-FIM' :  '#DE0F3F', 
        'SC-FIM-MGD' : '#DE0F3F',
        'SC-FIM-classExprTypes' :  '#DE0F3F', 
        'SC-FIM-classExprTypes-MGD' : '#DE0F3F',

        'SC-RLPG' :  '#DE0F3F',
        'SC-RLPG-MGD' : '#DE0F3F'
    }
    
    os.makedirs(os.path.join(results_dir, "figures", "score_at_k"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures", "rel_changes"), exist_ok=True)

    generate_plots(scores, scores_df, colors_d, scores_to_abbrevs, results_dir)

    df = df.assign(tokenizer_name=df['model'].map({
        'Salesforce/codegen-6B-multi': 'CG',
        'Salesforce/codegen-350M-multi': 'CG',
        'text-davinci-003': 'TD3', 
        'Salesforce/codegen-2B-multi': 'CG',
        'bigcode/santacoder': 'SC'
    }))

    def calc_identifier_complexity(x):
        y = list(x['num_tokens_required_for_first_identifier'].unique())
        assert len(y) == 1, x
        return y[0]

    xdf = df.groupby(['method_d', 'tokenizer_name']).apply(calc_identifier_complexity).unstack(-1)
    num_llm_toks_by_testcase = xdf.mean(axis=1)
    method_to_num_tokens_first_identifier = pd.cut(num_llm_toks_by_testcase, bins=[1, 2, 3, 4, max(5, math.ceil(num_llm_toks_by_testcase.max())+1)], right=False)
    assert not method_to_num_tokens_first_identifier.isna().any()
    df = df.assign(num_tokens_required_for_first_identifier_grouped=df.apply(lambda x: method_to_num_tokens_first_identifier[x['method_d']], axis=1))

    # Generate plot for distribution of methods by most complex identifier
    plt.rcParams["figure.figsize"] = (10,10)
    xdf = df.groupby(['repo', 'classFileName', 'methodStartIdx', 'methodStopIdx']).apply(lambda x: x['num_tokens_required_for_first_identifier_grouped'].max()).value_counts()
    xdf = xdf[sorted(xdf.index)]
    # Obtained from https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f
    fig1, ax1 = plt.subplots()
    (xdf).plot.pie(
        wedgeprops={"linewidth": 1, "edgecolor": "white"}, 
        frame=False, 
        autopct=lambda x: '{:.0f} ({:.2f}%)'.format(x*xdf.sum()/100, x),
        startangle=90,
        shadow=True,
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
        explode=tuple([0.0 for i in range(len(xdf)-1)] + [0.1]),
        ax=ax1
    )
    ax1.set_title("Distribution of methods by most complex identifier across DotPrompts dataset")
    fig1.savefig(os.path.join(results_dir, "figures", f"fig_method_dist_by_max_identifier_complexity_dist"), bbox_inches='tight')
    del fig1, ax1
    gc.collect()

    fim_by_identifier_complexity_scores = df.groupby(
        ["num_tokens_required_for_first_identifier_grouped", 'configuration']
    ).apply(
        lambda x: calculate_score_at_k_for_all_testcases_for_one_configuration(x, 'first_identifier_match', k=6)['score']
    ).unstack(-1)

    generate_plots_id_complexity(fim_by_identifier_complexity_scores, colors_d, results_dir)

    print("Writing summary in", os.path.join(results_dir, "Report.md"))

    with open(os.path.join(results_dir, "Report.md"), "w") as f:
        f.write(f"""# Results for "Guiding Language Models of Code with Global Context using Monitors"
## Summary of Results (Table 1 in the paper)
{scores_df_to_write.to_markdown(floatfmt=".2f")}

## Effect of MGD on Models across Parameter Scale and Architectures (Ref. section 4.1)
""")
        for score_name, score_title in scores.items():
            f.write(f"""**{score_title} score@k** | **Relative Change in {score_title} (score@6)**
:-------------------------:|:-------------------------:
![]({os.path.join("figures", "score_at_k", "fig_" + scores_to_abbrevs[score_name] + "_models.png")})  |  ![]({os.path.join("figures", "rel_changes", "fig_" + scores_to_abbrevs[score_name] + "_models_heatmap.png")})
""")
        
        f.write(f"""## Effect of MGD and Prompt Augmentation Strategies (Ref. section 4.2)
""")
        for score_name, score_title in scores.items():
            f.write(f"""**{score_title} score@k** | **Relative Change in {score_title} (score@6)**
:-------------------------:|:-------------------------:
![]({os.path.join("figures", "score_at_k", "fig_" + scores_to_abbrevs[score_name] + "_prompt.png")})  |  ![]({os.path.join("figures", "rel_changes", "fig_" + scores_to_abbrevs[score_name] + "_prompt_heatmap.png")})
""")
        
        f.write(f"""## Effect of MGD on Fill-in-the-middle (FIM) Decoding (Ref. section 4.3 and appendix E)
""")
        for score_name, score_title in scores.items():
            f.write(f"""**{score_title} score@k** | **Relative Change in {score_title} (score@6)**
:-------------------------:|:-------------------------:
![]({os.path.join("figures", "score_at_k", "fig_" + scores_to_abbrevs[score_name] + "_fim.png")})  |  ![]({os.path.join("figures", "rel_changes", "fig_" + scores_to_abbrevs[score_name] + "_fim_heatmap.png")})
""")
        
        f.write(f"""## Effect of Identifier Complexity on Next Identifier Match (Ref. section 4.4 and appendix F)
### Distribution of methods by most complex identifier in DotPrompts
![]({os.path.join("figures", "fig_method_dist_by_max_identifier_complexity_dist.png")})

### Next Identifier Match (NIM) score@6 by identifier complexity
""")
        for configuration_subset_name in ['models', 'prompt', 'fim']:
            f.write(f"""**NIM score@6** | **Relative Change in NIM (score@6)**
:-------------------------:|:-------------------------:
![]({os.path.join("figures", "score_at_k", "fig_next_identifier_match_" + configuration_subset_name + "_id_complexity.png")})  |  ![]({os.path.join("figures", "rel_changes", "fig_next_identifier_match_" + configuration_subset_name + "_id_complexity_heatmap.png")})
""")
        print("Written summary in", os.path.join(results_dir, "Report.md"))

def calculate_evaluation_metrics_for_inference_results(inference_results, fileContents):
    """
    This function receives a subset of the inference results for a single repository 
    and calculates the evaluation metrics for each testcase
    """
    data = []
    tokenizers = {}
    testcase_level_cache = {}
    indices = []

    for idx, row in inference_results.iterrows():
        if row['model'] not in tokenizers:
            if row['model'] == 'text-davinci-003':
                enc = tiktoken.encoding_for_model('text-davinci-003')
                tokenizers[row['model']] = enc
                del enc
            else:
                tokzer = AutoTokenizer.from_pretrained(row['model'])
                tokenizers[row['model']] = tokzer
                del tokzer
        
        method_d = {
            'classFileName' : row['classFileName'], 
            'methodStartIdx' : row['methodStartIdx'], 
            'methodStopIdx' : row['methodStopIdx'],
        }

        data.append(evaluate_generation(row['repo'], method_d, row['dot_idx'], row, fileContents, tokenizers[row['model']], testcase_level_cache))
        indices.append(idx)
        testcase_level_cache = {}
    
    df = pd.DataFrame(data, index=indices)
    del data, tokenizers, testcase_level_cache
    return df

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 eval_results.py <path to inference results - csv> <path to PragmaticCode filecontents - json> <path to output directory>")
        exit(1)
    
    inference_results_path = sys.argv[1]
    filecontents_path = sys.argv[2]
    output_dir = sys.argv[3]

    if os.path.exists(output_dir):
        print("Output directory already exists. Please rename or delete it and try again. This is to ensure that results are not overwritten")
        exit(1)

    os.makedirs(output_dir, exist_ok=False)

    inference_results = pd.read_csv(inference_results_path, encoding='utf-8')
    
    with open(filecontents_path, "r") as f:
        fileContentsByRepo = json.load(f)
    
    futures = {}
    # TODO: Make the following value (20) configurable
    with ProcessPoolExecutor(20) as executor:
        for ghrepo in tqdm(inference_results['repo'].unique()):
            futures[executor.submit(calculate_evaluation_metrics_for_inference_results, inference_results.loc[inference_results['repo'] == ghrepo], fileContentsByRepo[ghrepo])] = ghrepo

        chunks = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ghrepo = futures[future]
            df = future.result()
            chunks.append(df)
            del df
            gc.collect()
    
    df = pd.concat(chunks)

    dump_scores_and_plots_to_directory(df, output_dir)