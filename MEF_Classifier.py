import numpy as np

def mar_trainOnline(data_input, data_target, num_vars, max_cluster, half_life, threshold_mf, min_rule_weight, ablation, evo, interpret,
                     mode, ds_W, base_lambda, tau_match, score_q, rule_partial_match, start_test, start_external, *args):
    system = {}
    print(f"data input shape: {data_input.shape}, data_target shape: {data_target.shape}")
    if len(args) == 0:
        print(f'Create network {1}')
        system['total_network'] = 1
        system['net'] = {
            'name': 'eMFIS',
            'type': 'mamdani',
            'andMethod': 'min',
            'orMethod': 'max',
            'defuzzMethod': 'centroid',
            'impMethod': 'min',
            'aggMethod': 'max',
            'weight': 1,
            'lastLearned': 0,
            'created': 1,
            'winner': 0,
            'id': 1,
            'max_cluster': max_cluster,
            'half_life': half_life,
            'forgettor': 0.5 ** (1 / half_life),
            'forgettor_pos': 0.5 ** (1 / (half_life * 5)),
            'protect_T': -1,
            'num_active_rules': 0,
            'threshold_mf': threshold_mf,
            'min_rule_weight': min_rule_weight,
            'predicted': np.zeros((data_target.shape[0], data_target.shape[1])),
            'ruleCount': np.zeros((data_target.shape[0], 1)),
            'outputClusterCount': np.zeros((data_target.shape[0], 1)),
            'spatioTemporal': np.zeros((data_target.shape[0], 1)),
            # ---- classification readout config ----
            'clf_mode': mode,
            'ds_W': float(ds_W),          
            'base_lambda': float(base_lambda),  

            # running prior (EWMA) for ds-mode base rate a
            'prior_pos': 0.0,
            'prior_total': 0.0,
            'base_rate': 0.5,
        }
        system['dataProcessed'] = 0
        system['predicted'] = np.zeros((data_target.shape[0], data_target.shape[1]))
        
        system['net']['uncertainty'] = np.zeros_like(system['net']['predicted'], dtype=float)
        system['net']['num_vars'] = int(num_vars)
        system['net']['base_rate_hist'] = np.zeros((data_target.shape[0], 1), dtype=float)
        
        system['net']['half_life_prior'] = 3000          
        system['net']['gamma_prior'] = 0.5 ** (1 / system['net']['half_life_prior'])
        system['net']['w_max'] = 20.0
        system['net']['pos_w'] = 1.0
        system['net']['use_pos_weight'] = True 

        system['net']['tau_match'] = tau_match  
        system['net']['score_q'] = score_q  
        system['net']['use_partial_match'] = rule_partial_match  

        k_rules = 3 if interpret else 1
        system['top_k_rules'] = {
            'train': [], 'test': [], 'external': [],
            'train_pos': [], 'test_pos': [], 'external_pos': [],
            'train_neg': [], 'test_neg': [], 'external_neg': [],
            'train_invoke': [], 'test_invoke': [], 'external_invoke': [],
        }

    if evo:
        stopEvoAt = evo
        print('set evo limit to ', stopEvoAt)
    else:
        stopEvoAt = data_target.shape[0] + 100
        print('no evo limit set')

    for i in range(data_target.shape[0]):
        system['dataProcessed'] += 1
        current_count = system['dataProcessed']
        
        print(f'Processing {current_count}')

        if mode == 'ds':
            pred, info = mar_cri(
                data_input[i, :], system['net'], current_count,
                ablation=ablation, interpret=True, mode=mode
            )
            system['net']['predicted'][current_count - 1, :] = pred
            system['net']['uncertainty'][current_count - 1, 0] = float(info.get('u', 0.0))
        else:
            pred = mar_cri(
                data_input[i, :], system['net'], current_count,
                ablation=ablation, interpret=False, mode=mode
            )
            system['net']['predicted'][current_count - 1, :] = pred
            system['net']['uncertainty'][current_count - 1, 0] = 0.0

            
        curr_input = data_input[i,:]
        pred = system['net']['predicted'][current_count - 1, :]
        actual = data_target[i,:]
        # print(f'input: {curr_input}')
        # print(f'pred: {pred}')
        # print(f'actual: {actual}')
        

        if np.isnan(system['net']['predicted'][current_count - 1, :]).any():
            system['net']['predicted'][current_count - 1, :] = 0
            system['predicted'][current_count - 1, :] = 0
            system['net']['uncertainty'][current_count - 1, 0] = 0.0
            continue
        else:
            system['predicted'][current_count - 1, :] = system['net']['predicted'][current_count - 1, :]

        # XXXXXXXXXXX ONLINE LEARNING XXXXXXXXXXX
        if len(args) > 0:
            continue
        
        if current_count <= stopEvoAt:

            net = system['net']
            
        
            net = mar_online_mf4(net, data_input[i, :], data_target[i, :], current_count, update_output_mf=False) 
            net = mar_online_rule(net, data_input[i, :], data_target[i, :], current_count) 
            net = mar_pseudo_prune_rule(net, current_count) 

            net, merged = mar_clean_mf(net, merge_output=False)

            if merged:
                net = mar_clean_pop(net, current_count)
                net = mar_pseudo_prune_rule(net, current_count)

            net['ruleCount'][current_count - 1] = net['num_active_rules']
            net['outputClusterCount'][current_count - 1] = 0
            net['spatioTemporal'][current_count - 1] = 0

            system['net'] = net
            system['net']['base_rate_hist'][current_count-1, 0] = system['net'].get('base_rate', 0.5)
        else:
            print('no learning process due to reaching evo limit.')

        if current_count == start_test - 1:
            extract_snapshot(system, current_count, 'train', k_rules)

        elif (start_external is not None and current_count == start_external - 1) or \
             (start_external is None and current_count == len(data_target)):
            extract_snapshot(system, current_count, 'test', k_rules)

        elif start_external is not None and current_count == len(data_target):
            extract_snapshot(system, current_count, 'external', k_rules)

        

    return system

import copy
def extract_snapshot(system, current_count, target_phase_name, k_rules=1):
    # all_active_rules = [r for r in system['net']['rule'] if r.get('active', 1) == 1]
    pos_rules = [r for r in system['net']['rule'] if r['pos_evi'] > 0]
    neg_rules = [r for r in system['net']['rule'] if r['neg_evi'] > 0]
    invoke_rules = [r for r in system['net']['rule'] if r.get('num_invoked', 0) > 0]
    

    current_top_k_all = sorted(system['net']['rule'], key=lambda x: x['weight'], reverse=True)[:k_rules]
    current_top_k_pos = sorted(pos_rules, key=lambda x: x['weight'], reverse=True)[:k_rules]
    current_top_k_neg = sorted(neg_rules, key=lambda x: x['weight'], reverse=True)[:k_rules]
    current_top_k_invoke = sorted(invoke_rules, key=lambda x: x['num_invoked'], reverse=True)[:k_rules]
    
    system['top_k_rules'][target_phase_name] = copy.deepcopy(current_top_k_all)
    system['top_k_rules'][f'{target_phase_name}_pos'] = copy.deepcopy(current_top_k_pos)
    system['top_k_rules'][f'{target_phase_name}_neg'] = copy.deepcopy(current_top_k_neg)
    system['top_k_rules'][f'{target_phase_name}_invoke'] = copy.deepcopy(current_top_k_invoke)

    mf_snapshot = copy.deepcopy(system['net']['input'])
    system['top_k_rules'][target_phase_name].append(mf_snapshot)
    system['top_k_rules'][f'{target_phase_name}_pos'].append(mf_snapshot)
    system['top_k_rules'][f'{target_phase_name}_neg'].append(mf_snapshot)
    system['top_k_rules'][f'{target_phase_name}_invoke'].append(mf_snapshot)
    
    print(f"Snapshot taken for {target_phase_name} at step {current_count}")

def mar_online_mf4(net, data_input, data_target, current_count, update_output_mf=True):
    stop_phase1_at = 10
    default_width_input = 0.1
    default_width_output = 0.01
    
    phase = 2 if current_count >= stop_phase1_at else 1

    num_attributes = data_input.shape[0]
    num_outputs = data_target.shape[0]
    forgettor = net['forgettor']
    
    # First Data Handling
    if current_count == 1:
        net['input'] = {}
        for i in range(num_attributes):
            net['input'][i] = {
                'name': f'X_{i+1}',
                'range': [data_input[i], data_input[i]],
                'min': data_input[i],
                'max': data_input[i],
                'phase': 0,
                'spatio_temporal_dist': 0,
                'sum_square': 0,
                'sum': 0,
                'currency': 0,
                'last_mf_won': 0,
                'age': [0,],
                'mf': [{
                    'name': 'mf1',
                    'type': 'gauss2mf',
                    'num_invoked': 1,
                    'params': [default_width_input, data_input[i], default_width_input, data_input[i]],
                    'stability': 0,
                    'tw_sum': 0,
                    'mf_currency': 0,
                    'created_at': 1
                }]
            }

        net['output'] = {}
        for i in range(num_outputs):
            net['output'][i] = {
                'name': f'Y_{i+1}',
                'range': [data_target[i], data_target[i]],
                'min': data_target[i],
                'max': data_target[i],
                'phase': 0,
                'spatio_temporal_dist': 0,
                'sum_square': 0,
                'sum': 0,
                'currency': 0,
                'last_mf_won': 0,
                'age': [0,],
                'mf': [{
                    'name': 'mf1',
                    'type': 'gauss2mf',
                    'num_invoked': 1,
                    'params': [default_width_output, data_target[i], default_width_output, data_target[i]],
                    'stability': 0,
                    'tw_sum': 0,
                    'mf_currency': 0,
                    'created_at': 1
                }]
            }
    
    # Update input dimensions
    for i in range(num_attributes):
        if data_input[i] != np.inf and data_input[i] != -np.inf:
            net['input'][i]['sum_square'] = (net['input'][i]['sum_square'] * forgettor) + data_input[i] ** 2
            net['input'][i]['sum'] = (net['input'][i]['sum'] * forgettor) + data_input[i]
        else:
            net['input'][i]['sum_square'] = (net['input'][i]['sum_square'] * forgettor) 
            net['input'][i]['sum'] = (net['input'][i]['sum'] * forgettor)
        net['input'][i]['currency'] = (net['input'][i]['currency'] * forgettor) + 1
        
        # net['input'][i]['spatio_temporal_dist'] = 4 * np.sqrt((net['input'][i]['sum_square'] / net['input'][i]['currency']) - (net['input'][i]['sum'] / net['input'][i]['currency']) ** 2)
        cur = net['input'][i]['currency']
        ex2 = net['input'][i]['sum_square'] / cur
        ex  = net['input'][i]['sum'] / cur
        var = ex2 - ex**2  
        eps = 1e-10
        if var < 0 and var > -eps:
            pass  # small negative variance due to numerical errors
        elif var <= -eps:
            print(
                "[sqrt NEGATIVE (unexpected)] (input) "
                f"count={current_count}, dim={i}, "
                f"x={data_input[i]}, var={var:.6e}, "
                f"ex2={ex2:.6e}, ex={ex:.6e}, cur={cur:.6e}, "
                f"sum_sq={net['input'][i]['sum_square']:.6e}, sum={net['input'][i]['sum']:.6e}, "
                f"forgettor={net['forgettor']}"
            )
            raise FloatingPointError("Variance became significantly negative.")


        var = max(var, 0.0)
        net['input'][i]['spatio_temporal_dist'] = 4.0 * np.sqrt(var)

        drift_detected = False
        if current_count > 1:
            drift_detected, neuron = mar_calculate_age2(net['input'][i], data_input[i], net['threshold_mf'], current_count, stop_phase1_at)
            net['input'][i] = neuron

        if drift_detected:
            net['input'][i] = mar_generate_mf3(net['input'][i], data_input[i], current_count, default_width_input, phase, net['rule'], i, 'input')


        num_mf = len(net['input'][i]['mf'])

        mf_values = np.zeros(num_mf)
        for j in range(num_mf):
            membership = gauss2mf(data_input[i], net['input'][i]['mf'][j]['params'])
            mf_values[j] = membership
            centroid = (net['input'][i]['mf'][j]['params'][1] + net['input'][i]['mf'][j]['params'][3]) / 2
            if centroid == np.inf or centroid == -np.inf:
                centroid_width = 0
            else:
                centroid_width = net['input'][i]['mf'][j]['params'][3] - centroid
            curr_stability = net['input'][i]['mf'][j]['stability'] * forgettor
            if data_input[i] == np.inf or data_input[i] == -np.inf:
                centroid = centroid
            else:
                centroid = (curr_stability * centroid + membership * data_input[i]) / (curr_stability + membership)
            net['input'][i]['mf'][j]['params'][1] = centroid - centroid_width
            net['input'][i]['mf'][j]['params'][3] = centroid + centroid_width
            net['input'][i]['mf'][j]['stability'] = curr_stability + membership

        max_index = np.argmax(mf_values)
        net['input'][i]['last_mf_won'] = max_index

        # Update variance
        Unk_exist = False
        Pad_exist = False
        for j in range(num_mf - 1):
            if net['input'][i]['mf'][j+1]['params'][1] == np.inf:
                Unk_exist = True
                break
            elif net['input'][i]['mf'][j]['params'][1] == -np.inf:
                Pad_exist = True
                continue
            else:
                stability_gap = abs(net['input'][i]['mf'][j+1]['params'][1] - net['input'][i]['mf'][j]['params'][3]) / (net['input'][i]['mf'][j]['stability'] + net['input'][i]['mf'][j+1]['stability'])
            net['input'][i]['mf'][j]['params'][2] = stability_gap * net['input'][i]['mf'][j+1]['stability']
            net['input'][i]['mf'][j+1]['params'][0] = stability_gap * net['input'][i]['mf'][j]['stability']
        if Pad_exist:
            net['input'][i]['mf'][1]['params'][0] = net['input'][i]['mf'][1]['params'][2]
        else:
            net['input'][i]['mf'][0]['params'][0] = net['input'][i]['mf'][0]['params'][2]
        if Unk_exist:
            net['input'][i]['mf'][-2]['params'][2] = net['input'][i]['mf'][-2]['params'][0]
        else:
            net['input'][i]['mf'][-1]['params'][2] = net['input'][i]['mf'][-1]['params'][0]

    if not update_output_mf:
        return net

    # Update output dimensions
    for i in range(num_outputs):
        net['output'][i]['sum_square'] = (net['output'][i]['sum_square'] * forgettor) + data_target[i] ** 2
        net['output'][i]['sum'] = (net['output'][i]['sum'] * forgettor) + data_target[i]
        net['output'][i]['currency'] = (net['output'][i]['currency'] * forgettor) + 1
        net['output'][i]['spatio_temporal_dist'] = 4 * np.sqrt((net['output'][i]['sum_square'] / net['output'][i]['currency']) - (net['output'][i]['sum'] / net['output'][i]['currency']) ** 2)

        drift_detected = False
        if current_count > 1:
            drift_detected, neuron = mar_calculate_age2(net['output'][i], data_target[i], net['threshold_mf'], current_count, stop_phase1_at)
            net['output'][i] = neuron

        if drift_detected:
            net['output'][i] = mar_generate_mf3(net['output'][i], data_target[i], current_count, default_width_output, phase, net['rule'], i, 'output')

        num_mf = len(net['output'][i]['mf'])
        mf_values = np.zeros(num_mf)
        for j in range(num_mf):
            mf_values[j] = gauss2mf(data_target[i], net['output'][i]['mf'][j]['params'])
            centroid = (net['output'][i]['mf'][j]['params'][1] + net['output'][i]['mf'][j]['params'][3]) / 2
            centroid_width = net['output'][i]['mf'][j]['params'][3] - centroid
            curr_stability = net['output'][i]['mf'][j]['stability'] * forgettor
            centroid = (curr_stability * centroid + mf_values[j] * data_target[i]) / (curr_stability + mf_values[j])
            net['output'][i]['mf'][j]['params'][1] = centroid - centroid_width
            net['output'][i]['mf'][j]['params'][3] = centroid + centroid_width
            net['output'][i]['mf'][j]['stability'] = curr_stability + mf_values[j]

        max_index = np.argmax(mf_values)
        net['output'][i]['last_mf_won'] = max_index

        # Update variance
        for j in range(num_mf - 1):
            stability_gap = abs(net['output'][i]['mf'][j+1]['params'][1] - net['output'][i]['mf'][j]['params'][3]) / (net['output'][i]['mf'][j]['stability'] + net['output'][i]['mf'][j+1]['stability'])
            net['output'][i]['mf'][j]['params'][2] = stability_gap * net['output'][i]['mf'][j+1]['stability']
            net['output'][i]['mf'][j+1]['params'][0] = stability_gap * net['output'][i]['mf'][j]['stability']
        net['output'][i]['mf'][0]['params'][0] = net['output'][i]['mf'][0]['params'][2]
        net['output'][i]['mf'][-1]['params'][2] = net['output'][i]['mf'][-1]['params'][0]

    return net


def gauss2mf(x, params):
    if len(params) != 4:
        raise ValueError("GAUSS2MF needs four parameters.")

    sigma1, c1, sigma2, c2 = params

    if sigma1 == 0 or sigma2 == 0:
        raise ValueError("The sigma value must be non-zero.")

    Unk_indicator = False
    Pad_indicator = False
    if c1 == np.inf and c2 == np.inf and x == np.inf:
        Unk_indicator = True
        c1Index = 1
        c2Index = 0
    elif c1 == -np.inf and c2 == -np.inf and x == -np.inf:
        Pad_indicator = True
        c1Index = 1
        c2Index = 0
    else:
        c1Index = (x <= c1).astype(float)
        c2Index = (x >= c2).astype(float)

    if Unk_indicator or Pad_indicator:
        y1 = 1
        y2 = 1
    else:
        y1 = np.exp(-(x - c1) ** 2 / (2 * sigma1 ** 2)) * c1Index + (1 - c1Index)
        y2 = np.exp(-(x - c2) ** 2 / (2 * sigma2 ** 2)) * c2Index + (1 - c2Index)

   
    y = y1 * y2
    return y


def mar_cri(data_input, net, current_count, ablation=False, interpret=False, mode='base'):
    """
    Classification inference:
      - compute missing-aware firing for each active rule
      - shared rule state: pos_evi / neg_evi
      - mode='base' or 'ds'
    Returns:
      y: shape (num_outputs,) probability of positive class (binary; uses y[0])
    """
    num_outputs = len(net.get('output', {0: None}))
    y = np.zeros(num_outputs)
    num_vars = int(net.get('num_vars', 1))  # variables per timestep


    rules = net.get('rule', [])
    if not rules:
        y[:] = net.get('base_rate', 0.5)
        if interpret:
            return y, {'u': 1.0} if mode == 'ds' else {}
        return y

    num_attributes = len(data_input)
    num_rules = len(rules)

    firing = np.zeros(num_rules, dtype=float)
    miss_discount = np.ones(num_rules, dtype=float)
    weights = np.zeros(num_rules, dtype=float)
    active = np.zeros(num_rules, dtype=bool)
    paddingness = np.zeros(num_rules, dtype=float)
    rule_paddings = np.zeros(num_rules, dtype=int)


    for i, r in enumerate(rules):
        if not r.get('active', 1):
            continue
        active[i] = True
        weights[i] = float(r.get('weight', 0.0))

        row_vals = np.empty(num_attributes, dtype=float)
        row_vals[:] = 100.0  # ignored default
        miss = 0.0

        for j in range(num_attributes):
            aidx = int(r['antecedent'][j])
            if aidx == -1:
                row_vals[j] = 100.0
                continue
            params = net['input'][j]['mf'][aidx]['params']
            v = gauss2mf(data_input[j], params)

            if v == 0:
                v = 1e-100
                # missing token (x=inf or MF center=inf)
                if data_input[j] == np.inf or params[1] == np.inf:
                    v = 100.0
                    miss += 1.0
                # padding token mismatch
                if data_input[j] == -np.inf and params[1] == np.inf:
                    v = 300.0
                    paddingness[i] += 1.0
                elif data_input[j] == -np.inf:
                    v = 200.0
                    paddingness[i] += 1.0
                # rule expects padding but input not padded
                if params[1] == -np.inf:
                    v = 250.0
                    paddingness[i] -= 1.0
                    rule_paddings[i] += 1

            row_vals[j] = v

        f = float(np.min(row_vals))
        if f > 1:
            f = 1e-100
        firing[i] = f
        miss_discount[i] = float(np.exp(num_attributes - miss) / np.exp(num_attributes))

    # find the top-5 firing rules
    top_k_indices = np.argsort(firing)[-5:][::-1]
    for i in top_k_indices:
        rules[i]['infer_times'][current_count - 1] = 1


    g = firing * weights * miss_discount
    g[~active] = 0.0

    # partial matching on padded suffix
    input_paddings = int(np.sum(np.isneginf(data_input)))
    if input_paddings > 0 and num_vars > 0:
        nonpad_input = data_input[input_paddings:]
        L = int(len(nonpad_input))  # length of nonpad suffix (in flattened dims)

        if L > 0:
            g_part = np.zeros_like(g)

            for i, r in enumerate(rules):
                if not active[i]:
                    continue

                # diff padding in "timesteps"
                diff_steps = int(paddingness[i] / max(num_vars, 1))
                if diff_steps <= 0:
                    continue

                base_start = int(rule_paddings[i])
                w_best = 0.0

                for shift in range(diff_steps):
                    start = base_start + shift * num_vars
                    end = start + L
                    if end > num_attributes:
                        break

                    miss_p = 0.0
                    f_part = 1.0

                    # compute min membership over aligned suffix
                    for jj in range(L):
                        k = start + jj
                        aidx = int(r['antecedent'][k])
                        if aidx == -1:
                            continue  # ignore dim

                        params = net['input'][k]['mf'][aidx]['params']
                        vv = gauss2mf(nonpad_input[jj], params)

                        if vv == 0:
                            vv = 1e-100
                            if nonpad_input[jj] == np.inf or params[1] == np.inf:
                                vv = 100.0
                                miss_p += 1.0

                        # keep a running min (firing)
                        if vv < f_part:
                            f_part = vv

                    if f_part > 1:
                        f_part = 1e-100

                    # same style as original: missingness discount on the suffix
                    disc_p = float(np.exp(L - miss_p) / np.exp(L))

                    cand = f_part * weights[i] * disc_p
                    if f_part > 0.8:
                        cand = f_part * weights[i] * disc_p
                        w_best = max(w_best, cand)

                # optional threshold (your original code used ~0.8 to accept partial matches)
                g_part[i] += w_best

            g = g + g_part


    sum_g = float(np.sum(g))
    if sum_g <= 0:
        y[:] = net.get('base_rate', 0.5)
        if interpret:
            return y, {'u': 1.0} if mode == 'ds' else {}
        return y

    g_norm = g / sum_g

    pos = np.array([float(r.get('pos_evi', 0.0)) for r in rules], dtype=float)
    neg = np.array([float(r.get('neg_evi', 0.0)) for r in rules], dtype=float)

    if mode == 'ds':
        W = float(net.get('ds_W', 2.0))
        a = float(net.get('base_rate', 0.5))

        Spos = float(np.sum(g_norm * pos))
        Sneg = float(np.sum(g_norm * neg))
        T = Spos + Sneg + W

        b_pos = Spos / max(T, 1e-12)
        u = W / max(T, 1e-12)

        p = b_pos + a * u
        y[0] = float(np.clip(p, 0.0, 1.0))

        if interpret:
            return y, {'u': u, 'b_pos': b_pos, 'b_neg': (Sneg / max(T, 1e-12)), 'a': a}
        return y

    # base mode
    lam = float(net.get('base_lambda', 1.0))
    p_r = (pos + lam) / np.maximum(pos + neg + 2.0 * lam, 1e-12)
    p = float(np.sum(g_norm * p_r))
    y[0] = float(np.clip(p, 0.0, 1.0))

    if interpret:
        return y, {'p_r': p_r, 'g': g_norm}
    return y



def mar_calculate_age2(neuron, data, threshold_mf, current_count, end_of_phase_1):
    num_mf = len(neuron['mf'])  
    mf_values = np.zeros(num_mf)

    for j in range(num_mf):
        params = neuron['mf'][j]['params']
        if params[0] != 0 and params[2] != 0:
            mf_values[j] = gauss2mf(data, params)

    max_mf_value = np.max(mf_values)
    max_index = np.argmax(mf_values)

    neuron['age'].append(np.max(neuron['age']) + (1 - max_mf_value))

    second_diff_age = np.diff(np.diff(neuron['age']))

    drift_detected = False

    if current_count < end_of_phase_1:
        if max_mf_value < 0.5:
            drift_detected = True
        return drift_detected, neuron

    if (second_diff_age[current_count-1 - 2] > second_diff_age[current_count-1 - 3] and 
        max_mf_value < threshold_mf):
        drift_detected = True

    return drift_detected, neuron


def mar_generate_mf3(data_struct, data_value, current_count, default_width, phase, rule = None, index = None, stage = None):
    # print(f"Generate MF3 at current: {current_count} for {stage}_{index}")
    
    Alpha_phase1 = 2  
    
    num_mf = len(data_struct['mf'])
    left = -1

    for i in range(num_mf):
        current_centroid = data_struct['mf'][i]['params'][1]
        if current_centroid < data_value:
            left = i
        else:
            break

    new_mf = left + 1
    new_struct = {'mf': [None] * (num_mf + 1)}

    num_rule = len(rule)
    processed_rule = set()
    for i in range(num_mf):
        if i >= new_mf:
            new_struct['mf'][i + 1] = data_struct['mf'][i]
            new_struct['mf'][i + 1]['name'] = f"mf{i + 1 + 1}"
            if stage == 'input':
                for j in range(num_rule):
                    if rule[j]['antecedent'][index] == i:
                        if j in processed_rule:
                            continue
                        else:
                            rule[j]['antecedent'][index] = i+1
                            processed_rule.add(j)
            elif stage == 'output':
                for j in range(num_rule):
                    if rule[j]['consequent'][index] == i:
                        if j in processed_rule:
                            continue
                        else:
                            rule[j]['consequent'][index] = i+1
                            processed_rule.add(j)
        else:
            new_struct['mf'][i] = data_struct['mf'][i]


    new_struct['mf'][new_mf] = {
        'name': f"mf{new_mf+1}",
        'type': 'gauss2mf',
        'params': [default_width, data_value, default_width, data_value]
    }

    if data_value != np.inf and data_value != -np.inf:

        if phase == 1:
            Unk_exsit = False
            Pad_exsit = False
            for i in range(num_mf):
                if new_struct['mf'][i + 1]['params'][1] == np.inf:
                    Unk_exsit = True
                    break
                elif new_struct['mf'][i]['params'][1] == -np.inf:
                    Pad_exsit = True
                    continue
                else:
                    new_struct['mf'][i]['params'][2] = abs(
                        (new_struct['mf'][i]['params'][3] - new_struct['mf'][i + 1]['params'][1]) / np.sqrt(2 * np.log(Alpha_phase1))
                    )
                    new_struct['mf'][i + 1]['params'][0] = new_struct['mf'][i]['params'][2]
            if Pad_exsit:
                new_struct['mf'][1]['params'][0] = new_struct['mf'][1]['params'][2]
            else:
                new_struct['mf'][0]['params'][0] = new_struct['mf'][0]['params'][2]
            if Unk_exsit:
                new_struct['mf'][num_mf-1]['params'][2] = new_struct['mf'][num_mf-1]['params'][0]
            else:
                new_struct['mf'][num_mf]['params'][2] = new_struct['mf'][num_mf]['params'][0]
        else:
            if new_mf != 0 and new_struct['mf'][new_mf - 1]['params'][1] != -np.inf:
                new_struct['mf'][new_mf]['params'][0] = abs(
                    (data_value - np.mean([new_struct['mf'][new_mf - 1]['params'][3], data_value])) / np.sqrt(2 * np.log(2))
                )
                new_struct['mf'][new_mf - 1]['params'][2] = new_struct['mf'][new_mf]['params'][0]
            
            if new_mf != num_mf and new_struct['mf'][new_mf + 1]['params'][1] != np.inf:
                new_struct['mf'][new_mf]['params'][2] = abs(
                    (np.mean([new_struct['mf'][new_mf + 1]['params'][1], data_value]) - data_value) / np.sqrt(2 * np.log(2))
                )
                new_struct['mf'][new_mf + 1]['params'][0] = new_struct['mf'][new_mf]['params'][2]
            elif new_mf != num_mf:
                new_struct['mf'][new_mf]['params'][2] = new_struct['mf'][new_mf]['params'][0]


    
    new_struct['mf'][new_mf].update({
        'stability': 0,
        'tw_sum': 0,
        'mf_currency': 0,
        'num_invoked': 0, 
        'created_at': current_count
    })

    left_range = new_struct['mf'][0]['params'][1] - (np.sqrt(2 * np.log(100)) * new_struct['mf'][0]['params'][0])
    if new_struct['mf'][num_mf]['params'][3] != np.inf:
        right_range = new_struct['mf'][num_mf]['params'][3] + (np.sqrt(2 * np.log(100)) * new_struct['mf'][num_mf]['params'][2])
    else: 
        right_range = new_struct['mf'][num_mf-1]['params'][3] + (np.sqrt(2 * np.log(100)) * new_struct['mf'][num_mf-1]['params'][2])
    new_struct['range'] = [left_range, right_range]

    
    D = {
        'name': data_struct['name'],
        'range': new_struct['range'],
        'min': data_struct['min'],
        'max': data_struct['max'],
        'phase': data_struct['phase'],
        'spatio_temporal_dist': data_struct['spatio_temporal_dist'],
        'sum_square': data_struct['sum_square'],
        'sum': data_struct['sum'],
        'currency': data_struct['currency'],
        'last_mf_won': data_struct['last_mf_won'],
        'age': data_struct['age'],
        'mf': new_struct['mf']
    }

    return D




def mar_update_sliding_threshold(rule, postSignal, forgettor):
    offset = forgettor

    if forgettor == 1:
        inverse_forgettor = 0
    else:
        inverse_forgettor = 1 / (1 - forgettor)

    rule['weight'] *= offset
    rule['topCache'] = rule['topCache'] * offset + (postSignal * postSignal)
    rule['baseCache'] = rule['baseCache'] * offset + (1 - offset) * inverse_forgettor

    return rule


def mar_normalize_weights(net):
    num_rules = len(net.get('rule', []))
    if num_rules == 0:
        return net

    weight_database = np.zeros(num_rules, dtype=float)
    for i in range(num_rules):
        weight_database[i] = float(net['rule'][i].get('weight', 0.0))

    max_weight = float(np.max(weight_database))
    if max_weight <= 0.0:
        return net  

    for i in range(num_rules):
        net['rule'][i]['weight'] = float(net['rule'][i].get('weight', 0.0)) / max_weight
    return net



def mar_online_rule(net, data_input, data_target, current_count):
    """
    Classification rule update:
      - rule identity: antecedent only
      - consequent MF removed; maintain pos/neg evidence per rule
      - shared state works for both base/ds mode
    """
    if current_count == 1:
        print("tau_match=", net.get('tau_match', None), "score_q=", net.get('score_q', None))


    num_attributes = data_input.shape[0]
    forgettor = net['forgettor']
    forgettor_pos = net.get('forgettor_pos', forgettor)
    W = float(net.get('ds_W', 2.0))

    # label y in {0,1}
    y = float(np.ravel(data_target)[0])
    y = 1.0 if y >= 0.5 else 0.0


    # --- EWMA prior + dynamic pos_w ---
    eps = 1e-12
    gamma = float(net.get('gamma_prior', forgettor))   # 用你初始化的 gamma_prior；没给就回退到 forgettor
    net['prior_pos']   = gamma * float(net.get('prior_pos', 0.0))   + y
    net['prior_total'] = gamma * float(net.get('prior_total', 0.0)) + 1.0
    net['base_rate']   = float(net['prior_pos'] / max(net['prior_total'], eps))

    # dynamic positive weight: (neg/pos), clipped
    neg = net['prior_total'] - net['prior_pos']
    pos_w = (neg + eps) / (net['prior_pos'] + eps)

    w_max = float(net.get('w_max', 20.0))
    net['pos_w'] = float(np.clip(pos_w, 1.0, w_max))

    # winning MF per input dimension + store mf table for soft matching
    max_mf = np.zeros(num_attributes, dtype=float)
    max_set = np.zeros(num_attributes, dtype=int)
    mf_table = []  # list of np.array, each shape (num_mf,)
    top2_vals = []

    finite_dims = 0
    for i in range(num_attributes):
        num_mf = len(net['input'][i]['mf'])

        # missing/padding: make this dim "neutral" for matching
        if not np.isfinite(data_input[i]):
            mf_values = np.ones(num_mf, dtype=float)
            max_mf[i] = 1.0

            target_idx = 0 # 默认 fallback
            
            if data_input[i] == -np.inf:
                for k in range(num_mf):
                    if net['input'][i]['mf'][k]['params'][1] == -np.inf:
                        target_idx = k
                        break
            elif data_input[i] == np.inf:
                for k in range(num_mf):
                    if net['input'][i]['mf'][k]['params'][1] == np.inf:
                        target_idx = k
                        break

            max_set[i] = int(target_idx)

            mf_table.append(mf_values)
            top2_vals.append((1.0, 1.0))
            continue

        finite_dims += 1
        mf_values = np.zeros(num_mf, dtype=float)
        for j in range(num_mf):
            p = net['input'][i]['mf'][j]['params']
            if p[0] != 0 and p[2] != 0:
                v = gauss2mf(data_input[i], p)
                mf_values[j] = 0.0 if (not np.isfinite(v)) else float(v)

        # store top-2 memberships for partial matching compensation
        if num_mf >= 2:
            top2_idx = np.argpartition(mf_values, -2)[-2:]
            # sort so top2_idx[0] is best
            top2_idx = top2_idx[np.argsort(mf_values[top2_idx])[::-1]]
        else:
            top2_idx = np.array([int(np.argmax(mf_values))], dtype=int)

        if len(top2_idx) == 1:
            v1 = float(mf_values[top2_idx[0]])
            v2 = 0.0
        else:
            v1 = float(mf_values[top2_idx[0]])
            v2 = float(mf_values[top2_idx[1]])
        top2_vals.append((v1, v2))


        max_mf[i] = float(np.max(mf_values))
        max_set[i] = int(np.argmax(mf_values))
        mf_table.append(mf_values)

        net['input'][i]['mf'][max_set[i]]['num_invoked'] += 1

    antecedent = max_set.tolist()

    # firing strength proxy (clipped)
    if finite_dims == 0:
        f = 0.0
    else:
        f = float(np.min(max_mf))  # missing dims already set to 1.0, so OK
    if f > 1:
        f = 1e-100
    f = float(np.clip(f, 0.0, 1.0))



    if 'rule' not in net:
        net['rule'] = []

    protect_T = net.get('protect_T', 0)
    for r in net['rule']:
        lastp = r.get('last_pos_update', -1)

        in_protect = (lastp >= 0) and (np.isinf(protect_T) or (protect_T > 0 and (current_count - lastp) <= protect_T))

        if in_protect:
            r['pos_evi'] = float(r.get('pos_evi', 0.0)) * forgettor_pos
            r['neg_evi'] = float(r.get('neg_evi', 0.0)) * forgettor_pos
        else:
            r['pos_evi'] = float(r.get('pos_evi', 0.0)) * forgettor
            r['neg_evi'] = float(r.get('neg_evi', 0.0)) * forgettor
    
    
    use_pos_weight = net.get('use_pos_weight', False)
    pos_w = float(net.get('pos_w', 1.0)) if use_pos_weight else 1.0         # 你 EWMA 算出来的动态权重
    p_gate = 0
    g = float(f ** p_gate)

    # effective positive weight: in [1, pos_w]
    # pos_w_eff = 1.0 + (pos_w - 1.0) * g   # original
    pos_w_eff = 1.0  # ablation

    # soft match: choose best matching rule by membership lookup
    tau_match = float(net.get('tau_match', 0.7))  
    best_idx = None
    best_score = -1.0

    score_q = float(net.get('score_q', 0.0))  

    for idx, r in enumerate(net['rule']):
        a = r.get('antecedent', None)
        if a is None:
            continue

        mus = []
        ok = True
        for d in range(num_attributes):
            mf_id = int(a[d])
            vals = mf_table[d]
            if mf_id < 0 or mf_id >= len(vals):
                ok = False
                break
            mus.append(float(vals[mf_id]))

        if not ok:
            s = 0.0
        else:
            mus_arr = np.asarray(mus, dtype=float)

            if score_q <= 0.0:
                # --- base: strict min score ---
                s0 = float(np.min(mus_arr))
                s = s0

                # --- partial matching: one-dimension compensation (2nd best MF) ---
                if net.get('use_partial_match', True) and num_attributes >= 2:
                    d_min = int(np.argmin(mus_arr))             # bottleneck dim
                    min_other = float(np.min(np.delete(mus_arr, d_min)))  # min of remaining dims

                    v1, v2 = top2_vals[d_min]  # top-1 / top-2 memberships for THIS INPUT at dim d_min
                    cur = float(mus_arr[d_min])

                    # only helps when rule picked a poor MF at bottleneck dim
                    if v2 > cur:
                        cand = float(min(v2, min_other))
                        if cand > s:
                            s = cand

                        net['pm_try'] = net.get('pm_try', 0) + 1
                        if cand > s0 + 1e-12:
                            net['pm_hit'] = net.get('pm_hit', 0) + 1
                            net['pm_gain_sum'] = net.get('pm_gain_sum', 0.0) + (cand - s0)

            else:
                s = float(np.quantile(mus_arr, score_q))

        # early prune
        if s <= best_score:
            continue

        if s > best_score:
            best_score = s
            best_idx = idx

    # decide: update best rule if similar enough, else create new rule
    match_idx = best_idx if (best_score >= tau_match) else None

    net['dbg_newrule_cnt'] = net.get('dbg_newrule_cnt', 0) + (1 if match_idx is None else 0)

    if best_idx is not None:
        net['dbg_bestscore_min'] = min(float(net.get('dbg_bestscore_min', 1.0)), float(best_score))
        net['dbg_bestscore_max'] = max(float(net.get('dbg_bestscore_max', 0.0)), float(best_score))

    
    
    if match_idx is None:
        # create new rule
        pos = pos_w_eff * f * y
        neg = 1.0   * f * (1.0 - y)

        ev_sum = pos + neg
        weight = ev_sum / (ev_sum + W) if (ev_sum + W) > 0 else 0.0

        # invoke list
        invoked_times = np.zeros(net['predicted'].shape[0], dtype=int)
        invoked_times[current_count - 1] = 1
        infer_times = np.zeros(net['predicted'].shape[0], dtype=int)

        net['rule'].append({
            'antecedent': antecedent,
            'consequent': [0],   # placeholder (unused)
            'pos_evi': float(pos),
            'neg_evi': float(neg),
            'weight': float(weight),
            'connection': 1,
            'num_invoked': 1,
            'invoked_times': invoked_times,
            'infer_times': infer_times,
            'topCache': 0,
            'baseCache': 0,
            'lastUpdate': current_count,
            'last_pos_update': current_count if (y == 1 and f > 0) else -1,
            'belong': 1,
            'active': 1
        })
        # print('generate new rule')
    else:
        r = net['rule'][match_idx]
        r['pos_evi'] = float(r.get('pos_evi', 0.0)) + pos_w_eff * f * y
        r['neg_evi'] = float(r.get('neg_evi', 0.0)) + 1.0   * f * (1.0 - y)


        ev_sum = r['pos_evi'] + r['neg_evi']
        r['weight'] = float(ev_sum / (ev_sum + W)) if (ev_sum + W) > 0 else 0.0

        r['num_invoked'] = float(r.get('num_invoked', 0.0)) + 1
        r['lastUpdate'] = current_count
        r['active'] = 1
        if y == 1 and f > 0:
            r['last_pos_update'] = current_count

        invoked_times = r.get('invoked_times', np.zeros(net['predicted'].shape[0], dtype=int))
        invoked_times[current_count - 1] = 1
        r['invoked_times'] = invoked_times

        net['rule'][match_idx] = r

    net = mar_normalize_weights(net)
    return net



def mar_pseudo_prune_rule(net, current_count):
    min_weight = net['min_rule_weight']
    num_rules = len(net['rule'])
    count_rules = 0

    for i in range(num_rules):
        if net['rule'][i]['weight'] < min_weight:
            net['rule'][i]['active'] = 0
            net['rule'][i]['lastUpdate'] = current_count
        else:
            net['rule'][i]['active'] = 1
            count_rules += 1
    
    net['num_active_rules'] = count_rules
    return net


def mar_merge_mf(net, option, index):
    beta = 0.5
    TD = 0.5
    num_rules = len(net['rule'])
    merging_to_occur = False
    denom = net['max_cluster']
    
    if option == 'input':
        num_mf = len(net['input'][index]['mf'])
        if num_mf == 1:
            return net, False
        
        mf_database = np.zeros((num_mf, 4))
        
        # Identify mean of centroids of each MF
        for j in range(num_mf):
            mf_database[j, 0] = net['input'][index]['mf'][j]['params'][1]
            mf_database[j, 1] = net['input'][index]['mf'][j]['params'][3]
            mf_database[j, 2] = (mf_database[j, 0] + mf_database[j, 1]) / 2
        
        # Identify distances between centroids
        mf_distances = np.diff(mf_database[:, 2])
        
        # Calculate Reducibility criteria
        reducibility_threshold = net['input'][index]['spatio_temporal_dist'] / denom
        
        # Identify membership functions to be merged
        for j in range(num_mf - 1):
            if mf_distances[j] < reducibility_threshold:
                mf_database[j, 3] = 1
                mf_database[j + 1, 3] = 1
                merging_to_occur = True
        
        if merging_to_occur:
            start_merge = -1
            stop_merge = -1
            merge_mapping = np.zeros(num_mf, dtype=int)
            new_mf_count = 0
            new_mf_struct = []
            
            for j in range(num_mf):
                if mf_database[j, 3] == 1 and start_merge == -1 and stop_merge == -1:
                    start_merge = j
                elif ((j == num_mf - 1 and mf_database[-1, 3] == 1) or 
                      (mf_database[j, 3] == 1 and mf_database[j + 1, 3] == 0)) and start_merge != -1 and stop_merge == -1:
                    stop_merge = j
                
                if start_merge != -1 and stop_merge != -1:
                    new_mf_count += 1
                    left_centroid = np.zeros(stop_merge - start_merge + 1)
                    right_centroid = np.zeros(stop_merge - start_merge + 1)
                    stability = np.zeros(stop_merge - start_merge + 1)
                    
                    for k in range(start_merge, stop_merge + 1):
                        merge_mapping[j + start_merge - k] = new_mf_count
                        left_centroid[k - start_merge] = net['input'][index]['mf'][k]['params'][1]
                        right_centroid[k - start_merge] = net['input'][index]['mf'][k]['params'][3]
                        stability[k - start_merge] = net['input'][index]['mf'][k]['stability']
                        
                        if k > start_merge and k != stop_merge:
                            net['input'][index]['mf'][start_merge]['num_invoked'] += net['input'][index]['mf'][k]['num_invoked']
                        elif k == stop_merge:
                            net['input'][index]['mf'][start_merge]['num_invoked'] += net['input'][index]['mf'][k]['num_invoked']
                            net['input'][index]['mf'][start_merge]['params'][3] = net['input'][index]['mf'][k]['params'][3] ###
                            net['input'][index]['mf'][start_merge]['params'][2] = net['input'][index]['mf'][k]['params'][2] ###

                            net['input'][index]['mf'][start_merge]['plasticity'] = beta
                            net['input'][index]['mf'][start_merge]['tendency'] = TD
                            net['input'][index]['mf'][start_merge]['num_expanded'] = 0

                            net['input'][index]['mf'][start_merge]['params'][1] = (left_centroid.mean() if stability.sum() == 0 else np.sum(left_centroid * stability) / stability.sum())
                            net['input'][index]['mf'][start_merge]['params'][3] = (right_centroid.mean() if stability.sum() == 0 else np.sum(right_centroid * stability) / stability.sum())
                            net['input'][index]['mf'][start_merge]['stability'] = (0 if stability.sum() == 0 else stability.mean())
                    
                    start_merge = -1
                    stop_merge = -1
                elif start_merge == -1 and stop_merge == -1:
                    new_mf_count += 1
                    merge_mapping[j] = new_mf_count
            
            for j in range(num_mf):
                if j == 0:
                    new_mf_struct.append(net['input'][index]['mf'][j])
                elif merge_mapping[j] == merge_mapping[j - 1]:
                    for k in range(num_rules):
                        if net['rule'][k]['antecedent'][index] == j:
                            net['rule'][k]['antecedent'][index] = merge_mapping[j] - 1 ###
                elif merge_mapping[j] != merge_mapping[j - 1]:
                    new_mf_struct.append(net['input'][index]['mf'][j])
                    new_mf_struct[-1]['name'] = f'mf{merge_mapping[j]}'
                    for k in range(num_rules):
                        if net['rule'][k]['antecedent'][index] == j:
                            net['rule'][k]['antecedent'][index] = merge_mapping[j] - 1 ###
            
            net['input'][index]['mf'] = new_mf_struct
        
        return net, merging_to_occur

    elif option == 'output':
        num_mf = len(net['output'][index]['mf'])
        if num_mf == 1:
            return net, False
        
        mf_database = np.zeros((num_mf, 4))
        
        # Identify mean of centroids of each MF
        for j in range(num_mf):
            mf_database[j, 0] = net['output'][index]['mf'][j]['params'][1]
            mf_database[j, 1] = net['output'][index]['mf'][j]['params'][3]
            mf_database[j, 2] = (mf_database[j, 0] + mf_database[j, 1]) / 2
        
        # Identify distances between centroids
        mf_distances = np.diff(mf_database[:, 2])
        
        # Calculate Reducibility criteria
        reducibility_threshold = net['output'][index]['spatio_temporal_dist'] / denom
        
        # Identify membership functions to be merged
        for j in range(num_mf - 1):
            if mf_distances[j] < reducibility_threshold:
                mf_database[j, 3] = 1
                mf_database[j + 1, 3] = 1
                merging_to_occur = True
        
        if merging_to_occur:
            start_merge = -1
            stop_merge = -1
            merge_mapping = np.zeros(num_mf, dtype=int)
            new_mf_count = 0
            new_mf_struct = []
            
            for j in range(num_mf):
                if mf_database[j, 3] == 1 and start_merge == -1 and stop_merge == -1:
                    start_merge = j
                elif ((j == num_mf - 1 and mf_database[-1, 3] == 1) or 
                      (mf_database[j, 3] == 1 and mf_database[j + 1, 3] == 0)) and start_merge != -1 and stop_merge == -1:
                    stop_merge = j
                
                if start_merge != -1 and stop_merge != -1:
                    new_mf_count += 1
                    left_centroid = np.zeros(stop_merge - start_merge + 1)
                    right_centroid = np.zeros(stop_merge - start_merge + 1)
                    stability = np.zeros(stop_merge - start_merge + 1)
                    
                    for k in range(start_merge, stop_merge + 1):
                        merge_mapping[j + start_merge - k] = new_mf_count
                        left_centroid[k - start_merge] = net['output'][index]['mf'][k]['params'][1]
                        right_centroid[k - start_merge] = net['output'][index]['mf'][k]['params'][3]
                        stability[k - start_merge] = net['output'][index]['mf'][k]['stability']
                        if k > start_merge and k != stop_merge:
                            net['output'][index]['mf'][start_merge]['num_invoked'] += net['output'][index]['mf'][k]['num_invoked']
                        elif k == stop_merge:
                            net['output'][index]['mf'][start_merge]['num_invoked'] += net['output'][index]['mf'][k]['num_invoked']
                            net['output'][index]['mf'][start_merge]['params'][3] = net['output'][index]['mf'][k]['params'][3] ###
                            net['output'][index]['mf'][start_merge]['params'][2] = net['output'][index]['mf'][k]['params'][2] ###

                            net['output'][index]['mf'][start_merge]['plasticity'] = beta
                            net['output'][index]['mf'][start_merge]['tendency'] = TD
                            net['output'][index]['mf'][start_merge]['num_expanded'] = 0

                            net['output'][index]['mf'][start_merge]['params'][1] = (left_centroid.mean() if stability.sum() == 0 else np.sum(left_centroid * stability) / stability.sum())
                            net['output'][index]['mf'][start_merge]['params'][3] = (right_centroid.mean() if stability.sum() == 0 else np.sum(right_centroid * stability) / stability.sum())
                            net['output'][index]['mf'][start_merge]['stability'] = (0 if stability.sum() == 0 else stability.mean())
                    
                    start_merge = -1
                    stop_merge = -1
                elif start_merge == -1 and stop_merge == -1:
                    new_mf_count += 1
                    merge_mapping[j] = new_mf_count
            
            for j in range(num_mf):
                if j == 0:
                    new_mf_struct.append(net['output'][index]['mf'][j])
                elif merge_mapping[j] == merge_mapping[j - 1]:
                    for k in range(num_rules):
                        if net['rule'][k]['consequent'][index] == j:
                            net['rule'][k]['consequent'][index] = merge_mapping[j] - 1 ###
                elif merge_mapping[j] != merge_mapping[j - 1]:
                    new_mf_struct.append(net['output'][index]['mf'][j])
                    new_mf_struct[-1]['name'] = f'mf{merge_mapping[j]}'
                    for k in range(num_rules):
                        if net['rule'][k]['consequent'][index] == j:
                            net['rule'][k]['consequent'][index] = merge_mapping[j] - 1 ###
            
            net['output'][index]['mf'] = new_mf_struct
        
        return net, merging_to_occur


def mar_clean_mf(net, merge_output=True):
    num_attributes = len(net['input'])
    num_outputs = len(net['output'])
    mm = 0
    
    # merge membership functions
    if merge_output:
        for i in range(num_outputs):
            net, m = mar_merge_mf(net, 'output', i)
            # m = 1 means some mfs were merged
            if m:
                # if any mfs were merged, return 1 to caller
                mm = 1
                
                # Update variance
                num_mf = len(net['output'][i]['mf'])
                for j in range(num_mf - 1):
                    stability_gap = abs(net['output'][i]['mf'][j+1]['params'][1] - net['output'][i]['mf'][j]['params'][3]) / (net['output'][i]['mf'][j]['stability'] + net['output'][i]['mf'][j+1]['stability'])
                    net['output'][i]['mf'][j]['params'][2] = stability_gap * net['output'][i]['mf'][j+1]['stability']
                    net['output'][i]['mf'][j+1]['params'][0] = stability_gap * net['output'][i]['mf'][j]['stability']
                net['output'][i]['mf'][0]['params'][0] = net['output'][i]['mf'][0]['params'][2]
                net['output'][i]['mf'][-1]['params'][2] = net['output'][i]['mf'][-1]['params'][0]
    
    for i in range(num_attributes):
        net, m = mar_merge_mf(net, 'input', i)
        if m:
            mm = 1
            # Update variance
            num_mf = len(net['input'][i]['mf'])

            Unk_exist = False
            Pad_exist = False
            for j in range(num_mf - 1):
                if net['input'][i]['mf'][j+1]['params'][1] == np.inf:
                    Unk_exist = True
                    break
                elif net['input'][i]['mf'][j]['params'][1] == -np.inf:
                    Pad_exist = True
                    continue
                else:
                    stability_gap = abs(net['input'][i]['mf'][j+1]['params'][1] - net['input'][i]['mf'][j]['params'][3]) / (net['input'][i]['mf'][j]['stability'] + net['input'][i]['mf'][j+1]['stability'])
                net['input'][i]['mf'][j]['params'][2] = stability_gap * net['input'][i]['mf'][j+1]['stability']
                net['input'][i]['mf'][j+1]['params'][0] = stability_gap * net['input'][i]['mf'][j]['stability']
            if Pad_exist:
                net['input'][i]['mf'][1]['params'][0] = net['input'][i]['mf'][1]['params'][2]
            else:
                net['input'][i]['mf'][0]['params'][0] = net['input'][i]['mf'][0]['params'][2]
            if Unk_exist:
                net['input'][i]['mf'][-2]['params'][2] = net['input'][i]['mf'][-2]['params'][0]
            else:
                net['input'][i]['mf'][-1]['params'][2] = net['input'][i]['mf'][-1]['params'][0]
    
    return net, mm


def mar_clean_pop(net, current_count):
    """
    Classification cleanup after MF merge:
      - merge rules by antecedent only
      - aggregate evidence + metadata
    """
    new_rules = {}
    W = float(net.get('ds_W', 2.0))

    for r in net.get('rule', []):
        if r is None:
            continue

        # keep your original aging prune logic roughly
        if (r.get('active', 1) == 0 and
            (r.get('lastUpdate', 0) < (current_count - 2 * net['half_life']) or r.get('weight', 0.0) < 0.05)):
            continue

        key = tuple(r['antecedent'])
        if key not in new_rules:
            new_rules[key] = dict(r)
        else:
            keep = new_rules[key]
            keep['pos_evi'] = float(keep.get('pos_evi', 0.0)) + float(r.get('pos_evi', 0.0))
            keep['neg_evi'] = float(keep.get('neg_evi', 0.0)) + float(r.get('neg_evi', 0.0))
            keep['num_invoked'] = float(keep.get('num_invoked', 0.0)) + float(r.get('num_invoked', 0.0))
            keep['invoked_times'] = np.asarray(keep.get('invoked_times', np.zeros_like(r.get('invoked_times', np.zeros(1)))), dtype=int) + \
                                     np.asarray(r.get('invoked_times', np.zeros_like(keep.get('invoked_times', np.zeros(1)))), dtype=int)
            ## convert invoked_times to {0,1}
            keep['invoked_times'] = (keep['invoked_times'] > 0).astype(int)
            keep['infer_times'] = np.asarray(keep.get('infer_times', np.zeros_like(r.get('infer_times', np.zeros(1)))), dtype=int) + \
                                   np.asarray(r.get('infer_times', np.zeros_like(keep.get('infer_times', np.zeros(1)))), dtype=int)
            ## convert infer_times to {0,1}
            keep['infer_times'] = (keep['infer_times'] > 0).astype(int)
            keep['active'] = max(int(keep.get('active', 1)), int(r.get('active', 1)))
            keep['lastUpdate'] = max(int(keep.get('lastUpdate', 0)), int(r.get('lastUpdate', 0)))
            keep['topCache'] = float(keep.get('topCache', 0.0)) + float(r.get('topCache', 0.0))
            keep['baseCache'] = float(keep.get('baseCache', 0.0)) + float(r.get('baseCache', 0.0))
            new_rules[key] = keep

    net['rule'] = list(new_rules.values())

    # refresh evidence-based weights
    for r in net.get('rule', []):
        ev_sum = float(r.get('pos_evi', 0.0)) + float(r.get('neg_evi', 0.0))
        r['weight'] = float(ev_sum / (ev_sum + W)) if (ev_sum + W) > 0 else 0.0

    net = mar_normalize_weights(net)
    return net




