import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.analysis.woflan import algorithm as woflan
import os
from SPN_Simulator.SPN_Simulator import StochasticPetriNetSimulator
from utils.train_utils import splitRealLog, addStartEndEvents, addStartEndTransitions
from utils.createDataset import CreateDataset
from utils.Discriminator import Discriminator
from utils.repair_utils import updateModel
import json
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()

import random
random.seed(72)

parser.add_argument('--case_study', type=str, default='Purchasing')
parser.add_argument('--max_number_iterations', type=int, default=15)
parser.add_argument('--show_shap', type=bool, default=False)
parser.add_argument('--save_shap', type=bool, default=True)
parser.add_argument('--method', type=str, default='greedy')

args = parser.parse_args()

case_study = args.case_study
save_shap = args.save_shap
show_shap = args.show_shap
max_number_iterations = args.max_number_iterations
method = args.method

def run_repair(case_study, method='greedy', save_shap=True, show_shap=False, max_number_iterations=15):
    log_real = xes_importer.apply(f'data/{case_study}/log.xes')
    try:
        log_real = pm4py.filter_event_attribute_values(log_real, 'lifecycle:transition', ['complete'], level="event", retain=True)
    except:
        pass        
    log_real = addStartEndEvents(log_real)

    if 'logTrain.xes' in os.listdir(f'data/{case_study}') and 'logTrain.xes' in os.listdir(f'data/{case_study}') and 'logTrain.xes' in os.listdir(f'data/{case_study}'):
        real_train = xes_importer.apply(f'data/{case_study}/logTrain.xes')
        real_val = xes_importer.apply(f'data/{case_study}/logVal.xes')
    else:
        real_train, real_val, _ = splitRealLog(log_real, split_size = (0.6, 0.2, 0.2), split_temporal = True, save_to=f'data/{case_study}')

    try:
        n_exp = max([int(d.split('_')[-1]) for d in os.listdir(f'data/{case_study}') if d[:3]=='exp']) + 1
    except:
        n_exp = 1

    if case_study not in os.listdir('plots'):
        os.mkdir(f'plots/{case_study}')

    os.mkdir(f'data/{case_study}/exp_{n_exp}')
    os.mkdir(f'data/{case_study}/exp_{n_exp}/it_0')
    os.mkdir(f'plots/{case_study}/exp_{n_exp}')

    net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/diagram_0.pnml')
    
    # check if it is sound
    is_sound = woflan.apply(net, initial_marking, final_marking, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                                             woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                                             woflan.Parameters.RETURN_DIAGNOSTICS: False})
    if not is_sound:
        print('Petri Net must be sound.')
        return None
    
    # net, initial_marking, final_marking = addStartEndTransitions(net, initial_marking, final_marking)
    pm4py.write_pnml(net, initial_marking, final_marking, f'data/{case_study}/exp_{n_exp}/it_0/diagram_0.pnml')

    train_accuracy = []
    val_accuracy = []

    val_fit = []
    val_prec = []
    val_simpl = []
    val_har_f_p = []

    it_n = 0
    graph_is_updated = True

    while it_n <= max_number_iterations and graph_is_updated:
        net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/exp_{n_exp}/it_{it_n}/diagram_{it_n}.pnml')

        fitness = fitness_evaluator.apply(real_val, net, initial_marking, final_marking)
        fit = fitness['averageFitness']
        prec = precision_evaluator.apply(real_val, net, initial_marking, final_marking)
        simpl = simplicity_evaluator.apply(net)
        har_p_f = 2*fit*prec/(fit+prec)

        val_fit.append(fit)
        val_prec.append(prec)
        val_simpl.append(simpl)
        val_har_f_p.append(har_p_f)

        print(f'It. {it_n}: val_fitness {round(fit, 4)}')
        print(f'It. {it_n}: val_precision {round(prec, 4)}')
        print(f'It. {it_n}: val_simplicity {round(simpl, 4)}')
        print(f'It. {it_n}: val_harmonic_mean_fit_prec {round(har_p_f, 4)}')
        
        Simulator = StochasticPetriNetSimulator(net, initial_marking, final_marking, log=real_train)

        log_sim = Simulator.simulate(len(real_train) + len(real_val))

        data_creator = CreateDataset(real_train, real_val, log_sim)

        discriminator = Discriminator(data_creator, catboost_iterations=100, catboost_depth=3, catboost_lr=1e-1)
        discriminator.fit()

        acc_train, acc_val = discriminator.eval_accuracy()
        print(f'It. {it_n}: train_accuracy {round(acc_train, 4)} --- val_accuracy {round(acc_val, 4)}')
        train_accuracy.append(acc_train)
        val_accuracy.append(acc_val)

        # recommendations
        if save_shap:
            recommendations = discriminator.recommendations(show=show_shap, save_to=f'plots/{case_study}/exp_{n_exp}/shap_{it_n}.png')
        else:
            recommendations = discriminator.recommendations(show=show_shap)

        net, initial_marking, final_marking = pm4py.read_pnml(f'data/{case_study}/exp_{n_exp}/it_{it_n}/diagram_{it_n}.pnml')
        net, graph_is_updated = updateModel(net, initial_marking, recommendations, method, log=real_train, final_marking=final_marking)

        os.mkdir(f'data/{case_study}/exp_{n_exp}/it_{it_n+1}')
        pm4py.write_pnml(net, initial_marking, final_marking, f'data/{case_study}/exp_{n_exp}/it_{it_n+1}/diagram_{it_n+1}.pnml')
        
        it_n += 1
        print('\n')


    history = {
        'train_accuracy': train_accuracy, 
        'validation_accuracy': val_accuracy, 
        'validation_fitness': val_fit,
        'validation_precision': val_prec,
        'validation_simplicity': val_simpl,
        'validation_harmonic_mean_fit_prec': val_har_f_p
        }

    file = open(f"data/{case_study}/exp_{n_exp}/history.json", "w")
    json.dump(history, file)
    file.close()

    i = history['validation_harmonic_mean_fit_prec'][1:].index(max(history['validation_harmonic_mean_fit_prec'][1:])) + 1
    net, im, fm = pm4py.read_pnml(f'data/{case_study}/exp_{n_exp}/it_{i}/diagram_{i}.pnml')
    pm4py.view_petri_net(net, im, fm)

    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, 'o-', label='train accuracy')
    plt.plot(val_accuracy, 'o-', label='validation accuracy')
    plt.xticks(range(len(train_accuracy)))
    plt.title('Model Accuracy per iteration')
    plt.legend()
    plt.savefig(f'plots/{case_study}/exp_{n_exp}/history_accuracy.png')
    
    plt.figure()
    plt.plot(range(len(val_fit)), val_fit, 'o-', label='val fitness')
    plt.plot(range(len(val_fit)), val_prec, 'o-', label='val precision')
    plt.xticks(range(len(val_fit)))
    plt.title('Validation Fitness and Precision per iteration')
    plt.legend()
    plt.savefig(f'plots/{case_study}/exp_{n_exp}/history_val_fit_prec.png')
    
    plt.figure()
    plt.plot(range(len(val_fit)), val_har_f_p, 'o-')
    plt.title('Validation Harmonic Mean of Fitness and Precision per iteration')
    plt.xticks(range(len(val_fit)))
    plt.savefig(f'plots/{case_study}/exp_{n_exp}/history_val_har.png')
    
    plt.figure()
    plt.plot(range(len(val_fit)), val_simpl, 'o-')
    plt.title('Validation Simplicity per iteration')
    plt.xticks(range(len(val_fit)))
    plt.savefig(f'plots/{case_study}/exp_{n_exp}/history_val_sim.png')

    return net, im, fm, history



if __name__ == "__main__":
    run_repair(case_study, method, save_shap, show_shap, max_number_iterations)