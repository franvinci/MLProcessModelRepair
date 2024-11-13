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
import shutil



def apply(log_path, model_path, method='greedy', max_number_iterations=15, verbose=True):

    log_real = xes_importer.apply(log_path)
    try:
        log_real = pm4py.filter_event_attribute_values(log_real, 'lifecycle:transition', ['complete'], level="event", retain=True)
    except:
        pass        
    log_real = addStartEndEvents(log_real)

    if os.path.exists('mlrepair_info'):
        shutil.rmtree('mlrepair_info')

    os.mkdir('mlrepair_info')

    real_train, real_val, _ = splitRealLog(log_real, split_size = (0.6, 0.2, 0.2), split_temporal = True, save_to=f'mlrepair_info')

    net, initial_marking, final_marking = pm4py.read_pnml(model_path)
    
    # check if it is sound
    is_sound = woflan.apply(net, initial_marking, final_marking, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                                             woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                                             woflan.Parameters.RETURN_DIAGNOSTICS: False})
    if not is_sound:
        print('Petri Net must be sound.')
        return None
    
    os.mkdir('mlrepair_info/it_0')

    net, initial_marking, final_marking = addStartEndTransitions(net, initial_marking, final_marking)
    pm4py.write_pnml(net, initial_marking, final_marking, f'mlrepair_info/it_0/diagram_0.pnml')

    train_accuracy = []
    val_accuracy = []

    val_fit = []
    val_prec = []
    val_simpl = []
    val_har_f_p = []

    it_n = 0
    graph_is_updated = True


    while it_n <= max_number_iterations and graph_is_updated:
        net, initial_marking, final_marking = pm4py.read_pnml(f'mlrepair_info/it_{it_n}/diagram_{it_n}.pnml')

        fitness = fitness_evaluator.apply(real_val, net, initial_marking, final_marking)
        fit = fitness['averageFitness']
        prec = precision_evaluator.apply(real_val, net, initial_marking, final_marking)
        simpl = simplicity_evaluator.apply(net)
        har_p_f = 2*fit*prec/(fit+prec)

        val_fit.append(fit)
        val_prec.append(prec)
        val_simpl.append(simpl)
        val_har_f_p.append(har_p_f)

        if verbose:
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
        if verbose:
            print(f'It. {it_n}: train_accuracy {round(acc_train, 4)} --- val_accuracy {round(acc_val, 4)}')
        train_accuracy.append(acc_train)
        val_accuracy.append(acc_val)

        recommendations = discriminator.recommendations(show=False)

        net, initial_marking, final_marking = pm4py.read_pnml(f'mlrepair_info/it_{it_n}/diagram_{it_n}.pnml')
        net, graph_is_updated = updateModel(net, initial_marking, recommendations, method, log=real_train, final_marking=final_marking)

        os.mkdir(f'mlrepair_info/it_{it_n+1}')
        pm4py.write_pnml(net, initial_marking, final_marking, f'mlrepair_info/it_{it_n+1}/diagram_{it_n+1}.pnml')
        
        it_n += 1
        if verbose:
            print('\n')


    history = {
        'train_accuracy': train_accuracy, 
        'validation_accuracy': val_accuracy, 
        'validation_fitness': val_fit,
        'validation_precision': val_prec,
        'validation_simplicity': val_simpl,
        'validation_harmonic_mean_fit_prec': val_har_f_p
        }

    file = open(f"mlrepair_info/history.json", "w")
    json.dump(history, file)
    file.close()

    i = history['validation_harmonic_mean_fit_prec'][1:].index(max(history['validation_harmonic_mean_fit_prec'][1:])) + 1
    net, im, fm = pm4py.read_pnml(f'mlrepair_info/it_{i}/diagram_{i}.pnml')

    return net, im, fm, history