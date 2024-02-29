from run import run_repair
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from tqdm import tqdm

case_studies = [
    'Purchasing',
    'Purchasing_25',
    'Purchasing_50',
    'BPI12',
    'BPI12_25',
    'BPI12_50',
    'BPI12_100',
    'BPI17',
    'BPI17_25',
    'BPI17_50',
    'BPI17_75',
    'BPI17_100',
    'RoadTrafficFines_25',
    'RoadTrafficFines_50',
    'RoadTrafficFines_75',
    'RoadTrafficFines_100',
    'Sepsis_25',
    'Sepsis_50',
    'Sepsis_75'
]

if __name__ == "__main__":

    log_file_path = 'experiment_log.txt'
    with open(log_file_path, 'w') as log_file:
        pass

    with open(log_file_path, 'a') as log_file:
        log_file.write('Experiment Log\n')
        log_file.write('Description: BPM Conference 2024 experiments.\n\n\n')

    for case_study in tqdm(case_studies):
        print('\n')
        print('------------------------ ', case_study, ' ------------------------')

        # our framework
        net, im, fm, _ = run_repair(case_study, greedy_method=False, save_shap=True, show_shap=False, max_number_iterations=15)
        real_test = xes_importer.apply(f'data/{case_study}/logTest.xes')
        fitness = fitness_evaluator.apply(real_test, net, im, fm)
        fit = fitness['averageFitness']
        prec = precision_evaluator.apply(real_test, net, im, fm)
        simpl = simplicity_evaluator.apply(net)
        har_p_f = 2*fit*prec/(fit+prec)

        # our greedy framework
        h_net, h_im, h_fm, _ = run_repair(case_study, greedy_method=True, save_shap=True, show_shap=False, max_number_iterations=15)
        h_fitness = fitness_evaluator.apply(real_test, h_net, h_im, h_fm)
        h_fit = h_fitness['averageFitness']
        h_prec = precision_evaluator.apply(real_test, h_net, h_im, h_fm)
        h_simpl = simplicity_evaluator.apply(h_net)
        h_har_p_f = 2*h_fit*h_prec/(h_fit+h_prec)

        # starting model
        net_0, im_0, fm_0 = pm4py.read_pnml(f'data/{case_study}/exp_1/it_0/diagram_0.pnml')
        fitness_0 = fitness_evaluator.apply(real_test, net_0, im_0, fm_0)
        fit_0 = fitness_0['averageFitness']
        prec_0 = precision_evaluator.apply(real_test, net_0, im_0, fm_0)
        simpl_0 = simplicity_evaluator.apply(net_0)

        # sota
        sota_net, sota_im, sota_fm = pm4py.read_pnml(f'data/{case_study}/sota/repair.pnml')
        sota_fit_ = fitness_evaluator.apply(real_test, sota_net, sota_im, sota_fm)
        try:
            sota_sound = True
            sota_fit = sota_fit_['averageFitness']
        except:
            sota_sound = False
            sota_fit = sota_fit_['average_trace_fitness']
        sota_prec = precision_evaluator.apply(real_test, sota_net, sota_im, sota_fm)
        sota_sim = simplicity_evaluator.apply(sota_net)

        print('\nStarting Fitness: ', round(fit_0, 2))
        print('Starting Precision: ', round(prec_0, 2))
        print('Starting Harmonic Mean (FP): ', round(2*fit_0*prec_0/(fit_0+prec_0), 2))
        print('Starting Simplicity: ', round(simpl_0, 2))

        print('\nSOTA Fitness: ', round(sota_fit, 2))
        print('SOTA Precision: ', round(sota_prec, 2))
        print('SOTA Harmonic Mean (FP): ', round(2*sota_fit*sota_prec/(sota_fit+sota_prec), 2))
        print('SOTA Simplicity: ', round(sota_sim, 2))

        print('\nTest Fitness: ', round(fit, 2))
        print('Test Precision: ', round(prec, 2))
        print('Test Harmonic Mean (FP): ', round(har_p_f, 2))
        print('Test Simplicity: ', round(simpl, 2))

        print('\n[Greedy] Test Fitness: ', round(h_fit, 2))
        print('[Greedy] Test Precision: ', round(h_prec, 2))
        print('[Greedy] Test Harmonic Mean (FP): ', round(h_har_p_f, 2))
        print('[Greedy] Test Simplicity: ', round(h_simpl, 2))

        with open(log_file_path, 'a') as log_file:
            log_file.write(f'\n{case_study} Experiment Results:\n')
            log_file.write('\nStarting Fitness: ' + str(round(fit_0, 2)))
            log_file.write('\nStarting Precision: ' + str(round(prec_0, 2)))
            log_file.write('\nStarting Harmonic Mean (FP): ' + str(round(2*fit_0*prec_0/(fit_0+prec_0), 2)))
            log_file.write('\nStarting Simplicity: '+ str(round(simpl_0, 2)))

            if not sota_sound:
                log_file.write('\n\nSOTA model is not sound: Fitness and Precision are computed using TOKEN BASED METHOD')
                log_file.write('\nSOTA Fitness: ' + str(round(sota_fit, 2)))
            else:
                log_file.write('\n\nSOTA Fitness: ' + str(round(sota_fit, 2)))
            log_file.write('\nSOTA Precision: ' + str(round(sota_prec, 2)))
            log_file.write('\nSOTA Harmonic Mean (FP): ' + str(round(2*sota_fit*sota_prec/(sota_fit+sota_prec), 2)))
            log_file.write('\nSOTA Simplicity: ' + str(round(sota_sim, 2)))

            log_file.write('\n\nTest Fitness: ' + str(round(fit, 2)))
            log_file.write('\nTest Precision: ' + str(round(prec, 2)))
            log_file.write('\nTest Harmonic Mean (FP): ' + str(round(har_p_f, 2)))
            log_file.write('\nTest Simplicity: ' + str(round(simpl, 2)))

            log_file.write('\n\n[Greedy] Test Fitness: ' + str(round(h_fit, 2)))
            log_file.write('\n[Greedy] Test Precision: ' + str(round(h_prec, 2)))
            log_file.write('\n[Greedy] Test Harmonic Mean (FP): ' + str(round(h_har_p_f, 2)))
            log_file.write('\n[Greedy] Test Simplicity: ' + str(round(h_simpl, 2)))
            log_file.write('\n\n\n')

