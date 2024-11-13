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
    'Sepsis_75',
    'RoadTrafficFines',
    'Sepsis',
    'RoadTrafficFines_mod_10',
    'Sepsis_mod_10'
]

if __name__ == "__main__":

    log_file_path = 'experiment_log_ext_double_label_sepsis_mod_10.txt'
    with open(log_file_path, 'w') as log_file:
        pass

    with open(log_file_path, 'a') as log_file:
        log_file.write('Experiment Log\n')
        log_file.write('Description: Process Science (Extension BPM 2024) experiments.\n\n\n')

    for case_study in tqdm(case_studies):
        print('\n')
        print('------------------------ ', case_study, ' ------------------------')

        # our framework
        net, im, fm, _ = run_repair(case_study, method='alignments', save_shap=True, show_shap=False, max_number_iterations=15)
        real_test = xes_importer.apply(f'data/{case_study}/logTest.xes')
        fitness = fitness_evaluator.apply(real_test, net, im, fm)
        fit = fitness['averageFitness']
        prec = precision_evaluator.apply(real_test, net, im, fm)
        simpl = simplicity_evaluator.apply(net)
        har_p_f = 2*fit*prec/(fit+prec)

        print('\n[Align] Test Fitness: ', round(fit, 2))
        print('[Align] Precision: ', round(prec, 2))
        print('[Align] Harmonic Mean (FP): ', round(har_p_f, 2))
        print('[Align] Simplicity: ', round(simpl, 2))


        with open(log_file_path, 'a') as log_file:
            log_file.write(f'\n{case_study} Experiment Results:\n')
            log_file.write('\n[Align] Test Fitness: ' + str(round(fit, 2)))
            log_file.write('\n[Align] Test Precision: ' + str(round(prec, 2)))
            log_file.write('\n[Align] Test Harmonic Mean (FP): ' + str(round(har_p_f, 2)))
            log_file.write('\n[Align] Test Simplicity: ' + str(round(simpl, 2)))
            log_file.write('\n\n\n')